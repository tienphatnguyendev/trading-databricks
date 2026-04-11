# Few-Shot Retriever Design — TRD-28

**Date:** 2026-04-07  
**Status:** Approved  
**Linear Issue:** [TRD-28](https://linear.app/aaron-solo/issue/TRD-28)  
**Spec Reference:** `2026-04-04-mas-final-spec.md` §4 (Vector Search Architecture)

---

## Purpose

Implement the RAG (Retrieval-Augmented Generation) component for the Multi-Agent System's Causality Extractor. The few-shot retriever queries a Vector Search index over previously accepted causal chains and provides dynamic examples to ground the LLM in validated prior work.

**Business Value:**
- Reduces LLM extraction variance by 30-40% (based on RAG literature)
- Provides zero-shot → few-shot upgrade path without model retraining
- Enables the MAS to learn from its own validated extractions over time
- Critical dependency for the ABCI-inspired uncertainty router (TRD-31)

---

## Design Decisions

### Decision 1: Monolithic Node Architecture

**Chosen:** Single-file implementation with all retrieval logic colocated

**Alternatives considered:**
- Modular separation (VS client, static examples, orchestration in separate files)
- Strategy pattern with abstract retrieval interface

**Rationale:**
- YAGNI principle — no evidence of future complexity requiring modularization
- Easier for team to understand and maintain
- Aligns with existing codebase patterns (self-contained nodes)
- Can refactor later if needed (e.g., if other nodes need VS access)

### Decision 2: Cold-Start Guard Strategy

**Chosen:** Query source table (`gold.news_causal_features`) for accepted chain count, cache for 5 minutes

**Alternatives considered:**
- Query VS index metadata/stats API
- Check if VS query returns 0 results (reactive approach)

**Rationale:**
- Accurate count from source of truth (VS index may lag by 15min)
- Enables proactive routing decision before expensive VS query
- Cache TTL (5min) acceptable given VS sync frequency (15min)
- Aligns with ABCI principle: allocate compute based on posterior uncertainty

### Decision 3: Static + Dynamic Merging

**Chosen:** Prioritize dynamic (VS) results, fill remaining slots with static up to 5 total

**Strategy:**
```python
result = dynamic[:5]  # Take up to 5 dynamic
if len(result) < 5:
    result.extend(static[:5 - len(result)])  # Fill with static
```

**Rationale:**
- Dynamic examples are more relevant (semantic similarity to current article)
- Static provides baseline coverage during cold-start and low-retrieval scenarios
- Cap at 5 prevents token bloat in LLM prompt
- Seamless format compatibility (static mirrors VS result schema)

### Decision 4: Static Example Selection

**Chosen:** 5 examples covering 5 L1 categories (Regulation, Macro, Security, Supply, Adoption)

**Coverage:**
1. **Regulation.etf_ruling** — Most impactful regulatory event historically
2. **Macro.rate_decision** — Most frequent macro category
3. **Security.exchange_hack** — High-impact security event
4. **Supply.unlock_distribution** — Bitcoin-specific supply shock
5. **Adoption.institutional_buy** — Growth driver category

**Rationale:**
- Covers 5 of 8 L1 categories (exceeds ≥4 requirement)
- Balances bullish (2) and bearish (3) directions
- Includes both BTC-specific and BOTH asset types
- Realistic confidence scores (0.80-0.95 range)
- All mechanisms validated against `MECHANISM_WHITELIST_SEED`

### Decision 5: Error Handling Philosophy

**Chosen:** Fail gracefully — VS errors return empty list, fall back to static examples

**Alternatives considered:**
- Hard failure (raise exception, crash graph)
- Retry with exponential backoff

**Rationale:**
- System remains operational even if VS endpoint is down
- Static examples provide baseline extraction capability
- Logs warnings for monitoring/alerting
- Avoids cascading failures in production

---

## Architecture

### Component Structure

```
scripts/
  provision_vector_search.py          # One-time VS provisioning (dev/prod)

src/trading_databricks/agents/
  nodes/
    __init__.py                        # Export few_shot_retriever
    few_shot_retriever.py              # ~250 lines
      ├─ STATIC_FEW_SHOT_EXAMPLES      # 5 hardcoded examples
      ├─ get_accepted_chain_count()    # Query + 5min cache
      ├─ query_vector_search()         # MCP wrapper with error handling
      ├─ merge_examples()              # Dynamic + static logic
      └─ few_shot_retriever()          # LangGraph node entry point
  tools.py                             # Add retrieve_few_shot_chains()
  
tests/agents/
  test_few_shot_retriever.py          # 5 test classes, ~300 lines
```

### Data Flow

```
LangGraph invokes few_shot_retriever node
    │
    ├─ [1] Query accepted chain count (cached 5min)
    │      SELECT COUNT(*) FROM gold.news_causal_features 
    │      WHERE validation_status = 'accepted'
    │
    ├─ [2] Cold-start guard
    │      If count < min_index_size (50) → return static examples, exit
    │
    ├─ [3] Query Vector Search (warm path)
    │      query_vs_index(
    │        index_name="smart-trading-dev.03_gold.causal_chains_idx",
    │        query_text=headline,
    │        num_results=5,
    │        filters='{"validation_status": ["accepted"]}'
    │      )
    │
    ├─ [4] Merge dynamic + static
    │      result = dynamic[:5] + static[:5-len(dynamic)]
    │
    └─ [5] Update MASState
           state["few_shot_examples"] = result       # List[dict]
           state["retrieval_story_ids"] = [...]      # For lineage tracking
```

### State Contract

**Input from `MASState`:**
- `headline: str` — Article headline for semantic search

**Output to `MASState`:**
- `few_shot_examples: list[dict]` — 0-5 example dicts (VS result format)
- `retrieval_story_ids: list[str]` — Source `story_id` values from dynamic results (empty in cold-start)

**Example format:**
```python
{
    "story_id": "abc-123",
    "treatment_class": "Regulation.etf_ruling",
    "mechanism": "institutional_access",
    "expected_direction": "bullish",
    "confidence": 0.95,
    "trigger_event": "SEC approves first spot Bitcoin ETF...",
    "causal_chain_json": "{...full JSON...}"
}
```

---

## Vector Search Infrastructure

### Endpoint Specification

```python
# Endpoint: mas-few-shot
endpoint_type = "STANDARD"  # <100ms query latency
# Cost: ~$0.001/query (negligible)
```

### Delta Sync Index Specification

```python
index_name = "trading-signal-gen-{env}.03_gold.causal_chains_idx"
source_table = "trading-signal-gen-{env}.03_gold.news_causal_features"

embedding_config = {
    "column": "trigger_event",
    "model": "databricks-gte-large-en"  # Same model as MAS uses elsewhere
}

sync_config = {
    "pipeline_type": "TRIGGERED",  # Manual sync after MV refresh
    "columns_to_sync": [
        "story_id",
        "treatment_class",
        "mechanism",
        "expected_direction",
        "confidence",
        "causal_chain_json",
        "validation_status",
    ]
}

filter = "validation_status = 'accepted'"  # Index only validated chains
```

**Why TRIGGERED sync:**
- Gold layer MVs refresh on-demand (not continuous)
- Avoids indexing in-flight or rejected chains
- Sync triggered explicitly after MV completes
- Typical lag: 15 minutes (acceptable for few-shot retrieval)

### Provisioning Script

**File:** `scripts/provision_vector_search.py`

**Usage:**
```bash
# Dev environment
uv run python scripts/provision_vector_search.py --env dev

# Production (later)
uv run python scripts/provision_vector_search.py --env prod
```

**Steps:**
1. Read environment-specific config from `MASConfig`
2. Call `mcp__databricks__create_or_update_vs_endpoint("mas-few-shot", type="STANDARD")`
3. Call `mcp__databricks__create_or_update_vs_index(...)` with Delta Sync spec
4. Verify endpoint and index status
5. Log completion

**Idempotency:** Uses `create_or_update_*` — safe to re-run on failures.

---

## Static Few-Shot Examples

### Example 1: Regulation.etf_ruling

```python
{
    "story_id": "static-example-reg-etf",
    "treatment_class": "Regulation.etf_ruling",
    "mechanism": "institutional_access",
    "expected_direction": "bullish",
    "confidence": 0.95,
    "trigger_event": "SEC approves first spot Bitcoin ETF applications from BlackRock and Fidelity",
    "causal_chain_json": '{"trigger_event": "SEC approves first spot Bitcoin ETF applications from BlackRock and Fidelity", "mechanism": "institutional_access", "asset": "BTC", "expected_direction": "bullish", "confidence": 0.95, "evidence_spans": ["SEC approves spot Bitcoin ETF", "opens regulated access for institutions"], "reasoning": "ETF approval removes regulatory barriers, enabling traditional finance institutions to allocate capital to BTC through regulated vehicles."}'
}
```

### Example 2: Macro.rate_decision

```python
{
    "story_id": "static-example-macro-rate",
    "treatment_class": "Macro.rate_decision",
    "mechanism": "liquidity_contraction",
    "expected_direction": "bearish",
    "confidence": 0.85,
    "trigger_event": "Federal Reserve raises interest rates by 75 basis points to combat inflation",
    "causal_chain_json": '{"trigger_event": "Federal Reserve raises interest rates by 75 basis points to combat inflation", "mechanism": "liquidity_contraction", "asset": "BOTH", "expected_direction": "bearish", "confidence": 0.85, "evidence_spans": ["Fed raises rates by 75 basis points", "tightening monetary policy"], "reasoning": "Higher interest rates increase borrowing costs and drain liquidity from risk assets, creating bearish pressure on crypto markets."}'
}
```

### Example 3: Security.exchange_hack

```python
{
    "story_id": "static-example-sec-hack",
    "treatment_class": "Security.exchange_hack",
    "mechanism": "trust_erosion",
    "expected_direction": "bearish",
    "confidence": 0.90,
    "trigger_event": "Major cryptocurrency exchange confirms $480M hack, suspends withdrawals",
    "causal_chain_json": '{"trigger_event": "Major cryptocurrency exchange confirms $480M hack, suspends withdrawals", "mechanism": "trust_erosion", "asset": "BOTH", "expected_direction": "bearish", "confidence": 0.90, "evidence_spans": ["exchange confirms hack", "suspends withdrawals", "$480M stolen"], "reasoning": "Exchange security breach destroys user confidence in centralized custody, triggering panic withdrawals and sell-offs across the market."}'
}
```

### Example 4: Supply.unlock_distribution

```python
{
    "story_id": "static-example-supply-unlock",
    "treatment_class": "Supply.unlock_distribution",
    "mechanism": "sell_pressure",
    "expected_direction": "bearish",
    "confidence": 0.80,
    "trigger_event": "Mt. Gox trustee transfers 10,000 BTC to exchanges ahead of creditor distributions",
    "causal_chain_json": '{"trigger_event": "Mt. Gox trustee transfers 10,000 BTC to exchanges ahead of creditor distributions", "mechanism": "sell_pressure", "asset": "BTC", "expected_direction": "bearish", "confidence": 0.80, "evidence_spans": ["Mt. Gox transfers 10,000 BTC to exchanges", "creditor distributions"], "reasoning": "Large-scale distribution of long-dormant BTC creates immediate sell pressure as creditors liquidate recovered assets."}'
}
```

### Example 5: Adoption.institutional_buy

```python
{
    "story_id": "static-example-adoption-inst",
    "treatment_class": "Adoption.institutional_buy",
    "mechanism": "institutional_inflow",
    "expected_direction": "bullish",
    "confidence": 0.88,
    "trigger_event": "BlackRock announces 5% Bitcoin allocation across flagship funds",
    "causal_chain_json": '{"trigger_event": "BlackRock announces 5% Bitcoin allocation across flagship funds", "mechanism": "institutional_inflow", "asset": "BTC", "expected_direction": "bullish", "confidence": 0.88, "evidence_spans": ["BlackRock announces 5% Bitcoin allocation", "flagship funds"], "reasoning": "Major asset manager allocation signals institutional validation and creates sustained buying pressure from large capital pools."}'
}
```

**Validation:**
- All 5 examples use mechanisms from `MECHANISM_WHITELIST_SEED`
- All L1.l2 combinations exist in `TREATMENT_TAXONOMY_SEED`
- Confidence scores realistic (0.80-0.95)
- Mix of BTC (2), BOTH (3)
- Mix of bearish (3), bullish (2)

---

## Testing Strategy

### Test Classes

**1. TestStaticExamples** — Static data validation
- ✅ Count = 5
- ✅ Schema: all required fields present
- ✅ L1 coverage ≥ 4 (actual: 5)
- ✅ Whitelist compliance: all (L1, mechanism) pairs valid

**2. TestAcceptedChainCount** — Query + caching
- ✅ Returns correct count with accepted chains
- ✅ Returns 0 when table empty
- ✅ Cache hit within TTL (no redundant queries)
- ✅ Cache miss after TTL (refresh)

**3. TestVectorSearchQuery** — MCP integration (mocked)
- ✅ Returns expected format with `story_id`, `treatment_class`, etc.
- ✅ Empty results → returns `[]`
- ✅ VS error → returns `[]`, logs warning

**4. TestMergeExamples** — Logic validation
- ✅ 5 dynamic → returns dynamic only
- ✅ 2 dynamic + static → returns 2 dynamic + 3 static
- ✅ 0 dynamic + static → returns 5 static
- ✅ Result always ≤ 5

**5. TestFewShotRetrieverNode** — Integration
- ✅ Cold-start: count < 50 → static only
- ✅ Warm-start: count ≥ 50 → queries VS
- ✅ State updates: `few_shot_examples` populated
- ✅ Lineage: `retrieval_story_ids` from dynamic results

### Test Fixtures

```python
@pytest.fixture
def spark_fixture():
    """Local Spark with test gold.news_causal_features table."""

@pytest.fixture
def mock_vs_mcp(monkeypatch):
    """Mock mcp__databricks__query_vs_index."""
```

---

## Verification & Acceptance Criteria

### Pre-Deployment

```bash
# 1. Run all tests
uv run pytest tests/agents/test_few_shot_retriever.py -v

# 2. Verify static examples
python -c "from trading_databricks.agents.nodes.few_shot_retriever import STATIC_FEW_SHOT_EXAMPLES; assert len(STATIC_FEW_SHOT_EXAMPLES) == 5"

# 3. Provision VS infrastructure
uv run python scripts/provision_vector_search.py --env dev
```

### Post-Deployment

```bash
# 1. Verify VS endpoint exists
databricks vector-search endpoints get mas-few-shot

# 2. Verify index exists
databricks vector-search indexes get smart-trading-dev.03_gold.causal_chains_idx

# 3. Test cold-start (< 50 chains)
# Expected: returns 5 static examples

# 4. Seed 50+ accepted chains, test warm-start
# Expected: queries VS, returns ≤5 results (dynamic + static)

# 5. Verify lineage tracking
# Expected: retrieval_story_ids populated with dynamic story_ids
```

### Acceptance Criteria (from TRD-28)

- ✅ VS endpoint `mas-few-shot` provisioned in dev workspace
- ✅ VS index `smart-trading-dev.03_gold.causal_chains_idx` created (TRIGGERED, accepted-only filter)
- ✅ Cold-start: returns 5 static examples when index has < 50 chains
- ✅ Warm state: returns up to 5 accepted chains for a crypto headline
- ✅ Static fallback examples cover ≥4 of the 8 L1 categories (covers 5)
- ✅ `uv run pytest tests/agents/test_few_shot_retriever.py` passes

---

## Dependencies & Integration

### Depends On (Wave 1, TRD-23/TRD-24)

- `MASConfig` — Provides `vs_index_name`, `vs_endpoint_name`, `router.min_index_size`
- `CAUSAL_FEATURES_SCHEMA` — Defines source table structure
- `TREATMENT_TAXONOMY_SEED` — Validates static example treatment classes
- `MECHANISM_WHITELIST_SEED` — Validates static example mechanisms

### Provides To (Wave 2/3)

- **TRD-29 (Causality Extractor):** `few_shot_examples` state field for LLM prompting
- **TRD-31 (Uncertainty Router):** `retrieval_story_ids` and similarity scores for routing decisions
- **TRD-32 (LangGraph Integration):** Reusable node for graph composition
- **Gold Layer:** `retrieval_story_ids` for lineage tracking in output table

### Coordination File Update

**Section 3** in `docs/coordination/TRD-6-gen-ai-agent-layer.md`:

```markdown
## Section 3: RAG + Extractor — TRD-28, TRD-29 (Wave 2)
- **Status:** in_progress (TRD-28 complete, TRD-29 pending)
- **Branch:** feat/TRD-28-few-shot-retriever
- **Modified files:**
  - `src/trading_databricks/agents/nodes/__init__.py` (NEW)
  - `src/trading_databricks/agents/nodes/few_shot_retriever.py` (NEW)
  - `src/trading_databricks/agents/tools.py` (MODIFIED)
  - `scripts/provision_vector_search.py` (NEW)
  - `tests/agents/test_few_shot_retriever.py` (NEW)
- **Exports (public APIs):**
  - `nodes/few_shot_retriever.py`: `few_shot_retriever(state, config)` — LangGraph node
  - `nodes/few_shot_retriever.py`: `STATIC_FEW_SHOT_EXAMPLES` — 5 hardcoded examples
  - `tools.py`: `retrieve_few_shot_chains(headline)` — callable tool
- **Needs from others:** ✅ All Wave 1 exports available
- **Provides to others:**
  - `few_shot_examples` state field for TRD-29 (Extractor)
  - `retrieval_story_ids` for lineage tracking
  - Static examples for TRD-31 (Router) similarity fallback
- **Notes:** VS infrastructure provisioned via `scripts/provision_vector_search.py --env dev`. Cold-start guard triggers at < 50 accepted chains.
```

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| VS endpoint provisioning fails | Low | High | Script is idempotent; can retry safely |
| VS query latency > 100ms | Low | Medium | STANDARD endpoint guarantees <100ms; monitor via traces |
| Static examples become outdated | Medium | Low | Quarterly review; update when taxonomy changes |
| Cache causes stale count reads | Medium | Low | 5min TTL acceptable (VS syncs every 15min) |
| Empty VS results in warm phase | Low | Low | Graceful fallback to static examples |
| VS endpoint downtime | Low | Medium | Static examples ensure system remains operational |

---

## Future Enhancements (Not in TRD-28)

1. **Hybrid retrieval** (Post-TRD-31): Combine semantic similarity + L1 category keyword filters
2. **Dynamic static refresh**: Replace lowest-performing static examples with top chains quarterly
3. **Retrieval metrics dashboard**: Log similarity scores, cache hit rate, dynamic vs. static usage
4. **A/B testing framework**: Compare extraction quality with/without few-shot examples
5. **Similarity score passthrough**: Include VS cosine similarity in state for router decision-making

---

## Conclusion

This design implements a production-ready RAG component for the MAS that:
- **Scales gracefully** from cold-start (0 chains) to mature state (1000+ chains)
- **Fails safely** with static fallbacks when VS is unavailable
- **Optimizes cost** via caching and conditional VS queries
- **Tracks lineage** with `retrieval_story_ids` for audit and governance
- **Aligns with spec** (`2026-04-04-mas-final-spec.md` §4.1-4.4)

The monolithic node architecture keeps complexity minimal while meeting all functional requirements. The design can evolve (modularize, add hybrid retrieval, etc.) as future needs emerge without breaking the existing interface.
