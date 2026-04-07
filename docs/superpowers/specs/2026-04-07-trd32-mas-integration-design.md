# TRD-32 — MAS Integration Design Spec

**Date:** 2026-04-07
**Status:** Approved
**Linear:** TRD-32 — [W4] Wire complete 6-node LangGraph pipeline + UC function tools + SDP integration

## Context

TRD-32 is the integration step for TRD-6 (Build Gen-AI Agent Layer). All 5 LangGraph nodes
(context_retriever, few_shot_retriever, uncertainty_router, causality_extractor,
mechanism_validator) are individually implemented and merged (TRD-27–31). This ticket
assembles them into a compiled StateGraph, adds the missing UC tool, and wires the pipeline
into the SDP gold materialized view.

## Architecture

### Approach: Closure-bound graph (Approach A)

`build_mas_graph(*, config: MASConfig)` uses lambda closures to bind `MASConfig` to each
existing node function. Zero changes to "done" node files. `_mechanism_validator_node`
(module-private in graph.py) bridges `validate_extraction()`'s 5-arg signature to LangGraph's
`(state) -> dict` contract.

### Graph topology

```
ENTRY → context_retriever → few_shot_retriever → uncertainty_router
       ↓ conditional: route_article (V1: always → causality_extractor)
       causality_extractor → mechanism_validator → END
```

`route_article` returns `"causality_extractor"` for all paths (fast/standard/careful).
The `extraction_path` field in state drives path-based behaviour inside `causality_extractor`.
The conditional edge is kept for future extensibility.

### SDP pattern

Follows `ticks/pipeline.py`: logic in `agents/gold.py`, thin wrapper in `agents/pipeline.py`.

## State extension

Add to `MASState` TypedDict:
- `body: str` — article body text (input alongside headline)
- `extraction: dict | None` — LLM output from causality_extractor
- `validation_status: str | None` — "accepted" | "rejected_*"
- `rejected_mechanism_raw: str | None` — raw mechanism string when rejected
- `evidence_relevance_score: float | None` — Module B cosine score

## UC Tools

`tools.py` already has `retrieve_few_shot_chains()` and `fetch_market_context()`.
Add `fetch_mechanism_whitelist()` — reads `gold.mechanism_whitelist` as JSON array string.

## Gold Materialized View

`agents/gold.py :: news_causal_features_transform(spark, *, catalog, gatekeeper_config)`:
1. Gatekeeper pandas UDF calling `is_crypto_relevant()` per row
2. Filter `{catalog}.silver.news_cleaned`
3. `ai_query('mas-news-causal', to_json(struct(headline, body, published_at)))` column
4. Expand JSON causal_result into CAUSAL_FEATURES_SCHEMA fields

## Acceptance Criteria

- `build_mas_graph()` compiles without error
- End-to-end: 10 sample articles through full graph → outputs valid MASState
- All 3 UC function tools callable independently
- `uv run pytest tests/agents/test_graph.py` passes (mock LLM + mock VS)
