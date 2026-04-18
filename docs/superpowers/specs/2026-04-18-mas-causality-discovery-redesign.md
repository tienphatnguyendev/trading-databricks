# MAS Causality Discovery Redesign

**Status:** Design approved, awaiting implementation plan
**Date:** 2026-04-18
**Owner:** aaronng
**Supersedes:** parts of `2026-04-14-llm-judge-validator-design.md` (judge usage extended online)

---

## Problem

The current MAS causality pipeline collapses extraction into a single LLM call (`causality_extractor`) wrapped by a few RAG and routing nodes. Four entangled gaps limit causal-reasoning quality:

1. **Monolithic extractor.** One prompt has to read the article, recall vocabulary, generate hypotheses, ground them in evidence, and pick a final answer. Failure modes are opaque — when the answer is wrong, we cannot tell which step broke.
2. **No reflection or iteration.** The extractor's first answer is the final answer. Confidence calibration drifts and there is no inner loop that responds to weak evidence.
3. **Shallow RAG.** Few-shot retrieval only surfaces labeled exemplars; the model never sees the underlying news corpus or what actually happened in the market after similar past events.
4. **Disconnected news firehose.** The full news corpus (continuously ingested) is not indexed for retrieval — only the curated eval/few-shot set is searchable.

These together push the system toward "extraction" (slot-filling against a fixed taxonomy) rather than "discovery" (reasoning about novel causal structure).

---

## Goal

Redesign the inner causality node into a multi-step reasoning workflow that:

- **Decomposes** evidence extraction, hypothesis generation, judging, and reflection into independently observable steps.
- **Reuses** the LLM Judge infrastructure (designed in `2026-04-14-llm-judge-validator-design.md`) as an online reflection signal, not just an offline evaluator.
- **Deepens RAG** by joining retrieved precedent news to actual market outcomes (`ohlc_gold` / `dollar_bars` ±24h windows).
- **Taps the news firehose** via a new Vector Search index over the full continuously-ingested corpus.
- Stays within a quality-first latency budget (p50 ≤ 20s).

Non-goals: replacing `mechanism_validator` (whitelist + escape-hatch behavior is preserved), changing the soft-closed vocabulary contract, or reworking the few-shot retrieval that precedes the inner pipeline.

---

## Architecture

### Topology

The current outer graph is preserved:

```text
context_retriever → few_shot_retriever → uncertainty_router
   → [route_article: fast / standard / careful]
      → causality_extractor (REPLACED)
   → mechanism_validator → END
```

The `causality_extractor` node is replaced by a 5-node sub-pipeline with bounded reflection:

```text
evidence_extractor
   → rag_retriever
   → hypothesis_generator   ←──────────┐
   → judge_panel  (parallel fan-out)   │
   → reflector ──────────────────────┘ (loop back to hypothesis_generator
                                         while judge_passed=False
                                         and reflection_count < max_reflections)
   → mechanism_validator (existing)
```

Reflection is bounded to `max_reflections=1` by default (so worst-case is 2 hypothesis_generator calls per article).

### Components

| Node | Inputs (state keys read) | Outputs (state keys written) | Externals |
|------|--------------------------|------------------------------|-----------|
| `evidence_extractor` | `headline`, `body` | `evidence: list[dict]`, `evidence_status` | LLM |
| `rag_retriever` | `evidence`, `symbol`, `timestamp` | `precedents: list[dict]`, `precedent_status` | Vector Search (`news_firehose_idx`), OHLC join (`ohlc_gold` / `dollar_bars`) |
| `hypothesis_generator` | `headline`, `body`, `evidence`, `precedents`, `few_shot_examples`, `judge_failures` (if reflecting) | `extraction: dict`, `extraction_status` | LLM |
| `judge_panel` | `headline`, `body`, `extraction` | `judge_results: dict`, `judge_failures: dict \| None` | LLM x 3 (parallel via LangGraph Send API) |
| `reflector` | `judge_results`, `reflection_count` | routing decision, `reflection_count++`, `extraction_status` | none (pure logic) |

State immutability principle: each node only writes to keys named above. Upstream keys are never mutated, so MLflow traces remain replayable per-node.

### New state keys (added to `agents/state.py`)

```python
evidence: list[dict]              # default: []
evidence_status: str              # "ok" | "empty" | "failed"
precedents: list[dict]            # default: []
precedent_status: str             # "ok" | "no_match" | "unavailable"
extraction_status: str            # see error model
judge_results: dict | None
judge_failures: dict | None       # populated when judges fail, fed back on reflection
reflection_count: int             # default: 0
```

### New external resources

- **Vector Search index:** `news_firehose_idx` — indexes the continuously-ingested news corpus on `(headline, body, timestamp, symbol)`. Refresh cadence and embedding model are deferred to the implementation plan; the index identifier is added to `common/uc_identifiers.py`.
- **Review table:** `04_eval.mechanism_candidates_review` — already implied by the existing escape-hatch path; identifier added to `common/uc_identifiers.py` if not present.

---

## Data Flow (worked example)

Article: "BlackRock files for spot Ether ETF" (symbol=ETHUSDT).

1. **Entry state** carries `headline`, `body`, `symbol`, `extraction_path="standard"`, `few_shot_examples`, `reflection_count=0`.
2. **`evidence_extractor`** writes structured JSON: `[{actor: "BlackRock", action: "files", object: "spot Ether ETF", span: "BlackRock files for...", timestamp_hint: null}]`, `evidence_status="ok"`.
3. **`rag_retriever`** queries `news_firehose_idx` for similar past events, joins each hit to `ohlc_gold` ±24h, writes `precedents=[{headline: "Fidelity files Bitcoin ETF", body_excerpt: ..., timestamp: ..., return_24h: +0.043, direction: "up"}, ...]`, `precedent_status="ok"`.
4. **`hypothesis_generator`** prompt: system message contains vocab rules, taxonomy, whitelist, few-shot exemplars; user message contains headline + body + evidence + precedent table. On reflection retries, a footer is appended with the previous `judge_failures`. Writes `extraction={treatment_class: "regulatory_approval", mechanism: "etf_inflow_demand", expected_direction: "up", confidence: 0.78, reasoning: "..."}`, `extraction_status="ok"`.
5. **`judge_panel`** fans out via LangGraph `Send` API to three judges (direction, mechanism, grounding) in parallel. Strict-AND aggregation. Writes `judge_results` and (on failure) `judge_failures`.
6. **`reflector`** reads `judge_results`. If passed → routes to `mechanism_validator`. If failed and `reflection_count < max_reflections` → increments counter, routes back to `hypothesis_generator` with `judge_failures` attached. If failed and budget exhausted → routes to `mechanism_validator` with `extraction_status="judge_failed_accepted"`.
7. **`mechanism_validator`** (unchanged) applies the whitelist, persists any `mechanism_candidate` to `04_eval.mechanism_candidates_review`, returns terminal state.

---

## Error Handling

Principle: **degrade explicitly, never silently.** Every failure writes a structured signal that downstream nodes and MLflow traces can read.

- **`evidence_extractor`:** malformed/empty JSON → `evidence=[]`, `evidence_status="empty"`, continue. LLM call failure → retry once; on second failure, `extraction_status="failed"` and skip to END.
- **`rag_retriever`:** Vector Search timeout (>5s) or zero results → `precedents=[]`, `precedent_status` set accordingly, continue. Per-precedent OHLC join failure → drop that precedent, keep others.
- **`hypothesis_generator`:** JSON parse failure → retry once with stricter prompt suffix. Vocab violation without `mechanism_candidate` → treated as parse failure. Two failures → `extraction_status="parse_failed"`, skip to END.
- **`judge_panel`:** individual judge failure → judge returns `passed=False, score=None, reason="judge_unavailable"`. Strict-AND means missing judge fails the panel and triggers reflection. All three failing → `judge_results={}`, `extraction_status="unjudged"`, skip reflection.
- **`reflector`:** budget exhausted → accept current extraction with `extraction_status="judge_failed_accepted"`, route to validator. The flagged status surfaces in eval dashboards for triage.
- **`mechanism_validator`:** review table write failure → log + `validator_status="persist_failed"`, do not fail the extraction (caller can recover from returned state).

Single triage field: `extraction_status` ∈ {`ok`, `ok_after_reflection`, `judge_failed_accepted`, `unjudged`, `parse_failed`, `failed`}.

---

## Latency

Quality-first budget: p50 ≤ 20s end-to-end for the inner sub-pipeline. Worst case (one reflection) doubles the hypothesis_generator + judge_panel cost. Optimization deferred — measure first.

---

## Out of Scope

- Replacing or restructuring the outer graph (`context_retriever`, `few_shot_retriever`, `uncertainty_router`, `mechanism_validator`).
- Changing the soft-closed vocabulary or whitelist.
- Live trading impact analysis (this is a signal-generation change only).
- News firehose indexing job design (the index name is reserved here; the indexing pipeline is a separate ticket).
- Latency benchmarking and tuning.
