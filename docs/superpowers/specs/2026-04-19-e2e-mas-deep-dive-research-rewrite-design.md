---
title: E2E MAS Deep-Dive Research Notebook Rewrite
date: 2026-04-19
status: approved
ticket: TBD
---

# E2E MAS Deep-Dive Research Notebook Rewrite

## Goal

Full rewrite of `notebooks/qa/mas/e2e_mas_deep_dive_research.py` to align with the
post-TRD-68 causality sub-pipeline (5-node architecture), eliminate dead code,
and conform to the codebase's standard Databricks notebook QA pattern.

The notebook's sole purpose: **live Databricks environment test** — run the full MAS
graph against real LLM endpoints and the real `raw_news` table, producing
MLflow-logged quality metrics for human review.

---

## Files Changed

| File | Action |
|------|--------|
| `notebooks/qa/mas/e2e_mas_deep_dive_research.py` | Full rewrite (923 → ~250 lines) |
| `notebooks/qa/mas/_scorer_helpers.py` | Extend: 4 new predicates |

---

## What Is Removed

- `OFFLINE` / synthetic-data fallback mode (dropped entirely — the notebook only
  runs live)
- §11 failure autopsy widget sandbox (MLflow trace UI covers this)
- §12 human feedback stubs
- All references to `causality_extractor` (deprecated node, replaced by 5-node
  sub-pipeline in TRD-68)
- `mas_training_dataset` in the non-existent `others` schema
- Fragile multi-cell `graph_env_ctx.__enter__()` / `graph_env_ctx.__exit__()`
  spanning 13 cells

---

## Data Source Fix

The current notebook references `smart-trading-dev.others.mas_training_dataset`
which does not exist (`SCHEMA_NOT_FOUND` confirmed via MCP). Replace with:

```
smart-trading-dev.01_bronze.raw_news
```

Since `raw_news` rows don't carry a crypto symbol, inject `symbol = SYMBOL`
(env-configurable `MAS_SYMBOL`, default `"BTCUSDT"`) for every article before
invoking the graph. The `rag_retriever` node requires `symbol` in state.

---

## Infrastructure Known Gaps (document, don't fix)

- **`news_firehose_idx`** is not provisioned. `rag_retriever` will always return
  `precedent_status="unavailable"`. Log `eval.precedent_ok_rate ≈ 0` as a metric
  and note this explicitly in the §5 markdown cell. Not an error.
- **VS index** (`causal_chains_idx`) lives in the archived catalog. Few-shot
  retrieval is degraded. This is an existing infra gap, out of scope here.

---

## Notebook Structure (17 cells)

### Cell 1 — pip / restart
```python
# MAGIC %pip install -q -e .
# MAGIC %restart_python
```

### Cell 2 — §1 Setup (markdown)
```
# §1 Setup
```

### Cell 3 — Imports + config + MLflow session open
```python
import os
import mlflow
import polars as pl

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.graph import build_mas_graph
from trading_databricks.agents.graph_env import wire_graph_environment
from notebooks.qa.utils import log_plotly

SAMPLE_SIZE = int(os.getenv("MAS_SAMPLE_SIZE", "50"))
SEED        = int(os.getenv("MAS_SEED", "42"))
SYMBOL      = os.getenv("MAS_SYMBOL", "BTCUSDT")

config = MASConfig()

mlflow.set_experiment("/Shared/mas-deep-dive")
mlflow.langchain.autolog()
run = mlflow.start_run()
```

### Cell 4 — §2 Preflight (markdown)
```
# §2 Preflight
LLM endpoint: critical — abort on failure.
Embedding endpoint: warn only.
Whitelist/taxonomy tables: warn only.
```

### Cell 5 — Preflight checks
Calls the LLM endpoint with a single probe message. Raises `RuntimeError` if the
LLM endpoint is unreachable (hard fail — no point continuing without it).
Logs `preflight.llm_ok`, `preflight.embedding_ok`, `preflight.whitelist_ok`,
`preflight.taxonomy_ok` as MLflow metrics (1 = pass, 0 = fail).

### Cell 6 — §3 Data Load (markdown)

### Cell 7 — Data load
```python
raw = spark.table("smart-trading-dev.01_bronze.raw_news")
# Schema probe: display column names + dtypes
# Crypto-keyword filter (headline or body contains SYMBOL or "bitcoin"/"ethereum"/...)
# .limit(SAMPLE_SIZE * 5).sample(fraction=..., seed=SEED).limit(SAMPLE_SIZE)
# Collect to Python list of dicts; inject symbol=SYMBOL into each row
```

### Cell 8 — §4 Graph Execution (markdown)

### Cell 9 — Graph execution (SELF-CONTAINED context)
```python
results = []
with wire_graph_environment(config, offline=False):
    graph = build_mas_graph(config=config)
    for article in articles:
        out = graph.invoke({**article, "symbol": SYMBOL})
        results.append({**article, **out})

df = pl.DataFrame(results)
display(df.select(["headline", "symbol", "extraction_path", "validation_status",
                   "extraction_status", "evidence_status", "precedent_status",
                   "reflection_count"]))
```
The `wire_graph_environment` context is fully contained in this single cell —
no fragile multi-cell span.

### Cell 10 — §5 Sub-pipeline Diagnostics (markdown)
Notes that `precedent_ok_rate ≈ 0` is expected because `news_firehose_idx` is
not provisioned.

### Cell 11 — Sub-pipeline diagnostics
Compute and log:
- `evidence_status` distribution (`eval.evidence_ok_rate`)
- `precedent_status` distribution (`eval.precedent_ok_rate`)
- `extraction_status` distribution (`eval.extraction_ok_rate`)
- judge panel per-judge pass rates — unnest `judge_failures` dict
  (`eval.judge_all_pass_rate`)
- `reflection_count` distribution (`eval.reflection_triggered_rate`)

All 5 metrics logged to MLflow via `mlflow.log_metrics(...)`.

### Cell 12 — §6 Extraction Quality Scorers (markdown)

### Cell 13 — Scorer aggregation
Apply all 8 scorer functions (4 existing + 4 new) to every row. Log 8 aggregate
metrics to MLflow:
- `scorer.schema_conformance_mean`
- `scorer.taxonomy_compliance_rate`
- `scorer.routing_legality_rate`
- `scorer.evidence_grounding_mean`
- `scorer.evidence_quality_rate`
- `scorer.judge_all_pass_rate`
- `scorer.reflection_triggered_rate`
- `scorer.extraction_ok_rate`

Display a Polars summary table of per-article scorer results.

### Cell 14 — §7 Visualisations (markdown)

### Cell 15 — Plotly charts (4 charts, all logged via `log_plotly`)
1. **extraction_path distribution** — bar chart: fast / standard / careful counts
2. **validation_status distribution** — bar chart: valid / invalid / error counts
3. **extraction_status distribution** — bar chart: ok / ok_after_reflection /
   judge_failed_accepted / unjudged / error counts
4. **judge pass rates** — bar chart: per-judge pass rate from unnested
   `judge_failures`

### Cell 16 — §8 Sanity Assertions (markdown)

### Cell 17 — Hard invariants + MLflow run close
```python
# Hard invariants (raise AssertionError on violation)
assert df.height > 0, "No results produced"
assert df["extraction_status"].null_count() == 0, "extraction_status must be set on all rows"
assert df["evidence_status"].null_count() == 0, "evidence_status must be set on all rows"

mlflow.end_run()
```

---

## `_scorer_helpers.py` Extensions

Four new predicates added to `notebooks/qa/mas/_scorer_helpers.py`:

```python
def check_evidence_quality(evidence_status: str | None) -> bool:
    """Return True iff evidence_status is 'ok'."""
    return evidence_status == "ok"


def check_judge_all_pass(judge_failures: dict | None) -> bool:
    """Return True iff judge_failures is None (all judges passed)."""
    return judge_failures is None


def check_reflection_triggered(reflection_count: int | None) -> bool:
    """Return True iff at least one reflection loop ran."""
    return (reflection_count or 0) > 0


def check_extraction_ok(extraction_status: str | None) -> bool:
    """Return True iff extraction completed (directly or after reflection)."""
    return extraction_status in {"ok", "ok_after_reflection"}
```

Unit tests for these go in `tests/qa/test_scorer_helpers.py`.

---

## New MASState Keys Covered (TRD-68)

| Key | Type | Where used in notebook |
|-----|------|------------------------|
| `evidence` | `list[str]` | §5 evidence quality |
| `evidence_status` | `str` | §5 metric + §6 scorer |
| `precedents` | `list` | §5 precedent ok rate |
| `precedent_status` | `str` | §5 metric (expected ≈ 0) |
| `extraction_status` | `str` | §5 metric + §7 chart |
| `judge_results` | `list` | §5 per-judge rates |
| `judge_failures` | `dict \| None` | §5 metric + §6 scorer |
| `reflection_count` | `int` | §5 metric + §6 scorer |

Valid `extraction_status` values: `ok`, `ok_after_reflection`,
`judge_failed_accepted`, `unjudged`, `error`.

---

## Key Design Decisions

1. **No offline mode** — live endpoints only; the existing `99_e2e_graph.py`
   covers mocked E2E.
2. **Single-cell context** — `wire_graph_environment` fully scoped within §4
   cell; no multi-cell `__enter__`/`__exit__` spans.
3. **`mlflow.start_run()` / `mlflow.end_run()` explicit** — instead of
   `mlflow_component_run` context manager, because this notebook is a long-running
   interactive session, not a single-shot function call.
4. **Polars throughout** — all post-collection DataFrame work uses Polars
   (`pl.DataFrame`). No pandas.
5. **`display()` for Databricks rendering** — all DataFrames surfaced via Spark's
   `display()` built-in so they render as interactive tables in the notebook UI.
6. **`log_plotly` for chart persistence** — all 4 Plotly charts logged as MLflow
   artifacts so they survive run completion.
7. **Preflight is a hard gate** — LLM endpoint failure raises immediately; the
   notebook does not proceed to data load or graph execution.

---

## Testing

`_scorer_helpers.py` new predicates are unit-tested in
`tests/qa/test_scorer_helpers.py` — no Spark, no MLflow, no network.
All other notebook logic is integration-tested by running the notebook in
Databricks. Local `uv run pytest` must pass before marking done.
