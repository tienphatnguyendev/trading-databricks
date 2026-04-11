# MAS E2E Deep-Dive Research Notebook — Design

**File**: `notebooks/qa/mas/e2e_mas_deep_dive_research.py`
**Format**: Databricks notebook source (`# Databricks notebook source` + `# COMMAND ----------` + `# MAGIC %md`) — opens cell-by-cell in the workspace **and** runs top-to-bottom via `uv run python`
**Linear ticket**: TBD (create before branching per CLAUDE.md)
**Related**: `2026-04-09-mas-interactive-qa-notebooks-design.md`

## Objective

Build a deep-dive research notebook that runs the compiled MAS LangGraph (`build_mas_graph`) end-to-end on real news from `smart-trading-dev.others.mas_training_dataset` against **live** `databricks-claude-sonnet` and `databricks-gte-large-en` endpoints, produces per-node real-span observability, computes structural and LLM-as-judge quality scorers via `mlflow.genai.evaluate()`, and exposes an interactive failure-autopsy sandbox.

This notebook is a **research instrument**, not a CI smoke test. `notebooks/qa/mas/99_e2e_graph.py` stays as-is for the lightweight smoke test.

## Why a new file (and not a refactor of `99_e2e_graph.py`)

The existing file has three integrity problems that disqualify it as a research instrument:

1. **Fabricated node latencies** — `99_e2e_graph.py:57-60` literally synthesizes fake numbers and plots them as measurements.
2. **`_mechanism_validator_node` patched to `MagicMock`** in `notebooks/qa/utils.py:544` — the "E2E" graph never runs the real validator. The existing assertion `validation_status in {..., None}` passes vacuously because the mocked state never sets the key.
3. **No `mlflow.genai.evaluate()`, no scorers, no dataset lineage, no real span analysis.** Traces are captured by autolog but only `len(traces) >= 20` is asserted.

Keeping `99_e2e_graph.py` intact preserves the fast offline smoke test while isolating live endpoints, higher token cost, and larger sample sizes inside the new file.

## Approach (confirmed)

**Evaluate-first with interactive sandbox (Approach 3).** A single `mlflow.genai.evaluate(data, predict_fn, scorers)` call is the primary execution path. All downstream metrics, visualizations, and the failure-autopsy sandbox derive from the traces produced by that one call — no duplicated invocation loops, no fabricated data.

## Live-mode asset inventory

Two catalogs are involved and **they are different**:
- `smart-trading-dev` — holds the input dataset
- `smart-trading-dev` — holds the MAS reference tables and VS index (per `MASConfig` defaults)

| # | Asset | Location | Criticality | Consumer | If missing |
|---|---|---|---|---|---|
| 1 | `databricks-claude-sonnet` | Model Serving endpoint | **hard** | `causality_extractor` | graph crashes |
| 2 | `databricks-gte-large-en` | Model Serving endpoint | soft | `mechanism_validator` Module B | `evidence_relevance_score = None`, validator still runs |
| 3 | `treatment_taxonomy` | `smart-trading-dev.gold` Delta | **hard** | `causality_extractor.load_taxonomy` | crash on `.collect()` |
| 4 | `mechanism_whitelist` | `smart-trading-dev.gold` Delta | **hard** | `causality_extractor` + `_mechanism_validator_node` | both crash |
| 5 | `news_causal_features` | `smart-trading-dev.gold` Delta | soft | `few_shot_retriever.get_accepted_chain_count` | graceful cold-start |
| 6 | `causal_chains_idx` | VS index on endpoint `mas-few-shot` | soft | `few_shot_retriever.query_vector_search` | empty results, merges to static |
| 7 | `mas_training_dataset` | `smart-trading-dev.others` Delta | **hard** | §4 dataset ingestion | notebook cannot run |

Two **symbols** also require injection at runtime because the source code ships `NotImplementedError` placeholders (`few_shot_retriever.py:136`, `few_shot_retriever.py:263`):
- `trading_databricks.agents.nodes.few_shot_retriever.query_vs_index`
- `trading_databricks.agents.nodes.few_shot_retriever.get_spark_session`
- `trading_databricks.agents.nodes.causality_extractor.get_spark_session` (imported from the same module)

The notebook must monkey-patch these with real implementations before invoking the graph.

### Cold-start reality check

If `news_causal_features.accepted_count < 50` or the VS index is empty, `few_shot_retriever` returns 5 static examples whose `similarity_score` field is absent → `uncertainty_router` reads `0.0` → path is always `careful`. In dev today this is likely true. The notebook **must** detect and log `cold_start_mode` prominently so the resulting routing distribution is not mis-read as a model bug. The `routing_legality` scorer adapts its legal-path rule based on this flag.

## Input mapping: `mas_training_dataset` → `MASState`

Source columns: `story_id`, `published_at`, `request`, `expected_route`, `context`.

| MASState field | Source |
|---|---|
| `headline` | Regex-strip `r"^Can you summarize the market impact of the following news:\s*"` from `request`; fallback to full `request` on no match |
| `body` | `context` (use `""` if null — per `state.py:36` the field is always `str`, never `None`) |
| `published_at` | `published_at` (ISO string) |

Evaluation metadata carried alongside state (not fed to graph): `story_id`, `expected_route`.

**Cohort filter**: graph is run only on rows where `expected_route == 'news_agent'`. A separate negative-control slice of K non-`news_agent` rows is evaluated as a child run for behavioral comparison.

**Sampling**: stratified by published-date bucket, seeded, configurable:

| Env var | Default | Purpose |
|---|---|---|
| `MAS_RESEARCH_OFFLINE` | unset (live) | Bypass preflight hard-fails; inject deterministic mocks for LLM, VS, embeddings |
| `MAS_RESEARCH_SAMPLE_SIZE` | `50` | Main cohort size |
| `MAS_RESEARCH_NEG_CONTROL_SIZE` | `5` | Negative-control slice size |
| `MAS_RESEARCH_SEED` | `42` | Deterministic sampling |

Dataset lineage logged via `mlflow.data.from_pandas(df.to_pandas(), source=DeltaDatasetSource(delta_table_name="smart-trading-dev.others.mas_training_dataset"), name="mas_deep_dive_research_sample")` then `mlflow.log_input(dataset, context="research")`.

## Scope boundaries

**In scope:**
- New file `notebooks/qa/mas/e2e_mas_deep_dive_research.py`
- New helpers added to `notebooks/qa/utils.py` (shared with sibling notebooks): `_extract_headline`, `_get_dbutils`, `_display`, `_endpoint_ready`, `_table_has_rows`, `_vs_index_exists`, `build_instrumented_mas_graph`, `wire_graph_environment`
- Notebook-local scorer implementations (the scorers are research artifacts, not production code, and stay inside the notebook file)

**Out of scope (explicit non-goals):**
- Modifying any file in `src/trading_databricks/agents/**` — production code is untouched
- Modifying `notebooks/qa/mas/99_e2e_graph.py` — smoke test stays as-is
- Seeding `treatment_taxonomy` or `mechanism_whitelist` — prerequisites, not deliverables
- Populating `news_causal_features` or `causal_chains_idx` — the notebook handles cold-start gracefully
- Workflow-7 (judge alignment via MemAlign) and Workflow-8 (GEPA prompt optimization) — follow-ups consume this notebook's outputs

## Section-by-section design

### §0 — Header (MD)

Title; objective; how-to-run (both Databricks workspace and `uv run python`); estimated token cost; env var catalog; offline toggle; explicit warning that live mode hits real endpoints and will spend tokens; pointer to the interactive-QA spec.

### §1 — Setup

```python
import mlflow
import os
import polars as pl
from trading_databricks.agents.config import MASConfig
from notebooks.qa.utils import (
    mlflow_component_run, log_diag, log_plotly,
    build_instrumented_mas_graph, wire_graph_environment,
    _display, _get_dbutils, _extract_headline,
)

OFFLINE = os.environ.get("MAS_RESEARCH_OFFLINE") == "1"
SAMPLE_SIZE = int(os.environ.get("MAS_RESEARCH_SAMPLE_SIZE", "50"))
NEG_CONTROL_SIZE = int(os.environ.get("MAS_RESEARCH_NEG_CONTROL_SIZE", "5"))
SEED = int(os.environ.get("MAS_RESEARCH_SEED", "42"))

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/trading-databricks-qa/mas/e2e_deep_dive_research")
mlflow.langchain.autolog()
config = MASConfig()
```

`mlflow_component_run` context wraps the entire notebook body (same pattern as sibling notebooks). Tags added on the run: `layer=mas`, `component=e2e_deep_dive_research`, `mode=live|offline`, `sample_size`, `seed`, `cold_start` (set later in §3), plus git provenance from `_git_tags()`.

### §2 — System topology diagram

```python
graph = build_instrumented_mas_graph(config)
mermaid = graph.get_graph().draw_mermaid()
mlflow.log_text(mermaid, "topology/graph.mmd")
```

Followed by a `# MAGIC %md` cell embedding the mermaid source in a fenced ```` ```mermaid ```` block so it renders in the Databricks workspace. When run as plain Python the MD lines are no-op comments.

### §3 — Pre-flight checks

Single function `preflight(config, *, offline: bool) -> pl.DataFrame`. Each check:

- Records `(name, ok, detail, critical)`
- Logs `mlflow.log_metric(f"preflight.{name}.ok", 0|1)`
- Hard-fails the run if any critical check fails **and** `offline=False`

Checks:

| Check | Implementation | Critical? |
|---|---|---|
| `endpoint.claude_sonnet` | `WorkspaceClient().serving_endpoints.get(config.llm_endpoint).state.ready == 'READY'` | **yes** |
| `endpoint.gte_large_en` | same for `config.validator.embedding_model` | no (validator degrades gracefully) |
| `table.treatment_taxonomy` | `spark.table(...).limit(1).count() > 0` | **yes** |
| `table.mechanism_whitelist` | same | **yes** |
| `table.news_causal_features` | `spark.catalog.tableExists(...)` | no (cold-start) |
| `vs.causal_chains_idx` | `VectorSearchClient().get_index(endpoint_name=config.vs_endpoint_name, index_name=config.vs_index_name).describe()` | no (empty results) |
| `table.mas_training_dataset` | `spark.table("smart-trading-dev.others.mas_training_dataset").limit(1).count() > 0` | **yes** |

Cold-start detection:

```python
accepted_count = _query_accepted_count(config) if news_causal_features_exists else 0
cold_start_mode = accepted_count < config.router.min_index_size
mlflow.set_tag("cold_start", str(cold_start_mode).lower())
mlflow.log_metric("cold_start_mode", int(cold_start_mode))
mlflow.log_metric("accepted_chain_count", accepted_count)
```

Results logged via `log_diag("preflight", df)`. If cold-start, print a prominent warning banner via `_display`.

### §4 — Dataset ingestion, stratified sampling, lineage

```python
spark = get_spark()
raw = pl.from_arrow(
    spark.table("smart-trading-dev.others.mas_training_dataset").toArrow()
)
news = raw.filter(pl.col("expected_route") == "news_agent")
news = news.with_columns(
    pl.col("request").map_elements(_extract_headline, return_dtype=pl.Utf8).alias("headline"),
    pl.col("context").fill_null("").alias("body"),
    pl.col("published_at").dt.date().alias("_bucket"),
)
# seeded stratified sample
sample = _stratified_sample(news, n=SAMPLE_SIZE, by="_bucket", seed=SEED)

eval_dataset = [
    {"inputs": {
        "headline": r["headline"], "body": r["body"],
        "published_at": str(r["published_at"]), "story_id": r["story_id"],
    }}
    for r in sample.to_dicts()
]

# negative control — non-news_agent
negative = raw.filter(pl.col("expected_route") != "news_agent").sample(
    n=min(NEG_CONTROL_SIZE, raw.filter(pl.col("expected_route") != "news_agent").height),
    seed=SEED,
)
negative_control_dataset = [ ... similar shape ... ]

# lineage
from mlflow.data.delta_dataset_source import DeltaDatasetSource
ds = mlflow.data.from_pandas(
    sample.to_pandas(),
    source=DeltaDatasetSource(delta_table_name="smart-trading-dev.others.mas_training_dataset"),
    name="mas_deep_dive_research_sample",
)
mlflow.log_input(ds, context="research")
_display(sample.head(10))
```

`_extract_headline` lives in `utils.py` and handles the regex + fallback.

### §5 — Graph wiring + instrumentation

Two new helpers in `notebooks/qa/utils.py`:

**`build_instrumented_mas_graph(config) -> CompiledStateGraph`** — mirrors `build_mas_graph` but wraps each node in an explicit `mlflow.trace(name=<node>, span_type=SpanType.CHAIN)` decorator so per-node spans are guaranteed regardless of LangGraph autolog coverage or MLflow version drift. Uses the same node functions from `src/trading_databricks/agents/nodes/` — no production code changes.

```python
from mlflow.entities import SpanType

def build_instrumented_mas_graph(config):
    from langgraph.graph import StateGraph
    from trading_databricks.agents.nodes import (
        causality_extractor, context_retriever,
        few_shot_retriever, uncertainty_router,
    )
    from trading_databricks.agents.graph import route_article, _mechanism_validator_node
    from trading_databricks.agents.state import MASState

    def _traced(name, fn):
        @mlflow.trace(name=name, span_type=SpanType.CHAIN)
        def wrapper(state):
            return fn(state)
        return wrapper

    graph = StateGraph(MASState)
    graph.add_node("context_retriever",   _traced("context_retriever",   lambda s: context_retriever(s, config=config)))
    graph.add_node("few_shot_retriever",  _traced("few_shot_retriever",  lambda s: few_shot_retriever(s, config=config)))
    graph.add_node("uncertainty_router",  _traced("uncertainty_router",  lambda s: uncertainty_router(s, config=config)))
    graph.add_node("causality_extractor", _traced("causality_extractor", lambda s: causality_extractor(s, config=config)))
    graph.add_node("mechanism_validator", _traced("mechanism_validator", lambda s: _mechanism_validator_node(s, config)))
    graph.set_entry_point("context_retriever")
    graph.add_edge("context_retriever",   "few_shot_retriever")
    graph.add_edge("few_shot_retriever",  "uncertainty_router")
    graph.add_conditional_edges("uncertainty_router", route_article, {"causality_extractor": "causality_extractor"})
    graph.add_edge("causality_extractor", "mechanism_validator")
    graph.set_finish_point("mechanism_validator")
    return graph.compile()
```

**`wire_graph_environment(config, *, offline: bool)`** — monkey-patches the runtime hooks:

- `trading_databricks.agents.nodes.few_shot_retriever.get_spark_session → get_spark`
- `trading_databricks.agents.nodes.causality_extractor.get_spark_session → get_spark`
- `trading_databricks.agents.nodes.few_shot_retriever.query_vs_index → _real_vs_query` (live) or `_mock_vs_query` (offline). The live wrapper uses `VectorSearchClient(...).get_index(...).similarity_search(query_text=..., columns=..., num_results=..., filters=json.loads(filters_json))` and normalizes result rows into the dict shape expected by the node.
- Offline branch **additionally** patches `mlflow.deployments.get_deploy_client("databricks").predict` with a context-sensitive deterministic mock: returns a stable taxonomy-compliant JSON for the LLM endpoint, returns a fixed unit vector for the embedding endpoint.
- **`_mechanism_validator_node` is NEVER patched**, in live or offline mode. The real deterministic validation logic (whitelist lookup, substring matching, Module B) runs on every row. This is the critical regression fix vs `99_e2e_graph.py`.

### §6 — `predict_fn` + custom scorers

`predict_fn` **never raises** — row-level failures are returned as structured error dicts so `evaluate()` completes the full sample.

```python
def predict_fn(*, headline, body, published_at, story_id):
    try:
        out = graph.invoke({"headline": headline, "body": body, "published_at": published_at})
    except Exception as e:
        return {
            "error": repr(e),
            "extraction": None, "extraction_path": None,
            "validation_status": None, "evidence_relevance_score": None,
            "retrieval_story_ids": [],
        }
    return {
        "error": None,
        "extraction_path": out.get("extraction_path"),
        "extraction": out.get("extraction"),
        "validation_status": out.get("validation_status"),
        "evidence_relevance_score": out.get("evidence_relevance_score"),
        "retrieval_story_ids": out.get("retrieval_story_ids", []),
    }
```

Custom `@scorer` functions (all reference-free, defined in the notebook):

1. **`schema_conformance`** → float ∈ [0,1]. Checks required keys (`treatment_class`, `mechanism`, `expected_direction`, `confidence`, `trigger_event`, `evidence_spans`), valid types, `confidence ∈ [0,1]`, `expected_direction ∈ {bullish,bearish,volatile}`.
2. **`taxonomy_whitelist_compliance`** → bool. Closes over the real `whitelist` dict loaded once via `load_whitelist`. Checks `(L1, mechanism) ∈ whitelist` AND `expected_direction ∈ whitelist[(L1, mechanism)]`.
3. **`routing_legality`** → bool. Factory `make_routing_legality_scorer(cold_start_mode, config)`. If cold-start, the only legal path is `careful`. Otherwise applies the spec rule: `fast` iff top_similarity ≥ 0.85 AND ≥ 2 examples; `careful` iff top_similarity < 0.50 OR < 2 examples; `standard` otherwise.
4. **`evidence_grounding_substring`** → float ∈ [0,1]. Fraction of `evidence_spans` that appear as casefold substrings of `body`. Mirrors `_validate_evidence_spans` logic but returns a granular ratio so `evaluate()` can surface partial-grounding signal.
5. **`extraction_path_label`** → categorical. Records `extraction_path` so `evaluate()` aggregates the distribution.

LLM-as-judge:

6. **`groundedness_judge`** via `mlflow.genai.judges.make_judge`:
   ```python
   groundedness_judge = make_judge(
       name="groundedness",
       instructions="""Given the article body, the extracted mechanism, the trigger_event, and the evidence_spans, rate whether the causal chain is supported by the article body.
       Return 'faithful' if every claim is clearly backed by the evidence; 'partial' if some claims are supported but others drift; 'hallucinated' if the extraction introduces facts not present in the body.""",
       feedback_value_type="categorical",
       model=f"endpoints:/{config.llm_endpoint}",
       embedding_model=f"endpoints:/{config.validator.embedding_model}",  # explicit per GOTCHA
   )
   ```

All six scorers are passed to `evaluate()` in §7.

### §7 — `mlflow.genai.evaluate()` call

```python
eval_result = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=predict_fn,
    scorers=[
        schema_conformance,
        taxonomy_whitelist_compliance,
        make_routing_legality_scorer(cold_start_mode, config),
        evidence_grounding_substring,
        extraction_path_label,
        groundedness_judge,
    ],
)
```

A second `evaluate()` call runs `negative_control_dataset` under a nested run tagged `slice=negative_control` for behavioral comparison on off-domain text.

### §8 — Trace-derived analysis (replaces fabricated latencies)

```python
run_id = mlflow.active_run().info.run_id
traces_df = mlflow.search_traces(run_id=run_id, extract_fields=["spans"])
```

Explode spans → one row per `{trace_id, span_name, span_type, latency_ms, status, tokens_in, tokens_out}`. Derive `latency_ms = (end_time_ns - start_time_ns) / 1_000_000`. Pull token counts from LLM span attributes (`mlflow.chat.tokenUsage` or equivalent).

Join on `trace_id` to attach `story_id` and `extraction_path` (from the root trace's `outputs`). Aggregate into `node_latency_stats` (`node × path × status → p50, p95, p99, error_count, n_invocations`).

Log:
- `log_diag("node_latency_stats", stats_df)`
- `log_diag("span_metrics_raw", flat_span_df)`
- `log_plotly("node_latency_boxplot_real", build_box(flat_span_df, x="node", y="latency_ms"))` — real data, not fabricated
- `mlflow.log_metric("tokens.total_in", int(...))`, `tokens.total_out`, `tokens.est_cost_usd`
- `mlflow.log_metric("errors.total", int(...))`, per-node error count

### §9 — Execution-overlay topology diagram

Plotly directed graph built from §8 stats:
- Nodes placed horizontally: `context_retriever → few_shot_retriever → uncertainty_router → causality_extractor → mechanism_validator`
- Marker `size ∝ throughput`, `color = median_latency_ms` (viridis), annotation shows `n=…, p50=…ms, errors=…`
- Edges as Scatter lines, width ∝ transition count (from span parent-child data)

Logged via `log_plotly("topology_overlay", fig)`. This is the "system as a diagram with research signal overlaid" deliverable.

### §10 — Scorer result aggregation & visualization

Pull scorer results from `eval_result` into Polars. Produce:
- Per-scorer pass rate overall and grouped by `extraction_path` and `cold_start_mode`
- `validation_status` breakdown (pie)
- `groundedness_judge` categorical stacked bar, one bar per path
- `evidence_grounding_substring` histogram
- `extraction_path` distribution pie

All logged via `log_plotly(...)` and `log_diag("eval_metrics_table", df)`.

### §11 — Failure autopsy sandbox

Interactive widgets (gracefully degrade when `dbutils` is unavailable):

```python
dbu = _get_dbutils()
if dbu:
    dbu.widgets.text("story_id", "", "story_id to replay")
    dbu.widgets.text("override_headline", "", "override headline (optional)")
    dbu.widgets.text("override_body", "", "override body (optional)")
```

Helper `render_autopsy(story_id: str) -> None`:
1. Locate the trace matching `(run_id, story_id)` from the §8 span DataFrame
2. Pretty-print the `causality_extractor` span's full system prompt (the messages list from inputs — thanks to autolog)
3. Show every LLM choice (raw text, parsed JSON, majority vote result)
4. Show the `mechanism_validator` span's `validation_status`, `rejected_mechanism_raw`, per-span substring match results, Module B `evidence_relevance_score`
5. Show the `groundedness_judge` verdict + rationale
6. Link to the MLflow trace UI

Helper `replay_single(story_id, *, state_overrides=None) -> None`:
1. Load the original row from §4 sample
2. Build state; apply overrides
3. `with mlflow.start_run(nested=True): graph.invoke(state)` — tagged `replay=true`
4. Display resulting state + link to the nested run

Also prints top-K worst cases ranked by `groundedness_judge` so the user doesn't have to hunt.

### §12 — Human feedback hooks (Workflow-7 seed)

For each trace in the current run, emit a stubbed feedback record so downstream judge alignment can attach:

```python
from mlflow.entities import AssessmentSource, AssessmentSourceType
for tid in traces_df["trace_id"].unique():
    mlflow.log_feedback(
        trace_id=tid,
        name="sme_verdict",
        value=None,  # reserved slot
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,
            source_id="placeholder_sme",
        ),
    )
```

Comment block explains how to launch a UC-backed labeling session and align `groundedness_judge` via MemAlign in a follow-up.

### §13 — Sanity assertions (loose, research-appropriate)

- `eval_result.metrics is not None`
- `eval_result.predictions_count == SAMPLE_SIZE` (no rows silently dropped)
- `len(traces_df["trace_id"].unique()) >= SAMPLE_SIZE`
- At least one `preflight.*.ok` metric was logged on the run
- Offline mode additionally asserts every custom scorer executed (the judge may no-op)

**Deliberately NOT asserted**: `extraction_path` distribution, validation pass rate, `groundedness_judge` score distribution. Research notebooks observe — they do not gate.

### §14 — Summary & next steps (MD + code)

Rendered summary table of top-level metrics from this run; direct link to the MLflow experiment UI; next-steps list:
- Launch a UC labeling session against the `sme_verdict` slots (Workflow-7)
- Run MemAlign to align `groundedness_judge` to SME labels
- Use this notebook's eval dataset as input to GEPA prompt optimization (Workflow-8) against the system prompts in `trading_databricks.agents.prompts`

## Notebook file format conventions

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # MAS E2E Deep-Dive Research Notebook
# MAGIC
# MAGIC Objective ...

# COMMAND ----------

import mlflow
# ... code ...

# COMMAND ----------

# MAGIC %md
# MAGIC ## §1 Setup
```

Rules:
- First line **must** be `# Databricks notebook source`
- `# COMMAND ----------` (with blank lines around it) separates cells
- Markdown cells: every line begins with `# MAGIC %md` or `# MAGIC <content>`
- Code cells: plain Python after `# COMMAND ----------`
- No `%pip install` magic cells — dependencies live in `pyproject.toml` + `uv sync`
- No `%sql` magic cells — all SQL via `spark.sql(...)`

Dual-run guards (so the same file works via both Databricks workspace and `uv run python`):
- `_display(obj)` — tries `display(obj)` (Databricks builtin), falls back to `print` on `NameError`
- `_get_dbutils()` — returns `DBUtils(spark)` when available, `None` otherwise
- All widget calls gated on `_get_dbutils()` returning non-None
- `os.environ` is the authoritative config source in both environments

## Token cost estimate (live mode, N=50)

| Call | Tokens per article | Total (50 articles) |
|---|---|---|
| `causality_extractor` fast path | ~1.5k in + ~0.5k out | n/a under cold-start |
| `causality_extractor` careful path (3 choices) | ~3 × (2k in + 0.8k out) | ~420k |
| Module B embeddings (2 calls) | ~100 | ~5k |
| `groundedness_judge` | ~2k in + ~0.2k out | ~110k |
| **Approx total** | | **~535k tokens** |

Logged as `mlflow.log_metric("tokens.estimate_pre_run", 535_000)` at §1 so the reader is calibrated before the run starts.

## Assumptions

- `pyproject.toml` includes `mlflow>=3.0` with the GenAI eval API and `databricks-vectorsearch>=0.x`; if not, a one-line follow-up adds them
- `databricks.connect` is configured (`~/.databrickscfg`)
- `smart-trading-dev.03_gold.treatment_taxonomy` and `mechanism_whitelist` are already seeded via `seed_taxonomy()` before the first live run (prerequisite, not a deliverable)
- The MLflow experiment path `/Shared/trading-databricks-qa/mas/e2e_deep_dive_research` is creatable by the running user
- The cohort filter `expected_route == 'news_agent'` yields ≥ `MAS_RESEARCH_SAMPLE_SIZE` rows (to be verified in §3 preflight)

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| LLM endpoint rate limits / 429s | Retry-on-429 wrapper around the deploy client; default sample size of 50 is conservative; §0 documents cost estimate |
| Cold-start masks routing behavior | §3 detects and tags `cold_start_mode`; `routing_legality` scorer adapts; §0 warns reader |
| LangGraph autolog misses per-node spans | `build_instrumented_mas_graph` explicitly wraps every node in `mlflow.trace` |
| `predict_fn` exception aborts the whole eval | `predict_fn` catches and returns structured error dicts; scorers treat `error != None` as fail |
| Monkey-patches leak into other notebooks | Patches applied inside the top-level run context; offline mocks idempotent |
| Fabricated-latency regression | Spec forbids; §8 derives all latency from real spans — no `np.random` or time-padding code anywhere |
| Validator-mock regression | Spec forbids patching `_mechanism_validator_node` in any mode; offline mocks only the LLM + embedding + VS clients so the real validator logic still runs |
| Tightly-coupled assertions flake under cold-start | §13 asserts only on sample-size invariants, never on value distributions |
| Dataset field drift | `mas_training_dataset` schema is confirmed in the Q2 answer; `_extract_headline` has a fallback; §3 preflight verifies the table |

## Success criteria

- Notebook executes end-to-end in live mode against real endpoints for N=50 rows without raising
- Every scorer returns a value for every row (or a logged `error` field)
- MLflow run contains: preflight diagnostic table; dataset lineage record; `topology/graph.mmd` artifact; execution-overlay HTML; per-node real-latency parquet; scorer results; per-trace feedback stubs; ≥ 1 replay child run
- No fabricated data anywhere
- `_mechanism_validator_node` runs for real (never a `MagicMock`)
- Offline mode exercises the full scorer pipeline without touching live endpoints

## Follow-ups (not in this spec)

- Linear ticket for notebook creation (TRD-???)
- Workflow-7 judge-alignment notebook that consumes the `sme_verdict` feedback slots
- Workflow-8 GEPA prompt-optimization notebook that consumes this notebook's eval dataset
- Retrofit `build_instrumented_mas_graph` + real-span instrumentation onto `01_gatekeeper.py`…`06_mechanism_validator.py`
- Optional pytest marker that runs offline mode as a nightly CI job

## Spec self-review

- **Placeholder scan**: one TBD (Linear ticket ID) — acceptable per CLAUDE.md, created before branching. No other TBDs.
- **Internal consistency**: sections §1–§14 form a complete dependency chain. §2 depends on the graph built in §5 (resolved by placing §5's `build_instrumented_mas_graph` helper in `utils.py` so §2 can call it before the full wire-up); wire-up itself runs in §5 proper. §6's `predict_fn` closes over `graph`; §7 runs evaluate; §8–§13 read from §7's artifacts. No contradictions.
- **Scope check**: single notebook, single design, one implementation plan. Not a multi-deliverable spec.
- **Ambiguity check**: the load-bearing ambiguity was whether to modify `src/trading_databricks/agents/nodes/*.py` to add tracing decorators — resolved explicitly in §5 and Scope boundaries (we do **not**; tracing is injected via a local `build_instrumented_mas_graph` wrapper).
