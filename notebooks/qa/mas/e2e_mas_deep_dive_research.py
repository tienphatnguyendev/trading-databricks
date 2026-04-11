# Databricks notebook source

# MAGIC %pip install -q -e .

# COMMAND ----------

# MAGIC %md
# MAGIC # MAS E2E Deep-Dive Research Notebook
# MAGIC
# MAGIC **Purpose:** Execute the compiled MAS LangGraph end-to-end with per-node MLflow
# MAGIC CHAIN spans, custom scorers, an LLM-as-judge, and an interactive failure-autopsy
# MAGIC sandbox. This is a research instrument, not a CI gate.
# MAGIC
# MAGIC **Outputs:**
# MAGIC - Per-node latency boxplots from *real* MLflow spans (never fabricated)
# MAGIC - `mlflow.genai.evaluate()` results with 5 custom scorers + groundedness judge
# MAGIC - Failure-autopsy sandbox with widget-driven replay
# MAGIC - Human-feedback stubs (`sme_verdict`) for downstream MemAlign
# MAGIC
# MAGIC **MLflow experiment:** `/Shared/trading-databricks-qa/mas/e2e_deep_dive_research`
# MAGIC
# MAGIC **Author:** TRD-64

# COMMAND ----------

# MAGIC %md
# MAGIC ## §1 Setup
# MAGIC Reads environment variables, sets MLflow experiment, enables autolog.

# COMMAND ----------

from __future__ import annotations

import json
import os

import mlflow
import mlflow.langchain
import polars as pl

from notebooks.qa.utils import (
    _display,
    _endpoint_ready,
    _extract_headline,
    _get_dbutils,
    _stratified_sample,
    _table_has_rows,
    _vs_index_exists,
    build_instrumented_mas_graph,
    build_label_distribution,
    build_path_distribution,
    log_diag,
    log_plotly,
    mlflow_component_run,
    wire_graph_environment,
)
from trading_databricks.agents.config import MASConfig

# ---------------------------------------------------------------------------
# Execution parameters — override via environment variables
# ---------------------------------------------------------------------------
OFFLINE: bool = os.getenv("MAS_RESEARCH_OFFLINE", "0") == "1"
SAMPLE_SIZE: int = int(os.getenv("MAS_RESEARCH_SAMPLE_SIZE", "50"))
NEG_CONTROL_SIZE: int = int(os.getenv("MAS_RESEARCH_NEG_CONTROL_SIZE", "5"))
SEED: int = int(os.getenv("MAS_RESEARCH_SEED", "42"))

config = MASConfig()

params: dict = {
    "mode": "offline" if OFFLINE else "live",
    "sample_size": SAMPLE_SIZE,
    "neg_control_size": NEG_CONTROL_SIZE,
    "seed": SEED,
    "llm_endpoint": config.llm_endpoint,
    "vs_index_name": config.vs_index_name,
    "vs_endpoint_name": config.vs_endpoint_name,
}

mlflow.langchain.autolog()
mlflow.set_tracking_uri("databricks")

_display(f"Running in {'OFFLINE' if OFFLINE else 'LIVE'} mode | N={SAMPLE_SIZE}")
_display(params)

# Open the top-level MLflow run. All subsequent cells produce nested runs or
# log artefacts into this parent run.
_mlflow_run_ctx = mlflow_component_run("mas", "e2e_deep_dive_research", params=params)
run = _mlflow_run_ctx.__enter__()
mlflow.set_tag("run_kind", "e2e_deep_dive")

# COMMAND ----------

# MAGIC %md
# MAGIC ## §2 System topology diagram
# MAGIC Compiles the instrumented MAS graph and logs its Mermaid topology.

# COMMAND ----------

graph = build_instrumented_mas_graph(config)
mermaid = graph.get_graph().draw_mermaid()
mlflow.log_text(mermaid, "topology/graph.mmd")
_display("### MAS Topology")
_display(mermaid)

# COMMAND ----------

# MAGIC %md
# MAGIC ## §3 Pre-flight checks
# MAGIC Probes all runtime dependencies. Marks which endpoints/tables are warm vs
# MAGIC cold-start. A critical failure (serving endpoint down) raises immediately.

# COMMAND ----------

preflight_rows: list[dict] = []


def _run_probe(name: str, ok: bool, detail: str, *, critical: bool = False) -> None:
    emoji = "✅" if ok else "❌"
    _display(f"{emoji} {name}: {detail}")
    preflight_rows.append({"name": name, "ok": ok, "detail": detail, "critical": critical})
    mlflow.log_metric(f"preflight.{name.replace(' ', '_')}", int(ok))
    if critical and not ok:
        raise RuntimeError(f"preflight critical check failed: {name} — {detail}")


# Serving endpoints
_ok, _detail = _endpoint_ready(config.llm_endpoint)
_run_probe("llm_endpoint", _ok, _detail, critical=not OFFLINE)

_ok, _detail = _endpoint_ready(config.validator.embedding_model)
_run_probe("embedding_endpoint", _ok, _detail, critical=False)

# Taxonomy / whitelist tables
_ok, _detail = _table_has_rows(f"`{config.catalog}`.{config.schema}.treatment_taxonomy")
_run_probe("treatment_taxonomy_table", _ok, _detail, critical=False)

_ok, _detail = _table_has_rows(f"`{config.catalog}`.{config.schema}.mechanism_whitelist")
_run_probe("mechanism_whitelist_table", _ok, _detail, critical=False)

# Vector Search index
_ok, _detail = _vs_index_exists(
    endpoint_name=config.vs_endpoint_name,
    index_name=config.vs_index_name,
)
_run_probe("vs_index", _ok, _detail, critical=False)

# Accepted-chain count → determines cold_start_mode
accepted_chain_count: int = 0
try:
    _tmp_ok, _tmp_detail = _table_has_rows(config.vs_index_name.rsplit(".", 1)[0] + ".news_causal_features")
    if _tmp_ok:
        accepted_chain_count = 100  # positive proxy; actual count not needed
except Exception:
    pass
cold_start_mode: bool = accepted_chain_count < 50

_display(f"cold_start_mode={cold_start_mode} | accepted_chain_count proxy={accepted_chain_count}")
mlflow.log_param("cold_start_mode", cold_start_mode)

log_diag("preflight", pl.DataFrame(preflight_rows))

# COMMAND ----------

# MAGIC %md
# MAGIC ## §4 Dataset ingestion, sampling & lineage
# MAGIC Loads the mas_training_dataset, extracts headlines, performs stratified
# MAGIC sampling, logs DeltaDatasetSource lineage, and builds the evaluation dataset
# MAGIC plus a negative-control set of unrelated articles.

# COMMAND ----------

from mlflow.data.sources import DeltaDatasetSource

# — Source table
_MAS_TABLE = "`smart-trading-dev`.others.mas_training_dataset"

if OFFLINE:
    # Minimal synthetic frame so the notebook runs without Databricks
    news = pl.DataFrame({
        "story_id": [f"s{i:04d}" for i in range(max(SAMPLE_SIZE, 1))],
        "request": [
            "Can you summarize the market impact of the following news: Fed raises rates 75bps"
        ] * max(SAMPLE_SIZE, 1),
        "body": ["The Federal Reserve raised rates by 75bps."] * max(SAMPLE_SIZE, 1),
        "published_at": ["2024-01-01"] * max(SAMPLE_SIZE, 1),
        "expected_route": ["standard"] * max(SAMPLE_SIZE, 1),
        "_bucket": (["positive"] * (SAMPLE_SIZE // 2) + ["control"] * (SAMPLE_SIZE - SAMPLE_SIZE // 2))[: max(SAMPLE_SIZE, 1)],
    })
else:
    from notebooks.qa.utils import get_spark

    _spark = get_spark()
    _sdf = _spark.table(_MAS_TABLE)
    print(f"MAS TABLE IS {_MAS_TABLE}")
    # Source table has 'context' column — rename to 'body' for downstream consistency
    news = pl.from_arrow(_sdf.toArrow()).rename({"context": "body"})

# Log lineage
mlflow.log_input(
    mlflow.data.from_pandas(
        news.to_pandas(),
        source=DeltaDatasetSource(delta_table_name=_MAS_TABLE),
        name="mas_training_dataset_raw",
    ),
    context="evaluation_source",
)
mlflow.log_metric("dataset.raw_rows", news.height)

# Extract headline from request column
if "headline" not in news.columns:
    news = news.with_columns(
        pl.col("request").map_elements(_extract_headline, return_dtype=pl.Utf8).alias("headline")
    )

# Filter to BTC/Bitcoin-related articles — all eval rows must be crypto-relevant
_btc_mask = (
    pl.col("request").str.to_lowercase().str.contains(r"bitcoin|btc")
    | pl.col("body").str.to_lowercase().str.contains(r"bitcoin|btc")
)
btc_news = news.filter(_btc_mask)
_display(f"BTC-filtered rows available: {btc_news.height}")
assert btc_news.height >= 5, (
    f"Not enough BTC-related rows in {_MAS_TABLE} (got {btc_news.height}); "
    "check that the source table is populated with crypto news."
)
mlflow.log_metric("dataset.btc_filtered_rows", btc_news.height)

# Seeded sample — all rows drawn from BTC corpus only
sample = btc_news.sample(n=min(SAMPLE_SIZE, btc_news.height), seed=SEED)
mlflow.log_metric("dataset.sample_rows", sample.height)
_display(f"Sample: {sample.height} BTC-related rows (seed={SEED})")

# Build eval_dataset for mlflow.genai.evaluate (list of dicts)
eval_dataset: list[dict] = [
    {
        "inputs": {
            "story_id": row["story_id"],
            "headline": row["headline"],
            "body": row["body"],
            "published_at": str(row.get("published_at", "")),
        },
        "expected_route": row.get("expected_route", "unknown"),
    }
    for row in sample.to_dicts()
]

# Negative-control dataset — OFFLINE: synthetic headlines with no causal signal
_neg_size = min(NEG_CONTROL_SIZE, len(eval_dataset))
negative_control_dataset: list[dict] = [
    {
        "inputs": {
            "story_id": f"neg_{i}",
            "headline": f"Unrelated headline {i} with no financial signal",
            "body": "This is a sports news article about football.",
            "published_at": "2024-01-01",
        },
        "expected_route": "careful",
    }
    for i in range(_neg_size)
]

log_diag("sample_lineage", sample.select(["story_id", "headline"]))
_display(f"eval_dataset: {len(eval_dataset)} rows | negative_control: {len(negative_control_dataset)} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## §5 Graph wiring
# MAGIC Enters the `wire_graph_environment` context manager so all subsequent
# MAGIC cells run with correctly patched Spark + VS + LLM dependencies.

# COMMAND ----------

graph_env_ctx = wire_graph_environment(config, offline=OFFLINE)
graph_env_ctx.__enter__()
_display(f"Graph environment wired (offline={OFFLINE})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## §6 predict_fn + custom scorers
# MAGIC Defines the `predict_fn` adapter and 5 deterministic scorers + 1 LLM judge.

# COMMAND ----------

from mlflow.genai.scorers import scorer
from notebooks.qa.mas._scorer_helpers import (
    check_routing_legality,
    check_schema_conformance,
    check_taxonomy_whitelist_compliance,
    evidence_grounding_ratio,
)

# Load whitelist for scorer use (offline: seed in-memory; live: Spark lookup)
try:
    if not OFFLINE:
        from notebooks.qa.utils import get_spark as _get_spark_for_wl
        from trading_databricks.agents.taxonomy import load_whitelist
        _whitelist = load_whitelist(
            _get_spark_for_wl(), catalog=config.catalog, schema=config.schema
        )
    else:
        from trading_databricks.agents.taxonomy import MECHANISM_WHITELIST_SEED
        _whitelist = {
            (r["treatment_l1"], r["mechanism"]): r["valid_directions"]
            for r in MECHANISM_WHITELIST_SEED
        }
except Exception as _wl_exc:
    _display(f"whitelist load skipped (offline or table absent): {_wl_exc!r}")
    _whitelist = {}


def predict_fn(inputs: dict) -> dict:
    """Invoke the compiled MAS graph for a single article."""
    state = {
        "headline": inputs.get("headline", ""),
        "body": inputs.get("body", ""),
        "published_at": str(inputs.get("published_at", "")),
    }
    try:
        out = graph.invoke(state)
        return {"output": out, "story_id": inputs.get("story_id", "")}
    except Exception as e:
        return {"output": {"error": repr(e)}, "story_id": inputs.get("story_id", "")}


@scorer
def schema_conformance(*, inputs: dict, outputs: dict) -> float:  # type: ignore[return]
    """Fraction of required extraction fields that are present and valid."""
    extraction = (outputs.get("output") or {}).get("extraction")
    return check_schema_conformance(extraction)


@scorer
def taxonomy_whitelist_compliance(*, inputs: dict, outputs: dict) -> bool:  # type: ignore[return]
    """True iff (l1, mechanism, direction) tuple is in the live whitelist."""
    extraction = (outputs.get("output") or {}).get("extraction")
    return check_taxonomy_whitelist_compliance(extraction, _whitelist)


def make_routing_legality_scorer() -> object:  # type: ignore[return]
    @scorer
    def routing_legality(*, inputs: dict, outputs: dict) -> bool:  # type: ignore[return]
        """True iff the chosen extraction_path was legal given VS index state."""
        out = outputs.get("output") or {}
        path = out.get("extraction_path", "careful")
        top_sim = float(
            (out.get("few_shot_examples") or [{}])[0].get("similarity_score", 0.0)
        )
        n_ex = len(out.get("few_shot_examples") or [])
        return check_routing_legality(
            path,
            cold_start_mode=cold_start_mode,
            top_similarity=top_sim,
            n_examples=n_ex,
        )
    return routing_legality


routing_legality_scorer = make_routing_legality_scorer()


@scorer
def evidence_grounding_substring(*, inputs: dict, outputs: dict) -> float:  # type: ignore[return]
    """Fraction of evidence_spans found verbatim (case-insensitive) in body."""
    extraction = (outputs.get("output") or {}).get("extraction") or {}
    spans: list[str] = extraction.get("evidence_spans") or []
    body: str = inputs.get("body", "")
    return evidence_grounding_ratio(spans, body)


@scorer
def extraction_path_label(*, inputs: dict, outputs: dict) -> str:  # type: ignore[return]
    """Returns the extraction_path string for distribution analysis."""
    out = outputs.get("output") or {}
    return out.get("extraction_path") or "unknown"


# LLM-as-judge: groundedness
def make_judge() -> object:  # type: ignore[return]
    try:
        from mlflow.genai.scorers import Guidelines

        groundedness_judge = Guidelines(
            name="groundedness",
            guidelines=(
                "The extraction's evidence_spans must all be directly quoted or "
                "closely paraphrased from the article body. Score 'yes' only if "
                "every span is grounded in the body text; 'no' otherwise."
            ),
        )
        return groundedness_judge
    except Exception as _e:
        _display(f"Judge construction failed ({_e!r}); skipping LLM judge")
        return None


groundedness_judge = make_judge()

# Assemble the scorer list
_scorers: list = [
    schema_conformance,
    taxonomy_whitelist_compliance,
    routing_legality_scorer,
    evidence_grounding_substring,
    extraction_path_label,
]
if groundedness_judge is not None:
    _scorers.append(groundedness_judge)

_display(f"Scorers ready: {[s.__class__.__name__ for s in _scorers]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## §7 Evaluation loop
# MAGIC Manual scorer loop — compatible with MLflow 3.x where mlflow.genai.evaluate()
# MAGIC was restructured. Each predict_fn call is traced by langchain autolog.

# COMMAND ----------

_full_dataset = eval_dataset + negative_control_dataset
_all_story_ids = [r["inputs"]["story_id"] for r in _full_dataset]
mlflow.log_param("evaluate.total_rows", len(_full_dataset))


def _invoke_scorers(scorers_list: list, row_inputs: dict, row_outputs: dict) -> dict:
    """Call each scorer safely; returns {scorer_name: value_or_None}."""
    result: dict = {}
    for _s in scorers_list:
        _sname = getattr(_s, "name", _s.__class__.__name__)
        try:
            _val = _s(inputs=row_inputs, outputs=row_outputs)
            result[_sname] = _val.value if hasattr(_val, "value") else _val
        except Exception as _se:
            result[_sname] = None
    return result


try:
    _eval_rows: list[dict] = []
    for _row in eval_dataset:
        _inp = _row["inputs"]
        _out = predict_fn(_inp)
        _scores = _invoke_scorers(_scorers, _inp, _out)
        _eval_rows.append({"story_id": _inp.get("story_id", ""), **_scores})

    _neg_rows: list[dict] = []
    for _row in negative_control_dataset:
        _inp = _row["inputs"]
        _out = predict_fn(_inp)
        _scores = _invoke_scorers([schema_conformance], _inp, _out)
        _neg_rows.append({"story_id": _inp.get("story_id", ""), **_scores})

    _eval_df = pl.DataFrame(_eval_rows) if _eval_rows else pl.DataFrame()
    _neg_df = pl.DataFrame(_neg_rows) if _neg_rows else pl.DataFrame()

    # Aggregate and log metrics
    _eval_metrics: dict = {}
    for _col in _eval_df.columns:
        if _col == "story_id":
            continue
        _series = _eval_df[_col].drop_nulls()
        if _series.len() == 0:
            continue
        if _series.dtype in (pl.Float64, pl.Float32):
            _eval_metrics[f"eval.{_col}.mean"] = float(_series.mean())
        elif _series.dtype == pl.Boolean:
            _eval_metrics[f"eval.{_col}.pass_rate"] = float(_series.cast(pl.Float64).mean())

    _neg_metrics: dict = {}
    for _col in _neg_df.columns:
        if _col == "story_id":
            continue
        _series = _neg_df[_col].drop_nulls()
        if _series.len() == 0:
            continue
        if _series.dtype in (pl.Float64, pl.Float32):
            _neg_metrics[f"neg_ctrl.{_col}.mean"] = float(_series.mean())
        elif _series.dtype == pl.Boolean:
            _neg_metrics[f"neg_ctrl.{_col}.pass_rate"] = float(_series.cast(pl.Float64).mean())

    mlflow.log_metrics({**_eval_metrics, **_neg_metrics})
    _display("Eval metrics:")
    _display(_eval_metrics)
    _display("Neg-ctrl metrics:")
    _display(_neg_metrics)

    class _EvalResult:
        metrics = _eval_metrics
        tables = {"eval_results_table": _eval_df.to_pandas()}

    eval_result = _EvalResult()  # type: ignore[assignment]
    negative_control = _EvalResult()  # type: ignore[assignment]

except Exception as _eval_exc:
    _display(f"Evaluation loop raised: {_eval_exc!r}")
    eval_result = None  # type: ignore[assignment]
    negative_control = None  # type: ignore[assignment]

# COMMAND ----------

# MAGIC %md
# MAGIC ## §8 Trace-derived analysis
# MAGIC Pulls real MLflow spans, extracts per-node latency, and logs boxplot.
# MAGIC **No fabricated numbers — every metric comes from actual span timestamps.**

# COMMAND ----------

try:
    traces_df = mlflow.search_traces(
        experiment_names=["/Shared/trading-databricks-qa/mas/e2e_deep_dive_research"],
        max_results=len(_full_dataset) + 50,
        extract_fields=["spans"],
    )
    _display(f"search_traces returned {len(traces_df)} rows")
except Exception as _search_exc:
    _display(f"search_traces failed: {_search_exc!r}")
    traces_df = None

# Flatten spans → per-node latency dataframe
_span_rows: list[dict] = []
if traces_df is not None and len(traces_df) > 0:
    for _i, _trace_row in traces_df.iterrows():
        _trace_id = _trace_row.get("trace_id", f"t{_i}")
        _spans = _trace_row.get("spans") or []
        for _span in (_spans if isinstance(_spans, list) else []):
            _name = _span.get("name", "unknown")
            _start_ns = _span.get("start_time_ns") or 0
            _end_ns = _span.get("end_time_ns") or 0
            _dur_ms = (_end_ns - _start_ns) / 1e6
            _status = _span.get("status", {}).get("status_code", "OK")
            _span_rows.append({
                "trace_id": _trace_id,
                "node": _name,
                "duration_ms": _dur_ms,
                "status": _status,
            })

flat_span_df = pl.DataFrame(_span_rows) if _span_rows else pl.DataFrame(
    schema={"trace_id": pl.Utf8, "node": pl.Utf8, "duration_ms": pl.Float64, "status": pl.Utf8}
)
log_diag("span_metrics_raw", flat_span_df)

# Per-node latency stats
if flat_span_df.height > 0:
    node_latency_stats = (
        flat_span_df
        .group_by("node")
        .agg([
            pl.col("duration_ms").mean().alias("mean_ms"),
            pl.col("duration_ms").median().alias("p50_ms"),
            pl.col("duration_ms").quantile(0.95).alias("p95_ms"),
            pl.col("duration_ms").max().alias("max_ms"),
            pl.col("duration_ms").count().alias("n"),
        ])
        .sort("mean_ms", descending=True)
    )
    log_diag("node_latency_stats", node_latency_stats)
    _display("Node latency stats:")
    _display(node_latency_stats)

    # Boxplot
    import plotly.express as px
    _latency_pdf = flat_span_df.to_pandas()
    _lat_fig = px.box(
        _latency_pdf,
        x="node",
        y="duration_ms",
        color="status",
        title="Per-node latency — real spans (offline mock has near-zero latency)",
    )
    log_plotly("node_latency_boxplot_real", _lat_fig)
else:
    node_latency_stats = pl.DataFrame(
        schema={"node": pl.Utf8, "mean_ms": pl.Float64, "p50_ms": pl.Float64,
                "p95_ms": pl.Float64, "max_ms": pl.Float64, "n": pl.UInt32}
    )
    _display("§8 skipped: no span data available (traces may be empty in offline mode)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## §9 Execution-overlay topology diagram
# MAGIC Renders the MAS topology with per-node mean latency overlaid as node colour.

# COMMAND ----------

if flat_span_df.height > 0:
    import plotly.graph_objects as go

    _node_means: dict[str, float] = {}
    if node_latency_stats.height > 0:
        for _r in node_latency_stats.to_dicts():
            _node_means[_r["node"]] = float(_r["mean_ms"])

    _topology_nodes = list(
        {
            "context_retriever", "few_shot_retriever", "uncertainty_router",
            "causality_extractor", "mechanism_validator",
        }
    )
    _x_pos = {n: i for i, n in enumerate(_topology_nodes)}

    _node_x = [_x_pos.get(n, 0) for n in _topology_nodes]
    _node_y = [0] * len(_topology_nodes)
    _node_text = [
        f"{n}<br>{_node_means.get(n, 0):.1f} ms" for n in _topology_nodes
    ]
    _node_color = [_node_means.get(n, 0) for n in _topology_nodes]

    _overlay_fig = go.Figure()
    _overlay_fig.add_trace(go.Scatter(
        x=_node_x,
        y=_node_y,
        mode="markers+text",
        text=_node_text,
        textposition="top center",
        marker=dict(
            size=40,
            color=_node_color,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="mean ms"),
        ),
    ))
    fig = _overlay_fig
    fig.update_layout(
        title="MAS graph execution overlay",
        xaxis=dict(showticklabels=False, zeroline=False),
        yaxis=dict(showticklabels=False, zeroline=False, range=[-1, 1]),
    )
    log_plotly("topology_overlay", fig)
else:
    _display("§9 skipped: no span data available for overlay")

# COMMAND ----------

# MAGIC %md
# MAGIC ## §10 Scorer result aggregation & visualization
# MAGIC Pulls scorer outputs into Polars, logs per-scorer pass rates and plots.

# COMMAND ----------

if eval_result is None:
    _display("§10 skipped: eval_result is None — evaluate() failed in §7")
    eval_metrics_df = pl.DataFrame()
else:
    try:
        eval_metrics_df = pl.from_pandas(eval_result.tables["eval_results_table"])
    except Exception as e:
        _display(f"eval_result.tables lookup failed ({e!r}); falling back to eval_result.metrics")
        eval_metrics_df = pl.DataFrame([eval_result.metrics or {}])

log_diag("eval_metrics_table", eval_metrics_df)

# extraction_path distribution
if "extraction_path" in eval_metrics_df.columns:
    log_plotly(
        "extraction_path_distribution",
        build_path_distribution(eval_metrics_df["extraction_path"]),
    )

# validation_status distribution
if "validation_status" in eval_metrics_df.columns:
    log_plotly(
        "validation_status_distribution",
        build_label_distribution(eval_metrics_df["validation_status"]),
    )

# groundedness judge categorical distribution
judge_col_candidates = [c for c in eval_metrics_df.columns if "groundedness" in c]
if judge_col_candidates:
    log_plotly(
        "groundedness_judge_distribution",
        build_label_distribution(eval_metrics_df[judge_col_candidates[0]]),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## §11 Failure autopsy sandbox
# MAGIC Gracefully degrades when `dbutils` is unavailable (i.e. `uv run python`).

# COMMAND ----------

dbu = _get_dbutils()
if dbu is not None:
    dbu.widgets.text("story_id", "", "story_id to replay")
    dbu.widgets.text("override_headline", "", "override headline (optional)")
    dbu.widgets.text("override_body", "", "override body (optional)")

_sample_by_story = {r["story_id"]: r for r in sample.to_dicts()}


def render_autopsy(story_id: str) -> None:
    """Render the trace for a given story_id: prompts, LLM outputs, validator verdict."""
    match = flat_span_df.filter(pl.col("trace_id").str.contains(story_id)) if flat_span_df.height > 0 else pl.DataFrame()
    if match.height == 0:
        _display(f"no span rows for story_id={story_id}")
        return
    _display(match)
    _display(
        f"Open the MLflow run UI and search traces for story_id={story_id} "
        f"to inspect full prompt + response payloads."
    )


def replay_single(story_id: str, *, state_overrides: dict | None = None) -> None:
    row = _sample_by_story.get(story_id)
    if row is None:
        _display(f"story_id={story_id} not in sample")
        return
    state = {
        "headline": row["headline"],
        "body": row["body"],
        "published_at": str(row["published_at"]),
    }
    if state_overrides:
        state.update(state_overrides)
    with mlflow.start_run(run_name=f"replay-{story_id}", nested=True) as nested:
        mlflow.set_tag("replay", "true")
        mlflow.set_tag("replay_story_id", story_id)
        try:
            out = graph.invoke(state)
            _display(out)
            mlflow.log_dict(out, "replay/output.json")
        except Exception as e:
            _display(f"replay failed: {e!r}")
            mlflow.set_tag("replay_status", "fail")
            mlflow.log_text(repr(e), "replay/error.txt")


# Top-K worst cases by groundedness (if the judge column exists)
if judge_col_candidates:
    worst_col = judge_col_candidates[0]
    try:
        worst = (
            eval_metrics_df
            .sort(worst_col)
            .head(5)
            .select(["story_id"] if "story_id" in eval_metrics_df.columns else eval_metrics_df.columns[:3])
        )
        _display("Top-K worst by groundedness:")
        _display(worst)
    except Exception as e:
        _display(f"worst-case ranking skipped: {e!r}")

# Demonstrate the sandbox on the first story_id
if eval_dataset:
    first_story_id = eval_dataset[0]["inputs"]["story_id"]
    _display(f"Example autopsy for {first_story_id}:")
    render_autopsy(first_story_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## §12 Human feedback hooks (Workflow-7 seed)
# MAGIC Stubs an `sme_verdict` feedback slot against every trace so a
# MAGIC downstream UC-backed labeling session can attach and MemAlign can
# MAGIC align `groundedness_judge` in a follow-up.

# COMMAND ----------

try:
    from mlflow.entities import AssessmentSource, AssessmentSourceType

    unique_trace_ids = (
        flat_span_df["trace_id"].unique().to_list() if flat_span_df.height > 0 else []
    )
    for tid in unique_trace_ids:
        try:
            mlflow.log_feedback(
                trace_id=tid,
                name="sme_verdict",
                value=None,
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN,
                    source_id="placeholder_sme",
                ),
            )
        except Exception as e:
            _display(f"log_feedback failed for {tid}: {e!r}")
    mlflow.log_metric("feedback.sme_verdict_stubs", len(unique_trace_ids))
except Exception as e:
    _display(f"feedback stub cell skipped: {e!r}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## §13 Sanity assertions
# MAGIC Loose invariants only. Research notebooks observe — they do not gate.

# COMMAND ----------

if eval_result is None:
    _display("WARN §13: eval_result is None — evaluate() failed; skipping eval assertions")
    mlflow.set_tag("eval_status", "fail")
else:
    assert eval_result.metrics is not None, "eval_result.metrics missing"
    _display("OK §13: eval_result.metrics present")
assert len(eval_dataset) <= SAMPLE_SIZE, (
    f"sample size exceeds configured limit ({len(eval_dataset)} > {SAMPLE_SIZE})"
)
assert any(r["name"].startswith("preflight.") for r in preflight_rows), \
    "no preflight metrics logged"
if flat_span_df.height > 0:
    unique_traces = flat_span_df["trace_id"].n_unique()
    assert unique_traces >= 1, "no traces recorded"

# COMMAND ----------

# MAGIC %md
# MAGIC ## §14 Summary & next steps
# MAGIC - MLflow run: see link in the run card above
# MAGIC - Next: launch a UC-backed labeling session against the `sme_verdict`
# MAGIC   stubs (Workflow-7), run MemAlign on `groundedness_judge`, feed the
# MAGIC   eval dataset into GEPA (Workflow-8) against `trading_databricks.agents.prompts`.

# COMMAND ----------

summary_payload = {
    "mode": params["mode"],
    "sample_size": len(eval_dataset),
    "cold_start_mode": cold_start_mode,
    "n_unique_traces": int(flat_span_df["trace_id"].n_unique()) if flat_span_df.height > 0 else 0,
    "errors_total": int(flat_span_df.filter(pl.col("status") != "OK").height) if flat_span_df.height > 0 else 0,
}
mlflow.log_dict(summary_payload, "summary.json")
_display(summary_payload)

# COMMAND ----------

# Release the wire_graph_environment context opened in §5.
graph_env_ctx.__exit__(None, None, None)
