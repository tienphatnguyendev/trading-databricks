# Databricks notebook source

# MAGIC %pip install -q -e .
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC # MAS E2E Deep-Dive Research Notebook
# MAGIC
# MAGIC **Purpose:** Live Databricks environment test — run the full MAS graph against real
# MAGIC LLM endpoints and the real `raw_news` table, logging MLflow quality metrics for human review.
# MAGIC
# MAGIC **Sections:** Setup → Preflight → Data Load → Graph Execution → Sub-pipeline Diagnostics
# MAGIC → Extraction Quality Scorers → Visualisations → Sanity Assertions
# MAGIC
# MAGIC **MLflow experiment:** `/Shared/mas-deep-dive`

# COMMAND ----------

# MAGIC %md
# MAGIC ## §1 Setup

# COMMAND ----------

import os
import mlflow
import mlflow.langchain
import polars as pl
import plotly.express as px

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.graph import build_mas_graph
from trading_databricks.common.uc_identifiers import (
    GOLD_SCHEMA,
    MECHANISM_WHITELIST_TABLE,
    TREATMENT_TAXONOMY_TABLE,
    quoted_table_fqn,
)
from notebooks.qa.utils import (
    _display,
    _endpoint_ready,
    _table_has_rows,
    _vs_index_exists,
    get_spark,
    log_diag,
    log_plotly,
    wire_graph_environment,
)
from notebooks.qa.mas._scorer_helpers import (
    check_schema_conformance,
    check_taxonomy_whitelist_compliance,
    check_routing_legality,
    evidence_grounding_ratio,
    check_evidence_quality,
    check_judge_all_pass,
    check_reflection_triggered,
    check_extraction_ok,
)

SAMPLE_SIZE: int = int(os.getenv("MAS_SAMPLE_SIZE", "50"))
SEED: int = int(os.getenv("MAS_SEED", "42"))
SYMBOL: str = os.getenv("MAS_SYMBOL", "BTCUSDT")

config = MASConfig()

mlflow.set_experiment("/Shared/mas-deep-dive")
mlflow.langchain.autolog()

run = mlflow.start_run(tags={"notebook": "e2e_deep_dive", "symbol": SYMBOL})
mlflow.log_params({
    "sample_size": SAMPLE_SIZE,
    "seed": SEED,
    "symbol": SYMBOL,
    "llm_endpoint": config.llm_endpoint,
})
_display(f"MLflow run: {run.info.run_id} | N={SAMPLE_SIZE} | SYMBOL={SYMBOL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## §2 Preflight
# MAGIC
# MAGIC - **LLM endpoint** — critical: raises `RuntimeError` immediately on failure
# MAGIC - **Embedding endpoint** — warn only (logs `preflight.embedding_ok=0`)
# MAGIC - **Whitelist / taxonomy tables** — warn only
# MAGIC - **VS index** — warn only (few-shot retrieval may be degraded)

# COMMAND ----------

def _probe(name: str, ok: bool, detail: str, *, critical: bool = False) -> None:
    status = "OK" if ok else "FAIL"
    _display(f"[{status}] {name}: {detail}")
    mlflow.log_metric(f"preflight.{name}", int(ok))
    if critical and not ok:
        raise RuntimeError(f"Preflight critical failure: {name} — {detail}")


ok, detail = _endpoint_ready(config.llm_endpoint)
_probe("llm_ok", ok, detail, critical=True)

ok, detail = _endpoint_ready(config.validator.embedding_model)
_probe("embedding_ok", ok, detail)

ok, detail = _table_has_rows(
    quoted_table_fqn(config.catalog, GOLD_SCHEMA, TREATMENT_TAXONOMY_TABLE)
)
_probe("taxonomy_ok", ok, detail)

ok, detail = _table_has_rows(
    quoted_table_fqn(config.catalog, GOLD_SCHEMA, MECHANISM_WHITELIST_TABLE)
)
_probe("whitelist_ok", ok, detail)

ok, detail = _vs_index_exists(
    endpoint_name=config.vs_endpoint_name,
    index_name=config.vs_index_name,
)
_probe("vs_index_ok", ok, detail)

# COMMAND ----------

# MAGIC %md
# MAGIC ## §3 Data Load
# MAGIC
# MAGIC Source: `smart-trading-dev.01_bronze.raw_news`
# MAGIC
# MAGIC Crypto-keyword filter (BTC / ETH headlines and bodies); `symbol` injected into
# MAGIC every row so `rag_retriever` can perform OHLC joins when the firehose index is live.

# COMMAND ----------

_raw_sdf = spark.table("smart-trading-dev.01_bronze.raw_news")
_display(f"raw_news schema: {dict(_raw_sdf.dtypes)}")

_keyword_filter = (
    "lower(headline) LIKE '%bitcoin%' OR lower(headline) LIKE '%btc%' "
    "OR lower(body) LIKE '%bitcoin%' OR lower(body) LIKE '%btc%' "
    "OR lower(headline) LIKE '%ethereum%' OR lower(headline) LIKE '%eth%'"
)

_filtered_pdf = _raw_sdf.filter(_keyword_filter).limit(SAMPLE_SIZE * 5).toPandas()

import random as _random
_random.seed(SEED)
_sample_rows = _random.sample(
    _filtered_pdf.to_dict("records"),
    min(SAMPLE_SIZE, len(_filtered_pdf)),
)

articles: list[dict] = [
    {
        "headline": str(r.get("headline") or ""),
        "body": str(r.get("body") or r.get("content") or ""),
        "published_at": str(r.get("published_at") or ""),
        "symbol": SYMBOL,
    }
    for r in _sample_rows
]

mlflow.log_metric("data.raw_rows_fetched", len(_filtered_pdf))
mlflow.log_metric("data.sample_rows", len(articles))

assert len(articles) >= 1, (
    f"No crypto-relevant articles found in raw_news after filtering. "
    f"Check that the table has BTC/ETH-related rows."
)
_display(f"Loaded {len(articles)} articles (crypto-filtered, seed={SEED})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## §4 Graph Execution
# MAGIC
# MAGIC Runs the full MAS graph against live LLM endpoints. The `wire_graph_environment`
# MAGIC context is fully contained in this cell — no multi-cell span.
# MAGIC
# MAGIC Exceptions per-article are caught and recorded as `extraction_status="error"`.

# COMMAND ----------

results: list[dict] = []

with wire_graph_environment(config, offline=False):
    graph = build_mas_graph(config=config)
    for _article in articles:
        try:
            _out = graph.invoke(_article)
            results.append({**_article, **_out})
        except Exception as _exc:
            results.append({**_article, "extraction_status": "error", "error": repr(_exc)})

df = pl.DataFrame(results)
_display_cols = [
    c for c in [
        "headline", "symbol", "extraction_path", "validation_status",
        "extraction_status", "evidence_status", "precedent_status", "reflection_count",
    ] if c in df.columns
]
display(df.select(_display_cols).to_pandas())

# COMMAND ----------

# MAGIC %md
# MAGIC ## §5 Sub-pipeline Diagnostics
# MAGIC
# MAGIC Aggregates the 8 new `MASState` keys introduced in TRD-68.
# MAGIC
# MAGIC > **Expected:** `eval.precedent_ok_rate ≈ 0` — `news_firehose_idx` is not yet
# MAGIC > provisioned; `rag_retriever` always returns `precedent_status="unavailable"`.
# MAGIC > This is documented infrastructure gap, not a bug.

# COMMAND ----------

def _col_rate(col: str, value: str) -> float:
    """Return fraction of non-null rows where col == value."""
    if col not in df.columns:
        return 0.0
    s = df[col].drop_nulls()
    return float((s == value).sum()) / s.len() if not s.is_empty() else 0.0


def _col_bool_mean(col: str) -> float:
    """Mean of a boolean-like column (True/False or 1/0), null-safe."""
    if col not in df.columns:
        return 0.0
    s = df[col].drop_nulls()
    return float(s.cast(pl.Float64).mean() or 0.0) if not s.is_empty() else 0.0


eval_metrics: dict[str, float] = {
    "eval.evidence_ok_rate": _col_rate("evidence_status", "ok"),
    "eval.precedent_ok_rate": _col_rate("precedent_status", "ok"),
    "eval.extraction_ok_rate": float(
        df["extraction_status"]
        .drop_nulls()
        .is_in(["ok", "ok_after_reflection"])
        .cast(pl.Float64)
        .mean() or 0.0
    ) if "extraction_status" in df.columns else 0.0,
    "eval.judge_all_pass_rate": float(
        df["judge_failures"].is_null().cast(pl.Float64).mean() or 0.0
    ) if "judge_failures" in df.columns else 0.0,
    "eval.reflection_triggered_rate": float(
        df["reflection_count"].fill_null(0).gt(0).cast(pl.Float64).mean() or 0.0
    ) if "reflection_count" in df.columns else 0.0,
}
mlflow.log_metrics(eval_metrics)

# extraction_status distribution table
if "extraction_status" in df.columns:
    _status_dist = (
        df.group_by("extraction_status")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    log_diag("extraction_status_distribution", _status_dist)

# judge failure detail
_judge_rows: list[dict] = []
if "judge_failures" in df.columns:
    for _r in df.filter(pl.col("judge_failures").is_not_null()).to_dicts():
        for _judge_name, _reason in (_r.get("judge_failures") or {}).items():
            _judge_rows.append({"judge": _judge_name, "reason": _reason})
    if _judge_rows:
        log_diag("judge_failures_detail", pl.DataFrame(_judge_rows))

_display(eval_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## §6 Extraction Quality Scorers
# MAGIC
# MAGIC Applies all 8 scorer predicates per article and logs aggregate MLflow metrics.

# COMMAND ----------

from trading_databricks.agents.taxonomy import load_whitelist

_whitelist: dict = {}
try:
    _whitelist = load_whitelist(get_spark(), catalog=config.catalog, schema=config.schema)
except Exception as _wl_exc:
    _display(f"whitelist load skipped: {_wl_exc!r}")

scorer_rows: list[dict] = []
for _row in df.to_dicts():
    _extraction = _row.get("extraction")
    _few_shot = _row.get("few_shot_examples") or []
    _top_sim = float(_few_shot[0].get("similarity_score", 0.0)) if _few_shot else 0.0
    scorer_rows.append({
        "headline": str(_row.get("headline") or "")[:60],
        "schema_conformance": check_schema_conformance(_extraction),
        "taxonomy_compliance": check_taxonomy_whitelist_compliance(_extraction, _whitelist),
        "routing_legality": check_routing_legality(
            _row.get("extraction_path") or "careful",
            cold_start_mode=False,
            top_similarity=_top_sim,
            n_examples=len(_few_shot),
        ),
        "evidence_grounding": evidence_grounding_ratio(
            (_extraction or {}).get("evidence_spans") or [],
            _row.get("body") or "",
        ),
        "evidence_quality": check_evidence_quality(_row.get("evidence_status")),
        "judge_all_pass": check_judge_all_pass(_row.get("judge_failures")),
        "reflection_triggered": check_reflection_triggered(_row.get("reflection_count")),
        "extraction_ok": check_extraction_ok(_row.get("extraction_status")),
    })

scorer_df = pl.DataFrame(scorer_rows)
log_diag("scorer_results_per_article", scorer_df)

scorer_metrics: dict[str, float] = {
    "scorer.schema_conformance_mean": float(scorer_df["schema_conformance"].mean() or 0.0),
    "scorer.taxonomy_compliance_rate": float(scorer_df["taxonomy_compliance"].cast(pl.Float64).mean() or 0.0),
    "scorer.routing_legality_rate": float(scorer_df["routing_legality"].cast(pl.Float64).mean() or 0.0),
    "scorer.evidence_grounding_mean": float(scorer_df["evidence_grounding"].mean() or 0.0),
    "scorer.evidence_quality_rate": float(scorer_df["evidence_quality"].cast(pl.Float64).mean() or 0.0),
    "scorer.judge_all_pass_rate": float(scorer_df["judge_all_pass"].cast(pl.Float64).mean() or 0.0),
    "scorer.reflection_triggered_rate": float(scorer_df["reflection_triggered"].cast(pl.Float64).mean() or 0.0),
    "scorer.extraction_ok_rate": float(scorer_df["extraction_ok"].cast(pl.Float64).mean() or 0.0),
}
mlflow.log_metrics(scorer_metrics)
display(scorer_df.to_pandas())

# COMMAND ----------

# MAGIC %md
# MAGIC ## §7 Visualisations
# MAGIC
# MAGIC All 4 charts are logged as MLflow artifacts via `log_plotly` so they persist
# MAGIC beyond the notebook session.

# COMMAND ----------

def _bar_col(col_name: str, title: str) -> None:
    if col_name not in df.columns:
        return
    _vc = df[col_name].drop_nulls().value_counts(sort=True)
    log_plotly(
        f"{col_name}_dist",
        px.bar(_vc.to_pandas(), x=col_name, y="count", title=title),
    )


_bar_col("extraction_path", "Extraction Path Distribution")
_bar_col("validation_status", "Validation Status Distribution")
_bar_col("extraction_status", "Extraction Status Distribution (TRD-68 sub-pipeline)")

if _judge_rows:
    _jdf = pl.DataFrame(_judge_rows)
    _jrate = (
        _jdf.group_by("judge")
        .agg(pl.len().alias("fail_count"))
        .sort("fail_count", descending=True)
    )
    log_plotly(
        "judge_failure_counts",
        px.bar(_jrate.to_pandas(), x="judge", y="fail_count",
               title="Judge Failure Counts by Judge"),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## §8 Sanity Assertions

# COMMAND ----------

assert df.height > 0, (
    "No results produced — graph invocation loop returned zero rows. "
    "Check §3 data load and §4 error handling."
)
assert "extraction_status" in df.columns and df["extraction_status"].null_count() == 0, (
    "extraction_status is null on at least one row. "
    "The error handler in §4 should always set it."
)

_error_count = int((df["extraction_status"] == "error").sum())
mlflow.log_metric("assertions.passed", 1)
mlflow.log_metric("graph.error_count", _error_count)

_display(
    f"All assertions passed. "
    f"errors={_error_count}/{df.height} | run={run.info.run_id}"
)

mlflow.end_run()
