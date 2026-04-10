from __future__ import annotations

import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from typing import Literal

import mlflow
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from databricks.connect import DatabricksSession

logger = logging.getLogger(__name__)

def get_spark():
    return DatabricksSession.builder.serverless(True).getOrCreate()

def load_ml_training_data(limit: int | None = None) -> pl.DataFrame:
    spark = get_spark()
    query = "SELECT * FROM `smart-trading-dev`.others.ml_training_data ORDER BY timestamp"
    args = {}
    if limit is not None:
        query += " LIMIT :limit"
        args["limit"] = limit
    return pl.from_arrow(spark.sql(query, args=args).toArrow())

def load_raw_aggtrades(limit: int | None = None) -> pl.DataFrame:
    df = load_ml_training_data(limit)
    if df.is_empty():
        return pl.DataFrame(schema={"timestamp": pl.Int64, "bar_start_time": pl.Datetime("us"), "symbol": pl.Utf8})
    return df.with_columns([
        pl.from_epoch(pl.col("timestamp"), time_unit="us").alias("bar_start_time"),
        pl.lit("BTCUSDT").alias("symbol")
    ])

def load_raw_dollar_bars(
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
    catalog: str = "trading-signal-gen-dev",
    limit: int | None = None,
) -> pl.DataFrame:
    spark = get_spark()
    # Catalog name cannot be parameterized, so we use identifier backticks
    # and ensure it's a simple string, though here it's usually internal.
    query = f"SELECT * FROM `{catalog}`.gold.dollar_bars WHERE symbol = :symbol"
    args = {"symbol": symbol}

    if start:
        query += " AND bar_end_time >= :start"
        args["start"] = start.isoformat()
    if end:
        query += " AND bar_end_time <= :end"
        args["end"] = end.isoformat()

    query += " ORDER BY bar_index"

    if limit:
        query += " LIMIT :limit"
        args["limit"] = limit

    try:
        return pl.from_arrow(spark.sql(query, args=args).toArrow())
    except Exception as e:
        raise RuntimeError(f"gold.dollar_bars not found in {catalog}. Run pipeline or use notebook 00. Error: {e}")

def load_article_sample(n: int = 10) -> list[dict]:
    spark = get_spark()
    query = "SELECT * FROM `smart-trading-dev`.others.mas_training_dataset ORDER BY story_id LIMIT :limit"
    # Use fixed seed sampling if needed, but for now just ordered limit
    pdf = spark.sql(query, args={"limit": n}).toPandas()
    records = pdf.to_dict(orient="records")
    results = []
    prefix = "Can you summarize the market impact of the following news: "
    for r in records:
        headline = r.get("request", "")
        if headline.startswith(prefix):
            headline = headline[len(prefix):]
        results.append({
            "story_id": r.get("story_id"),
            "headline": headline.strip(),
            "body": r.get("context"),
            "published_at": r.get("published_at")
        })
    return results

@contextmanager
def mlflow_component_run(layer: str, component: str, params: dict | None = None):
    user = os.environ.get("USER", "default_user")
    mlflow.set_experiment(f"/Users/{user}/trading-databricks-qa/{layer}/{component}")

    with mlflow.start_run() as run:
        if params:
            mlflow.log_params(params)
        mlflow.set_tags({"layer": layer, "component": component})
        try:
            yield run
            mlflow.set_tag("status", "pass")
        except Exception:
            mlflow.set_tag("status", "fail")
            raise

def log_diag(name: str, df: pl.DataFrame | pl.LazyFrame, metrics: dict | None = None) -> None:
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, f"{name}.parquet")
        df.write_parquet(path)
        mlflow.log_artifact(path, artifact_path="diagnostics")
    if metrics:
        mlflow.log_metrics(metrics)

def log_plotly(name: str, fig) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, f"{name}.html")
        fig.write_html(path)
        mlflow.log_artifact(path, artifact_path="plots")

def log_sample(name: str, df: pl.DataFrame, n: int = 100) -> None:
    log_diag(name, df.head(n))

def log_phase_metrics(phase: str, metrics: dict) -> None:
    prefixed = {f"{phase}_{k}": v for k, v in metrics.items()}
    mlflow.log_metrics(prefixed)

def summarize(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        series = df[c]
        rows.append({
            "col": c,
            "n": len(series),
            "null_count": series.null_count(),
            "min": series.min() if series.dtype.is_numeric() else None,
            "max": series.max() if series.dtype.is_numeric() else None,
            "mean": series.mean() if series.dtype.is_numeric() else None,
            "std": series.std() if series.dtype.is_numeric() else None,
            "p50": series.median() if series.dtype.is_numeric() else None,
        })
    return pl.DataFrame(rows)

def null_ratios(df: pl.DataFrame) -> pl.DataFrame:
    summ = summarize(df, df.columns)
    return summ.select([
        pl.col("col"),
        pl.col("null_count"),
        (pl.col("null_count") / pl.col("n")).alias("null_ratio")
    ])

def compare_to_expected(summary: pl.DataFrame, expected_ranges: dict) -> pl.DataFrame:
    rows = []
    for r in summary.iter_rows(named=True):
        col = r["col"]
        if col in expected_ranges:
            lo, hi = expected_ranges[col]
            passed = True
            if r["min"] is not None and r["min"] < lo:
                passed = False
            if r["max"] is not None and r["max"] > hi:
                passed = False
            rows.append({"col": col, "pass": passed, "expected_lo": lo, "expected_hi": hi})
    return pl.DataFrame(rows)

def build_scatter(df: pl.DataFrame, x: str, y: str, color: str | None = None, overlay: pl.DataFrame | None = None) -> go.Figure:
    pdf = df.to_pandas()
    fig = px.scatter(pdf, x=x, y=y, color=color)
    if overlay is not None:
        opdf = overlay.to_pandas()
        fig.add_trace(go.Scatter(x=opdf[x], y=opdf[y], mode="markers", marker=dict(size=10, symbol="triangle-up")))
    return fig

def build_candlestick(bars: pl.DataFrame) -> go.Figure:
    pdf = bars.to_pandas()
    return go.Figure(data=[go.Candlestick(
        x=pdf["bar_end_time"], open=pdf["open"], high=pdf["high"], low=pdf["low"], close=pdf["close"]
    )])

def build_histogram(series: pl.Series, bins: int = 30, title: str | None = None) -> go.Figure:
    fig = px.histogram(series.to_pandas(), nbins=bins, title=title)
    return fig

def build_box(df: pl.DataFrame, x: str, y: str, color: str | None = None) -> go.Figure:
    return px.box(df.to_pandas(), x=x, y=y, color=color)

def build_label_distribution(labels: pl.DataFrame | pl.Series) -> go.Figure:
    if isinstance(labels, pl.DataFrame):
        # Assume it's a pre-aggregated counts df or we just plot all boolean cols
        pdf = labels.to_pandas()
        return px.bar(pdf)
    # Series of categorical labels
    counts = labels.value_counts().to_pandas()
    return px.bar(counts, x=labels.name, y="count")

def build_path_distribution(routing_results: pl.Series | list[str]) -> go.Figure:
    if isinstance(routing_results, list):
        routing_results = pl.Series("path", routing_results)
    counts = routing_results.value_counts().to_pandas()
    return px.pie(counts, names=routing_results.name, values="count")

def build_equity_curve(pnl_series: np.ndarray) -> go.Figure:
    cum = np.cumprod(1 + pnl_series)
    return px.line(y=cum, title="Equity Curve")

def build_heatmap(features: pl.DataFrame, corr: bool = True) -> go.Figure:
    pdf = features.to_pandas()
    if corr:
        pdf = pdf.corr()
    return px.imshow(pdf)

NODE_NAMES = ("context_retriever", "few_shot_retriever", "uncertainty_router", "causality_extractor", "mechanism_validator")

def _raise_embed_error(text: str) -> list[float]:
    raise RuntimeError("embedding endpoint down")

def call_embedding(text: str) -> list[float]:
    if os.environ.get("MAS_QA_MOCK_EMBED") == "1":
        return [1.0, 0.0, 0.0]
    # In real notebook, would call mlflow.deployments
    return [1.0, 0.0, 0.0]

def call_claude(messages: list[dict], temperature=0.0, n=1, max_tokens=1000) -> list[str]:
    if os.environ.get("MAS_QA_MOCK_LLM") == "1":
        # Mock deterministic response based on path/n
        if n == 3:
            return [
                '{"treatment_class": "Macro.rate_decision", "mechanism": "risk_on", "expected_direction": "bullish", "confidence": 0.9}',
                '{"treatment_class": "Macro.rate_decision", "mechanism": "risk_on", "expected_direction": "bullish", "confidence": 0.8}',
                '{"treatment_class": "Regulation.etf_ruling", "mechanism": "institutional_access", "expected_direction": "bullish", "confidence": 0.7}'
            ]
        return ['{"treatment_class": "Macro.rate_decision", "mechanism": "risk_on", "expected_direction": "bullish", "confidence": 0.9}']

    import mlflow.deployments
    client = mlflow.deployments.get_deploy_client("databricks")
    # Using a placeholder endpoint name; should ideally come from config
    resp = client.predict(endpoint="databricks-claude-sonnet", inputs={
        "messages": messages, "temperature": temperature, "n": n, "max_tokens": max_tokens
    })
    return [c.get("message", {}).get("content", "") for c in resp.get("choices", [])]

@contextmanager
def _patch_vs_index(mode: Literal["cold", "success", "fail", "empty"]):
    from unittest.mock import patch

    def mock_query(*args, **kwargs):
        if mode == "fail":
            raise RuntimeError("VS offline")
        if mode == "success":
            return [
                {"similarity_score": 0.9, "trigger_event": "A", "causal_chain_json": "{}"},
                {"similarity_score": 0.7, "trigger_event": "B", "causal_chain_json": "{}"},
                {"similarity_score": 0.4, "trigger_event": "C", "causal_chain_json": "{}"}
            ]
        return []

    # Apply patches
    with patch("trading_databricks.agents.nodes.few_shot_retriever.query_vs_index", side_effect=mock_query):
        with patch("trading_databricks.agents.nodes.few_shot_retriever.get_accepted_chain_count", return_value=0 if mode == "cold" else 100):
            yield

@contextmanager
def _prepare_local_graph_environment():
    from unittest.mock import patch

    from trading_databricks.agents.taxonomy import (
        MECHANISM_WHITELIST_SEED,
        TREATMENT_TAXONOMY_SEED,
    )

    def mock_whitelist(*args, **kwargs):
        return {(r["treatment_l1"], r["mechanism"]): r["valid_directions"] for r in MECHANISM_WHITELIST_SEED}

    def mock_taxonomy(*args, **kwargs):
        res = {}
        for r in TREATMENT_TAXONOMY_SEED:
            res.setdefault(r["l1_category"], []).append(r["l2_subcategory"])
        return res

    with patch("trading_databricks.agents.nodes.few_shot_retriever.get_spark_session", get_spark):
        with patch("trading_databricks.agents.nodes.causality_extractor.get_spark_session", get_spark):
            with patch("trading_databricks.agents.taxonomy.load_whitelist", side_effect=mock_whitelist):
                with patch("trading_databricks.agents.taxonomy.load_taxonomy", side_effect=mock_taxonomy):
                    with _patch_vs_index("success"):
                        # Patch embed_fn injection in graph to use graceful mock
                        with patch("trading_databricks.agents.graph._mechanism_validator_node") as mock_validator:
                            yield
