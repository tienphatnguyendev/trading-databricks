from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
from databricks.connect import DatabricksSession

# Set up McKinsey color palette theme for Plotly
_MC_LIGHT_GRAY = "#f5f5f5"
_MC_DARK_GRAY = "#222222"
_MC_BLUE = "#005eb8"
_MC_MEDIUM_GRAY = "#a2aaad"
_MC_YELLOW = "#f3c13a"

pio.templates["mckinsey"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=_MC_LIGHT_GRAY,
        plot_bgcolor=_MC_LIGHT_GRAY,
        font=dict(color=_MC_DARK_GRAY),
        title=dict(font=dict(color=_MC_DARK_GRAY)),
        xaxis=dict(
            gridcolor=_MC_MEDIUM_GRAY,
            zerolinecolor=_MC_MEDIUM_GRAY,
            tickfont=dict(color=_MC_DARK_GRAY),
            title=dict(font=dict(color=_MC_DARK_GRAY))
        ),
        yaxis=dict(
            gridcolor=_MC_MEDIUM_GRAY,
            zerolinecolor=_MC_MEDIUM_GRAY,
            tickfont=dict(color=_MC_DARK_GRAY),
            title=dict(font=dict(color=_MC_DARK_GRAY))
        ),
        colorway=[_MC_BLUE, _MC_YELLOW, _MC_DARK_GRAY, _MC_MEDIUM_GRAY]
    )
)
pio.templates.default = "mckinsey"

logger = logging.getLogger(__name__)


def _git_tags() -> dict[str, str]:
    """Capture git provenance metadata; returns empty dict if git is unavailable."""
    tags: dict[str, str] = {}
    try:
        tags["git.commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        tags["git.branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        tags["git.dirty"] = "true" if dirty else "false"
    except Exception:
        pass
    return tags

def get_spark():
    return DatabricksSession.builder.serverless(True).getOrCreate()

def load_ml_training_data(limit: int | None = None) -> pl.DataFrame:
    spark = get_spark()
    query = "SELECT * FROM `smart-trading-dev`.`01_bronze`.raw_aggtrades_samples ORDER BY timestamp"
    if limit is not None:
        query += f" LIMIT {limit}"
    return pl.from_arrow(spark.sql(query).toArrow())

def load_raw_aggtrades(limit: int | None = None) -> pl.DataFrame:
    from pyspark.sql import functions as F

    spark = get_spark()
    query = "SELECT * FROM `smart-trading-dev`.`01_bronze`.raw_aggtrades_samples ORDER BY timestamp"
    if limit is not None:
        query += f" LIMIT {limit}"
    sdf = (
        spark.sql(query)
        .withColumn("timestamp", F.col("timestamp").cast("timestamp"))
        .withColumn("price", F.col("price").cast("double"))
        .withColumn("quantity", F.col("quantity").cast("double"))
    )
    df = pl.from_arrow(sdf.toArrow())
    if df.is_empty():
        return pl.DataFrame(schema={"timestamp": pl.Datetime("us"), "price": pl.Float64, "quantity": pl.Float64, "symbol": pl.Utf8})
    return df

def _build_dollar_bars_from_bronze(symbol: str, threshold: float = 1_000_000.0, limit: int | None = None) -> pl.DataFrame:
    """Build hourly OHLCV bars in Spark from bronze when gold is not yet populated.

    Uses 1-hour time buckets instead of cumulative-dollar buckets to avoid an
    expensive global window function over 52M+ rows. Returns bars spanning the
    full sample period (672 bars for a 28-day month), which is enough for CUSUM.
    All aggregation runs on Databricks serverless; only the small result is
    transferred locally via Arrow.
    """
    from pyspark.sql import functions as F

    spark = get_spark()
    sdf = (
        spark.sql(
            f"SELECT "
            f"  CAST(timestamp AS TIMESTAMP) AS ts, "
            f"  CAST(price AS DOUBLE) AS price, "
            f"  CAST(quantity AS DOUBLE) AS quantity, "
            f"  symbol "
            f"FROM `smart-trading-dev`.`01_bronze`.raw_aggtrades_samples "
            f"WHERE symbol = '{symbol}'"
        )
        # Bar index = Unix epoch hour — strictly monotonically increasing, unique per hour
        .withColumn("bar_index", F.floor(F.unix_timestamp("ts") / 3600).cast("long"))
        .groupBy("bar_index", "symbol")
        .agg(
            F.min("ts").alias("bar_start_time"),
            F.max("ts").alias("bar_end_time"),
            F.avg("price").alias("open"),
            F.max("price").alias("high"),
            F.min("price").alias("low"),
            F.avg("price").alias("close"),
            F.sum("quantity").alias("volume"),
            F.sum(F.col("price") * F.col("quantity")).alias("dollar_volume"),
            F.count("*").cast("long").alias("tick_count"),
        )
        .orderBy("bar_index")
    )
    if limit:
        sdf = sdf.limit(limit)
    return pl.from_arrow(sdf.toArrow())


def load_raw_dollar_bars(symbol: str, start: datetime | None = None, end: datetime | None = None, catalog: str = "smart-trading-dev", limit: int | None = None) -> pl.DataFrame:
    spark = get_spark()
    query = f"SELECT * FROM `{catalog}`.`03_gold`.dollar_bars WHERE symbol = '{symbol}'"
    if start:
        query += f" AND bar_end_time >= '{start.isoformat()}'"
    if end:
        query += f" AND bar_end_time <= '{end.isoformat()}'"
    query += " ORDER BY bar_index"
    if limit:
        query += f" LIMIT {limit}"
    try:
        return pl.from_arrow(spark.sql(query).toArrow())
    except Exception:
        logger.warning("03_gold.dollar_bars not found in %s — building from 01_bronze", catalog)
        return _build_dollar_bars_from_bronze(symbol, limit=limit)

def load_article_sample(n: int = 10) -> list[dict]:
    spark = get_spark()
    query = f"SELECT * FROM `smart-trading-dev`.`01_bronze`.news_agent_eval_dataset ORDER BY story_id LIMIT {n}"
    records = pl.from_arrow(spark.sql(query).toArrow()).to_dicts()
    return [
        {
            "story_id": r.get("story_id"),
            "headline": (r.get("headline") or "").strip(),
            "body": r.get("body"),
            "published_at": r.get("published_at"),
        }
        for r in records
    ]

@contextmanager
def mlflow_component_run(layer: str, component: str, params: dict | None = None):
    mlflow.set_tracking_uri("databricks")
    try:
        mlflow.enable_system_metrics_logging()
    except Exception:
        logger.warning("System metrics logging unavailable (psutil not installed)")
    mlflow.set_experiment(f"/Shared/trading-databricks-qa/{layer}/{component}")

    run_name = f"{component}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)
        mlflow.set_tags({
            "layer": layer,
            "component": component,
            "python.version": sys.version.split()[0],
            **_git_tags(),
        })
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
    mlflow.log_text(
        fig.to_html(full_html=False, include_plotlyjs="cdn"),
        f"plots/{name}.html",
    )


def log_dataset(
    df: pl.DataFrame,
    name: str,
    source: str,
    context: str = "training",
) -> None:
    """Log dataset lineage to the active MLflow run.

    Converts to pandas at the MLflow boundary (MLflow data API requires pandas).
    Logs row/column counts as metrics for quick inspection in the UI.
    """
    dataset = mlflow.data.from_pandas(df.to_pandas(), source=source, name=name)
    mlflow.log_input(dataset, context=context)
    mlflow.log_metrics({
        f"dataset.{name}.n_rows": df.height,
        f"dataset.{name}.n_cols": len(df.columns),
    })

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

def build_triple_barrier_chart(events: pl.DataFrame, labels: pl.DataFrame, bars: pl.DataFrame, n_samples: int = 5) -> go.Figure:
    """Build a visualization showing price trajectories and triple barrier bounds for a sample of events."""
    # Ensure data is loaded
    events_df = events.collect() if isinstance(events, pl.LazyFrame) else events
    labels_df = labels.collect() if isinstance(labels, pl.LazyFrame) else labels
    bars_df = bars.collect() if isinstance(bars, pl.LazyFrame) else bars

    # Join labels to events to get barrier levels and hit times
    if "event_ts" in labels_df.columns:
        # Align timezones if they differ
        if events_df.schema["event_ts"] != labels_df.schema["event_ts"]:
            # Drop timezone before join (safest cross-timezone approach in Polars)
            e_df = events_df.with_columns(pl.col("event_ts").dt.replace_time_zone(None))
            l_df = labels_df.with_columns(pl.col("event_ts").dt.replace_time_zone(None))
            labeled_events = e_df.join(l_df, on="event_ts", how="inner")
        else:
            labeled_events = events_df.join(labels_df, on="event_ts", how="inner")
    else:
        # Fallback if label index is bar_index or something else
        labeled_events = events_df.join(labels_df, left_on="bar_index", right_on="bar_index", how="inner")

    # Sample a continuous subset
    if labeled_events.height == 0:
        return go.Figure()

    # Take the first n_samples valid labeled events
    sample = labeled_events.head(n_samples).to_pandas()

    # Determine the time window for the chart + 10% margin
    min_ts = sample["event_ts"].min()
    max_ts = sample["matured_at"].max()

    time_span = max_ts - min_ts
    margin = time_span * 0.1 if time_span.total_seconds() > 0 else pd.Timedelta(hours=1)

    start_window = min_ts - margin
    end_window = max_ts + margin

    # Ensure start/end window are timezone-aware if the bars DF is timezone-aware
    if bars_df.schema["bar_end_time"].time_zone is not None:
        if start_window.tzinfo is None:
            start_window = start_window.tz_localize("UTC")
            end_window = end_window.tz_localize("UTC")
    else:
        if start_window.tzinfo is not None:
            start_window = start_window.tz_localize(None)
            end_window = end_window.tz_localize(None)

    # Filter bars to this window
    window_bars = bars_df.filter(
        (pl.col("bar_end_time") >= start_window) &
        (pl.col("bar_end_time") <= end_window)
    ).to_pandas()

    fig = go.Figure()

    # 1. Base trace: Close price
    fig.add_trace(go.Scatter(
        x=window_bars["bar_end_time"],
        y=window_bars["close"],
        mode="lines",
        name="Close Price",
        line=dict(color="#222222") # mc_dark_gray
    ))

    # 2. Entries (Buying points)
    if "close_bar" in sample.columns:
        entry_prices = sample["close_bar"]
    else:
        entry_prices = sample["bar_index"].apply(
            lambda bi: window_bars.loc[window_bars["bar_index"] == bi, "close"].values[0]
            if bi in window_bars["bar_index"].values else np.nan
        )

    fig.add_trace(go.Scatter(
        x=sample["event_ts"],
        y=entry_prices,
        mode="markers",
        name="CUSUM Event (Entry)",
        marker=dict(color="#005eb8", size=10, symbol="triangle-up") # mc_blue
    ))

    # 3. Draw Barriers and Hit Markers for each event
    for i, row in sample.iterrows():
        entry_time = row["event_ts"]
        entry_price = entry_prices.loc[i]
        mature_time = row["matured_at"]
        hit_type = row["barrier_hit"] # usually 1 (upper), -1 (lower), 0 (time)

        # We need the barrier thresholds. Usually config is applied.
        # For visualization, we can infer upper/lower from the return if barrier_hit != 0
        # However, to be exact we rely on the hit price.
        # Since we don't have the exact threshold levels in the return DF standardly,
        # we will plot a horizontal line from entry to mature_time for the Hit price if hit.

        hit_price = window_bars.loc[window_bars["bar_end_time"] == mature_time, "close"]
        if not hit_price.empty:
            hit_price = hit_price.values[0]
        else:
            hit_price = entry_price # fallback

        # Upper/Lower/Time hit marker
        color = "#a2aaad" # Default time barrier
        name = "Time Barrier Hit"
        if hit_type == 1:
            color = "#005eb8" # Upper
            name = "Take Profit Hit"
        elif hit_type == -1:
            color = "#f3c13a" # Lower
            name = "Stop Loss Hit"

        # Draw a dotted line to the hit point
        fig.add_trace(go.Scatter(
            x=[entry_time, mature_time],
            y=[entry_price, hit_price],
            mode="lines",
            line=dict(color=color, width=1, dash="dot"),
            showlegend=False
        ))

        # Draw the hit marker
        fig.add_trace(go.Scatter(
            x=[mature_time],
            y=[hit_price],
            mode="markers",
            name=name,
            marker=dict(color=color, size=12, symbol="x"),
            showlegend=False # We'll just rely on colors, legend gets too cluttered if repeated
        ))

    # Add dummy traces for legend entries for the hit markers
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="#005eb8", size=12, symbol="x"), name="Take Profit Hit"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="#f3c13a", size=12, symbol="x"), name="Stop Loss Hit"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="#a2aaad", size=12, symbol="x"), name="Time Barrier Hit"))

    fig.update_layout(
        title="Triple Barrier Labeling Trajectories",
        xaxis_title="Time",
        yaxis_title="Price"
    )

    return fig

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
                        with patch("trading_databricks.agents.graph._mechanism_validator_node"):
                            yield


# ---------------------------------------------------------------------------
# Deep-Dive Research Notebook helpers (Tasks 1-6)
# These helpers are used by notebooks/qa/mas/e2e_mas_deep_dive_research.py
# ---------------------------------------------------------------------------
import re as _re  # noqa: E402

_HEADLINE_PREFIX_RE = _re.compile(
    r"^\s*Can you summarize the market impact of the following news:\s*",
    flags=_re.IGNORECASE,
)


def _extract_headline(request: str | None) -> str:
    """Strip the mas_training_dataset request prefix to derive a headline.

    Falls back to the full request on no match. Returns empty string
    on None input so downstream state-building is safe.
    """
    if not request:
        return ""
    return _HEADLINE_PREFIX_RE.sub("", request).strip()


def _display(obj: object) -> None:
    """Dual-run display: uses Databricks ``display`` builtin when available,
    falls back to ``print`` under ``uv run python``.
    """
    try:
        _display_fn = display  # type: ignore[name-defined]
    except NameError:
        print(obj)
        return
    _display_fn(obj)


def _get_dbutils() -> object | None:
    """Return a DBUtils instance when available (Databricks workspace),
    otherwise None. Never raises.
    """
    try:
        from pyspark.dbutils import DBUtils  # type: ignore[import]
    except Exception:
        return None
    try:
        spark = get_spark()
        return DBUtils(spark)
    except Exception:
        return None


# Lazy-imported SDK references so unit tests can patch at the module level.
try:
    from databricks.sdk import WorkspaceClient  # noqa: F401
except Exception:  # pragma: no cover
    WorkspaceClient = None  # type: ignore[assignment,misc]

try:
    from databricks.vector_search.client import VectorSearchClient  # noqa: F401
except Exception:  # pragma: no cover
    VectorSearchClient = None  # type: ignore[assignment,misc]


def _endpoint_ready(endpoint_name: str) -> tuple[bool, str]:
    """Return (ok, detail) for a Databricks Model Serving endpoint."""
    try:
        if WorkspaceClient is None:
            return False, "databricks-sdk not installed"
        client = WorkspaceClient()
        ep = client.serving_endpoints.get(endpoint_name)
        ready = getattr(getattr(ep, "state", None), "ready", None)
        ok = "READY" in str(ready).upper()
        return ok, f"state.ready={ready}"
    except Exception as e:
        return False, repr(e)


def _table_has_rows(fqn: str) -> tuple[bool, str]:
    """Return (ok, detail) for a Unity Catalog table; ok iff ≥ 1 row visible."""
    try:
        spark = get_spark()
        count = spark.table(fqn).limit(1).count()
        ok = count > 0
        return ok, f"visible_rows>=1 ({count})"
    except Exception as e:
        return False, repr(e)


def _vs_index_exists(*, endpoint_name: str, index_name: str) -> tuple[bool, str]:
    """Return (ok, detail) for a Databricks Vector Search index."""
    try:
        if VectorSearchClient is None:
            return False, "databricks-vectorsearch not installed"
        vsc = VectorSearchClient()
        idx = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)
        desc = idx.describe()
        return True, f"describe_ok ({type(desc).__name__})"
    except Exception as e:
        return False, repr(e)


def _stratified_sample(
    df: pl.DataFrame,
    *,
    n: int,
    by: str,
    seed: int,
) -> pl.DataFrame:
    """Seeded stratified sample of ``df`` by column ``by``, returning up to ``n`` rows.

    Allocates proportionally to each bucket's share, with a floor of 1 per
    bucket. If ``n >= df.height``, returns the full frame untouched (still
    seeded for deterministic downstream operations).
    """
    if n >= df.height:
        return df

    bucket_counts = df.group_by(by).len().sort(by)
    total = df.height
    parts: list[pl.DataFrame] = []
    remaining = n
    buckets = bucket_counts.to_dicts()
    for i, row in enumerate(buckets):
        bucket_value = row[by]
        bucket_size = int(row["len"])
        if i == len(buckets) - 1:
            take = remaining
        else:
            take = max(1, round(n * bucket_size / total))
            take = min(take, bucket_size, remaining - (len(buckets) - i - 1))
            take = max(take, 0)
        if take <= 0:
            continue
        slice_df = df.filter(pl.col(by) == bucket_value).sample(
            n=min(take, bucket_size), seed=seed, shuffle=True,
        )
        parts.append(slice_df)
        remaining -= slice_df.height
    return pl.concat(parts) if parts else df.head(0)


def build_instrumented_mas_graph(config: object) -> object:
    """Compile the 5-node MAS graph with every node wrapped in mlflow.trace.

    Topology mirrors trading_databricks.agents.graph.build_mas_graph exactly
    but guarantees per-node CHAIN spans for §8 real-latency analysis.
    Production node code is unchanged.
    """
    from langgraph.graph import StateGraph
    from mlflow.entities import SpanType

    from trading_databricks.agents.graph import (
        _mechanism_validator_node,
        route_article,
    )
    from trading_databricks.agents.nodes import (
        causality_extractor,
        context_retriever,
        few_shot_retriever,
        uncertainty_router,
    )
    from trading_databricks.agents.state import MASState

    def _traced(name: str, fn: object) -> object:
        @mlflow.trace(name=name, span_type=SpanType.CHAIN)
        def _wrapper(state: object) -> object:
            return fn(state)  # type: ignore[operator]
        _wrapper.__name__ = f"traced_{name}"
        return _wrapper

    graph = StateGraph(MASState)
    graph.add_node("context_retriever",   _traced("context_retriever",   lambda s: context_retriever(s, config=config)))
    graph.add_node("few_shot_retriever",  _traced("few_shot_retriever",  lambda s: few_shot_retriever(s, config=config)))
    graph.add_node("uncertainty_router",  _traced("uncertainty_router",  lambda s: uncertainty_router(s, config=config)))
    graph.add_node("causality_extractor", _traced("causality_extractor", lambda s: causality_extractor(s, config=config)))
    graph.add_node("mechanism_validator", _traced("mechanism_validator", lambda s: _mechanism_validator_node(s, config)))

    graph.set_entry_point("context_retriever")
    graph.add_edge("context_retriever",   "few_shot_retriever")
    graph.add_edge("few_shot_retriever",  "uncertainty_router")
    graph.add_conditional_edges(
        "uncertainty_router",
        route_article,
        {"causality_extractor": "causality_extractor"},
    )
    graph.add_edge("causality_extractor", "mechanism_validator")
    graph.set_finish_point("mechanism_validator")
    return graph.compile()


@contextmanager
def wire_graph_environment(config: object, *, offline: bool):  # type: ignore[return]
    """Patch runtime hooks so the MAS graph can execute from a notebook.

    Live mode (offline=False):
      - Replaces ``get_spark_session`` placeholders with a Databricks Connect session
      - Replaces ``query_vs_index`` placeholder with a real VectorSearchClient call

    Offline mode (offline=True):
      - All of the above with deterministic mocks
      - Additionally patches mlflow.deployments get_deploy_client.predict to return
        taxonomy-compliant JSON (LLM) or a fixed unit vector (embeddings)

    The ``_mechanism_validator_node`` is NEVER patched in any mode — the real
    deterministic validation logic runs on every row.
    """
    import json as _json
    from unittest.mock import patch as _patch
    import importlib
    
    _ce_mod = importlib.import_module("trading_databricks.agents.nodes.causality_extractor")
    _fsr_mod = importlib.import_module("trading_databricks.agents.nodes.few_shot_retriever")

    patches: list = []

    patches.append(_patch.object(_fsr_mod, "get_spark_session", get_spark))
    patches.append(_patch.object(_ce_mod, "get_spark_session", get_spark))

    if offline:
        def _mock_vs_query(**kwargs: object) -> list[dict]:
            return [
                {"story_id": "mock-1", "treatment_class": "Regulation.etf_ruling",
                 "mechanism": "institutional_access", "expected_direction": "bullish",
                 "confidence": 0.9, "trigger_event": "mock trigger",
                 "causal_chain_json": "{}", "similarity_score": 0.9},
                {"story_id": "mock-2", "treatment_class": "Macro.rate_decision",
                 "mechanism": "liquidity_contraction", "expected_direction": "bearish",
                 "confidence": 0.8, "trigger_event": "mock trigger 2",
                 "causal_chain_json": "{}", "similarity_score": 0.7},
            ]
        patches.append(_patch.object(_fsr_mod, "query_vs_index", _mock_vs_query))

        llm_endpoint = config.llm_endpoint  # type: ignore[attr-defined]
        embed_endpoint = config.validator.embedding_model  # type: ignore[attr-defined]

        _mock_json = _json.dumps({
            "treatment_class": "Macro.rate_decision",
            "mechanism": "liquidity_contraction",
            "expected_direction": "bearish",
            "confidence": 0.8,
            "trigger_event": "mock trigger event",
            "evidence_spans": ["mock evidence"],
            "reasoning": "deterministic offline mock",
        })

        class _MockDeployClient:
            def predict(self, *, endpoint: str, inputs: dict) -> dict:
                if endpoint == embed_endpoint:
                    return {"data": [{"embedding": [1.0, 0.0, 0.0]}]}
                if endpoint == llm_endpoint:
                    n = int(inputs.get("n", 1))
                    return {
                        "choices": [
                            {"message": {"content": _mock_json}} for _ in range(n)
                        ]
                    }
                raise RuntimeError(f"Unexpected offline endpoint: {endpoint}")

        patches.append(
            _patch("mlflow.deployments.get_deploy_client", return_value=_MockDeployClient())
        )
    else:
        import json as _json2

        def _real_vs_query(**kwargs: object) -> list[dict]:
            vsc = VectorSearchClient()
            idx = vsc.get_index(
                endpoint_name=config.vs_endpoint_name,  # type: ignore[attr-defined]
                index_name=kwargs["index_name"],  # type: ignore[index]
            )
            raw = idx.similarity_search(
                query_text=kwargs["query_text"],  # type: ignore[index]
                columns=kwargs["columns"],  # type: ignore[index]
                num_results=kwargs["num_results"],  # type: ignore[index]
                filters=_json2.loads(kwargs["filters_json"]),  # type: ignore[index]
            )
            rows = raw.get("result", {}).get("data_array", []) or []
            cols = kwargs["columns"]  # type: ignore[index]
            out: list[dict] = []
            for row in rows:
                rec = {col: row[i] for i, col in enumerate(cols)}
                if len(row) > len(cols):
                    rec["similarity_score"] = row[len(cols)]
                out.append(rec)
            return out

        patches.append(_patch.object(_fsr_mod, "query_vs_index", _real_vs_query))

    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in patches:
            p.stop()
