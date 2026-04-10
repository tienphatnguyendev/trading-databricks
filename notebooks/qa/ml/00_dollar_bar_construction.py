"""
ml/00_dollar_bar_construction.py — QA notebook for dollar bar construction.

Run: uv run python notebooks/qa/ml/00_dollar_bar_construction.py
"""
import polars as pl

from notebooks.qa.utils import (
    build_candlestick,
    load_raw_aggtrades,
    log_diag,
    log_plotly,
    mlflow_component_run,
    summarize,
)
from trading_databricks.common.config import DollarBarConfig
from trading_databricks.ml.validation import validate_bars

config = DollarBarConfig(threshold=1_000_000.0)

with mlflow_component_run(
    layer="ml", component="dollar_bar_construction",
    params={"dollar_bar.threshold": config.threshold},
):
    # Section 1 — Load input
    ticks = load_raw_aggtrades(limit=500_000)
    log_diag("input_ticks_summary", summarize(ticks, ["price", "quantity", "timestamp"]))

    # Section 2 — Build dollar bars inline (TRD-5 not yet merged; replicated here)
    ticks_with_dv = ticks.with_columns(
        (pl.col("price") * pl.col("quantity")).alias("dollar_value")
    )
    ticks_with_idx = ticks_with_dv.with_columns(
        (pl.col("dollar_value").cum_sum() / config.threshold)
        .floor()
        .cast(pl.Int64)
        .alias("bar_index")
    )
    bars = (
        ticks_with_idx
        .group_by("bar_index")
        .agg([
            pl.col("timestamp").first().alias("bar_start_time"),
            pl.col("timestamp").last().alias("bar_end_time"),
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("quantity").sum().alias("volume"),
            pl.col("dollar_value").sum().alias("dollar_volume"),
            pl.col("symbol").first().alias("symbol"),
            pl.len().alias("tick_count"),
        ])
        .sort("bar_index")
    )

    # Section 3 — Diagnostic tables
    log_diag("bars_summary", summarize(bars, ["close", "volume", "dollar_volume"]))

    # Section 4 — Visual inspection
    log_plotly("bars_candlestick", build_candlestick(bars))

    # Section 5 — Sanity assertions
    assert bars.height > 0, "no bars generated"
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()
    assert bars["bar_index"].diff().drop_nulls().min() > 0
    validate_bars(bars)
