from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from trading_databricks.ml.config import FeatureConfig

_TECHNICAL_SCHEMA = {
    "event_id": pl.Utf8,
    "symbol": pl.Utf8,
    "event_ts": pl.Datetime,
    "computed_at": pl.Datetime,
    "rsi_14": pl.Float64,
    "macd_signal": pl.Float64,
    "bb_pct": pl.Float64,
    "sma_ratio_5_20": pl.Float64,
    "sma_ratio_10_50": pl.Float64,
    "log_return_5": pl.Float64,
    "log_return_20": pl.Float64,
}


def compute_technical(
    events: pl.LazyFrame,
    bars: pl.LazyFrame,
    *,
    config: FeatureConfig | None = None,
) -> pl.LazyFrame:
    """Compute technical indicators at each event's bar_index.

    All indicators use bars strictly ≤ bar_index (PIT-safe by rolling join).
    Returns a LazyFrame. Caller collects.
    """
    if config is None:
        config = FeatureConfig()
    now = datetime.now(UTC)

    # --- Pre-compute rolling indicators on full bar series ---
    bars_with_indicators = bars.sort("bar_index").with_columns(
        # Price deltas for RSI (standard RSI uses price changes, not log returns)
        pl.col("close").diff().alias("_delta"),
        # Log returns (used for log_return_5, log_return_20 below)
        pl.when(pl.col("close") > 0).then(pl.col("close").log()).otherwise(None).diff().alias("_log_ret"),
    ).with_columns(
        # RSI-14 (on price deltas per AFML/standard definition)
        (
            pl.lit(100.0)
            - pl.lit(100.0) / (
                pl.lit(1.0)
                + pl.col("_delta").clip(lower_bound=0).rolling_mean(14)
                / ((-pl.col("_delta").clip(upper_bound=0)).rolling_mean(14) + 1e-12)
            )
        ).alias("rsi_14"),
        # MACD signal (EMA12 - EMA26, then EMA9 of that)
        (
            pl.col("close").ewm_mean(span=12, min_samples=12)
            - pl.col("close").ewm_mean(span=26, min_samples=26)
        )
        .ewm_mean(span=9, min_samples=9)
        .alias("macd_signal"),
        # Bollinger %B = (close - lower) / (upper - lower)
        # lower = SMA - 2*std, upper = SMA + 2*std, bandwidth = 4*std
        (
            (pl.col("close") - pl.col("close").rolling_mean(20))
            / (4 * pl.col("close").rolling_std(20) + 1e-12)
            + 0.5
        ).alias("bb_pct"),
        # SMA ratios
        (
            pl.col("close").rolling_mean(5)
            / (pl.col("close").rolling_mean(20) + 1e-12)
        ).alias("sma_ratio_5_20"),
        (
            pl.col("close").rolling_mean(10)
            / (pl.col("close").rolling_mean(50) + 1e-12)
        ).alias("sma_ratio_10_50"),
        # Log returns over N bars
        pl.when(pl.col("close") > 0).then(pl.col("close").log()).otherwise(None).diff(5).alias("log_return_5"),
        pl.when(pl.col("close") > 0).then(pl.col("close").log()).otherwise(None).diff(20).alias("log_return_20"),
    ).with_columns(
        pl.lit(now).alias("computed_at"),
    )

    feature_cols = [
        "bar_index", "computed_at",
        "rsi_14", "macd_signal", "bb_pct",
        "sma_ratio_5_20", "sma_ratio_10_50",
        "log_return_5", "log_return_20",
    ]

    return (
        events.join(
            bars_with_indicators.select(feature_cols),
            on="bar_index",
            how="left",
        )
        .select(list(_TECHNICAL_SCHEMA.keys()))
    )
