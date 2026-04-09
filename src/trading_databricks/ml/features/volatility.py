from __future__ import annotations

import math
from datetime import UTC, datetime

import polars as pl

from trading_databricks.ml.config import FeatureConfig

_VOLATILITY_SCHEMA = {
    "event_id": pl.Utf8,
    "symbol": pl.Utf8,
    "event_ts": pl.Datetime,
    "computed_at": pl.Datetime,
    "realized_vol_20": pl.Float64,
    "realized_vol_50": pl.Float64,
    "parkinson_vol_20": pl.Float64,
    "garman_klass_20": pl.Float64,
    "vol_ratio_20_50": pl.Float64,
}


def compute_volatility(
    events: pl.LazyFrame,
    bars: pl.LazyFrame,
    *,
    config: FeatureConfig | None = None,
) -> pl.LazyFrame:
    """Compute volatility features at each event's bar_index.

    annualization_factor = sqrt(avg_bars_per_day * 365)
    Uses a representative avg_bars_per_day=24 (4h bars) for bar-level rolling.
    Post-join: replace with event's own avg_bars_per_day.
    """
    if config is None:
        config = FeatureConfig()
    now = datetime.now(UTC)
    ln2 = math.log(2)

    bars_with_vol = (
        bars.sort("bar_index")
        .with_columns(
            pl.col("close").log().diff().alias("_log_ret"),
            (pl.col("high") / pl.col("low")).log().pow(2).alias("_log_hl_sq"),
            (pl.col("close") / pl.col("open")).log().pow(2).alias("_log_co_sq"),
            (pl.col("high") / pl.col("low")).log().pow(2).mul(0.5).alias("_log_hl_half"),
        )
        .with_columns(
            # UN-annualized rolling volatility
            # (annualized post-join with event avg_bars_per_day)
            pl.col("_log_ret").rolling_std(20).alias("realized_vol_20"),
            pl.col("_log_ret").rolling_std(50).alias("realized_vol_50"),
            # Parkinson: sqrt(rolling_mean(log(H/L)^2) / (4*ln2))
            (pl.col("_log_hl_sq").rolling_mean(20) / (4 * ln2))
            .sqrt()
            .alias("parkinson_vol_20"),
            # Garman-Klass: sqrt(rolling_mean(0.5*ln(H/L)^2 - (2*ln2-1)*ln(C/O)^2))
            (pl.col("_log_hl_half") - pl.col("_log_co_sq").mul(2 * ln2 - 1))
            .rolling_mean(20)
            .clip(lower_bound=0.0)
            .sqrt()
            .alias("garman_klass_20"),
        )
        .with_columns(
            (pl.col("realized_vol_20") / (pl.col("realized_vol_50") + 1e-12)).alias(
                "vol_ratio_20_50"
            ),
            pl.lit(now).alias("computed_at"),
        )
    )

    feature_cols = [
        "bar_index",
        "computed_at",
        "realized_vol_20",
        "realized_vol_50",
        "parkinson_vol_20",
        "garman_klass_20",
        "vol_ratio_20_50",
    ]

    # Join to events, then annualize using event-specific avg_bars_per_day
    ann_factor = (pl.col("avg_bars_per_day") * 365.0).sqrt()
    return (
        events.join(bars_with_vol.select(feature_cols), on="bar_index", how="left")
        .with_columns(
            (pl.col("realized_vol_20") * ann_factor).alias("realized_vol_20"),
            (pl.col("realized_vol_50") * ann_factor).alias("realized_vol_50"),
            (pl.col("parkinson_vol_20") * ann_factor).alias("parkinson_vol_20"),
            (pl.col("garman_klass_20") * ann_factor).alias("garman_klass_20"),
            # vol_ratio unchanged — annualization cancels in ratio
        )
        .select(list(_VOLATILITY_SCHEMA.keys()))
    )
