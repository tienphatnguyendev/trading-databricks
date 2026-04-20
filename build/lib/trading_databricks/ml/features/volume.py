from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl

from trading_databricks.ml.config import FeatureConfig

_VOLUME_SCHEMA = {
    "event_id": pl.Utf8,
    "symbol": pl.Utf8,
    "event_ts": pl.Datetime,
    "computed_at": pl.Datetime,
    "vwap_dev_20": pl.Float64,
    "relative_volume_20": pl.Float64,
    "obv_slope_20": pl.Float64,
    "dollar_volume_ratio": pl.Float64,
}


def _obv_slope_fn(vals: pl.Series) -> float:
    """Linear regression slope of OBV series over window."""
    arr = vals.to_numpy().astype(float)
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean, y_mean = x.mean(), arr.mean()
    num = float(((x - x_mean) * (arr - y_mean)).sum())
    den = float(((x - x_mean) ** 2).sum())
    return num / den if abs(den) > 1e-12 else 0.0


def compute_volume(
    events: pl.LazyFrame,
    bars: pl.LazyFrame,
    *,
    config: FeatureConfig | None = None,
) -> pl.LazyFrame:
    """Compute volume-based features at each event's bar_index."""
    if config is None:
        config = FeatureConfig()
    now = datetime.now(UTC)

    bars_sorted = (
        bars.sort("bar_index")
        .with_columns(
            (pl.col("close") * pl.col("volume")).alias("_dollar_vol"),
        )
        .with_columns(
            # VWAP rolling 20 bars
            (
                pl.col("_dollar_vol").rolling_sum(20)
                / (pl.col("volume").rolling_sum(20) + 1e-12)
            ).alias("_vwap_20"),
            # Rolling mean volume
            pl.col("volume").rolling_mean(20).alias("_vol_mean_20"),
            # Rolling mean dollar volume
            pl.col("_dollar_vol").rolling_mean(20).alias("_dv_mean_20"),
            # OBV (signed volume)
            pl.when(pl.col("close").diff() > 0)
            .then(pl.col("volume"))
            .when(pl.col("close").diff() < 0)
            .then(-pl.col("volume"))
            .otherwise(0.0)
            .cum_sum()
            .alias("_obv"),
        )
        .with_columns(
            # VWAP deviation
            ((pl.col("close") - pl.col("_vwap_20")) / (pl.col("_vwap_20") + 1e-12)).alias(
                "vwap_dev_20"
            ),
            # Relative volume
            (pl.col("volume") / (pl.col("_vol_mean_20") + 1e-12)).alias(
                "relative_volume_20"
            ),
            # Dollar volume ratio
            (pl.col("_dollar_vol") / (pl.col("_dv_mean_20") + 1e-12)).alias(
                "dollar_volume_ratio"
            ),
            # OBV slope via rolling_map
            pl.col("_obv")
            .rolling_map(_obv_slope_fn, window_size=20)
            .alias("obv_slope_20"),
            pl.lit(now).alias("computed_at"),
        )
    )

    feature_cols = [
        "bar_index",
        "computed_at",
        "vwap_dev_20",
        "relative_volume_20",
        "obv_slope_20",
        "dollar_volume_ratio",
    ]

    return events.join(
        bars_sorted.select(feature_cols), on="bar_index", how="left"
    ).select(list(_VOLUME_SCHEMA.keys()))
