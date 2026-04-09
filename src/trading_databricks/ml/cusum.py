from __future__ import annotations

import logging
import math
import uuid
from datetime import UTC, datetime

import polars as pl

from trading_databricks.ml.config import CUSUMConfig

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = CUSUMConfig()

_EVENTS_COLUMNS = [
    "event_id",
    "symbol",
    "event_ts",
    "bar_index",
    "log_return",
    "cusum_side",
    "cusum_value",
    "threshold_used",
    "ewma_vol",
    "avg_bars_per_day",
    "_ingested_at",
]

_STATS_COLUMNS = [
    "symbol",
    "date",
    "event_count",
    "threshold_used",
    "ewma_vol",
    "avg_bars_per_day",
]


def _empty_events() -> pl.DataFrame:
    return pl.DataFrame(
        {col: [] for col in _EVENTS_COLUMNS},
        schema={
            "event_id": pl.Utf8,
            "symbol": pl.Utf8,
            "event_ts": pl.Datetime,
            "bar_index": pl.Int64,
            "log_return": pl.Float64,
            "cusum_side": pl.Utf8,
            "cusum_value": pl.Float64,
            "threshold_used": pl.Float64,
            "ewma_vol": pl.Float64,
            "avg_bars_per_day": pl.Float64,
            "_ingested_at": pl.Datetime,
        },
    )


def detect_cusum_events(
    bars: pl.DataFrame,
    *,
    config: CUSUMConfig | None = None,
) -> pl.DataFrame:
    """Detect CUSUM events on dollar bar log returns.

    Args:
        bars: Dollar bars with columns: bar_index, symbol, close, bar_end_time.
              Must span ≥ config.lookback_days calendar days.
        config: CUSUM hyperparameters. Defaults to CUSUMConfig().

    Returns:
        pl.DataFrame matching CUSUM_EVENTS_SCHEMA columns. Empty if warm-up guard fires.
    """
    if config is None:
        config = _DEFAULT_CONFIG

    if bars.is_empty():
        return _empty_events()

    bars = bars.sort("bar_index")

    # Warm-up guard: need ≥ lookback_days calendar days of history
    min_ts = bars["bar_end_time"].min()
    max_ts = bars["bar_end_time"].max()
    span_days = (max_ts - min_ts).total_seconds() / 86400  # type: ignore[operator]
    if span_days < config.lookback_days:
        logger.warning(
            "CUSUM warm-up guard: only %.1f days of bars (need %d). Returning empty.",
            span_days,
            config.lookback_days,
        )
        return _empty_events()

    # --- Step 1: Vectorized log returns ---
    bars = bars.with_columns(
        pl.when(pl.col("close") > 0)
        .then(pl.col("close").log())
        .otherwise(None)
        .diff()
        .alias("log_return"),
    )

    # --- Step 2: EWMA volatility (vectorized) ---
    # alpha = 1 - lambda (decay factor for ewm_mean)
    alpha = 1.0 - config.ewma_lambda
    bars = bars.with_columns(
        pl.col("log_return")
        .pow(2)
        .ewm_mean(alpha=alpha, min_periods=1)
        .sqrt()
        .alias("ewma_vol"),
    )

    # --- Step 3: avg_bars_per_day + rolling threshold ---
    # Compute avg bars per day: bars_seen / days_elapsed
    lookback_size = int(75 * config.lookback_days)
    # initial estimate: 2250 bars for 30 days

    # Calculate seconds since min_ts and convert to days, ensuring minimum of 1.0
    days_since_start_expr = (pl.col("bar_end_time") - pl.lit(min_ts)).cast(
        pl.Duration
    ).dt.total_seconds() / 86400.0
    bars = bars.with_columns(
        pl.when(days_since_start_expr < 1.0)
        .then(1.0)
        .otherwise(days_since_start_expr)
        .alias("_days_since_start"),
    )

    # Rolling calculation: for each bar, count bars as min(bar_index, lookback_size)
    bars = bars.with_columns(
        (
            pl.min_horizontal(
                (pl.int_range(pl.len()) + 1).cast(pl.Float64),
                pl.lit(float(lookback_size)),
            )
            / pl.col("_days_since_start")
        ).alias("avg_bars_per_day"),
    )

    bars = bars.with_columns(
        (
            config.threshold_multiplier
            * pl.col("ewma_vol")
            * pl.col("avg_bars_per_day").sqrt()
        ).alias("threshold"),
    )

    # --- Step 4: Sequential CUSUM loop ---
    events: list[dict] = []
    s_pos, s_neg = 0.0, 0.0
    last_event_idx = -config.min_inter_event_bars

    for row in bars.iter_rows(named=True):
        lr = row["log_return"]
        if lr is None:
            continue
        ewma_vol = row["ewma_vol"]
        if ewma_vol is not None and ewma_vol < 1e-10:
            continue  # zero-vol guard: skip bar, don't update accumulators

        threshold = row["threshold"]
        if threshold is None or (isinstance(threshold, float) and math.isnan(threshold)):
            continue

        s_pos = max(0.0, s_pos + lr)
        s_neg = min(0.0, s_neg + lr)

        spacing_ok = (row["bar_index"] - last_event_idx) >= config.min_inter_event_bars

        if s_pos > threshold and spacing_ok:
            events.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "symbol": row["symbol"],
                    "event_ts": row["bar_end_time"],
                    "bar_index": row["bar_index"],
                    "log_return": lr,
                    "cusum_side": "up",
                    "cusum_value": s_pos,
                    "threshold_used": threshold,
                    "ewma_vol": ewma_vol,
                    "avg_bars_per_day": row["avg_bars_per_day"],
                }
            )
            s_pos, s_neg = 0.0, 0.0
            last_event_idx = row["bar_index"]
        elif abs(s_neg) > threshold and spacing_ok:
            events.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "symbol": row["symbol"],
                    "event_ts": row["bar_end_time"],
                    "bar_index": row["bar_index"],
                    "log_return": lr,
                    "cusum_side": "down",
                    "cusum_value": abs(s_neg),
                    "threshold_used": threshold,
                    "ewma_vol": ewma_vol,
                    "avg_bars_per_day": row["avg_bars_per_day"],
                }
            )
            s_pos, s_neg = 0.0, 0.0
            last_event_idx = row["bar_index"]

    if not events:
        return _empty_events()

    now = datetime.now(UTC)
    result = pl.DataFrame(events)
    result = result.with_columns(pl.lit(now).alias("_ingested_at"))
    return result


def compute_cusum_stats(events: pl.DataFrame) -> pl.DataFrame:
    """Aggregate CUSUM events into daily statistics by symbol.

    Args:
        events: Output of detect_cusum_events(). May be empty.

    Returns:
        pl.DataFrame with columns: symbol, date, event_count,
        threshold_used, ewma_vol, avg_bars_per_day.
    """
    _stats_schema = {
        "symbol": pl.Utf8,
        "date": pl.Date,
        "event_count": pl.Int32,
        "threshold_used": pl.Float64,
        "ewma_vol": pl.Float64,
        "avg_bars_per_day": pl.Float64,
    }
    if events.is_empty():
        return pl.DataFrame(schema=_stats_schema)

    return (
        events.with_columns(pl.col("event_ts").dt.date().alias("date"))
        .group_by(["symbol", "date"])
        .agg(
            pl.len().cast(pl.Int32).alias("event_count"),
            pl.col("threshold_used").mean(),
            pl.col("ewma_vol").mean(),
            pl.col("avg_bars_per_day").mean(),
        )
        .sort(["symbol", "date"])
    )
