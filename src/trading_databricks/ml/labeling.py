from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import polars as pl

from trading_databricks.ml.config import LabelConfig

logger = logging.getLogger(__name__)

_LABELS_SCHEMA = {
    "event_id": pl.Utf8,
    "symbol": pl.Utf8,
    "event_ts": pl.Datetime,
    "label": pl.Int32,
    "barrier_hit": pl.Utf8,
    "return_pct": pl.Float64,
    "upper_level": pl.Float64,
    "lower_level": pl.Float64,
    "vertical_bar": pl.Int64,
    "matured_at": pl.Datetime,
    "avg_uniqueness": pl.Float64,
}


def _empty_labels() -> pl.DataFrame:
    return pl.DataFrame(schema=_LABELS_SCHEMA)


def compute_triple_barrier_labels(
    events: pl.DataFrame,
    bars: pl.DataFrame,
    *,
    config: LabelConfig | None = None,
) -> pl.DataFrame:
    """Label CUSUM events using the triple-barrier method.

    Args:
        events: CUSUM events DataFrame (CUSUM_EVENTS_SCHEMA columns).
        bars: Dollar bars DataFrame with bar_index, close, bar_end_time.
        config: Labeling hyperparameters. Defaults to LabelConfig().

    Returns:
        pl.DataFrame matching TRIPLE_BARRIER_LABELS_SCHEMA. Only matured events included.
    """
    if config is None:
        config = LabelConfig()
    if events.is_empty() or bars.is_empty():
        logger.warning("compute_triple_barrier_labels: empty input, returning empty DF.")
        return _empty_labels()

    bars = bars.sort("bar_index")
    max_bar_index = bars["bar_index"].max()

    # --- Step 1: Join events with bars to get close at bar_index ---
    events_with_close = events.join(
        bars.select("bar_index", "close", "bar_end_time"),
        on="bar_index",
        how="left",
        suffix="_bar",
    ).filter(pl.col("close").is_not_null())

    # --- Step 2: Compute barrier levels (vectorized) ---
    events_with_close = events_with_close.with_columns(
        (
            pl.col("close") * (1.0 + config.upper_barrier_multiplier * pl.col("ewma_vol"))
        ).alias("upper_level"),
        (
            pl.col("close") * (1.0 - config.lower_barrier_multiplier * pl.col("ewma_vol"))
        ).alias("lower_level"),
        (
            pl.col("bar_index")
            + (pl.col("avg_bars_per_day") * config.vertical_barrier_days).cast(pl.Int64)
        ).alias("vertical_bar"),
    )

    # --- Step 3: Forward-scan loop ---
    bars_array = bars.select("bar_index", "close", "bar_end_time").to_dicts()
    bar_by_idx: dict[int, dict] = {b["bar_index"]: b for b in bars_array}

    labeled: list[dict] = []

    for event in events_with_close.iter_rows(named=True):
        vertical_bar = int(event["vertical_bar"])

        # Maturation filter: skip events that haven't matured yet
        if vertical_bar > max_bar_index:
            continue

        upper = float(event["upper_level"])
        lower = float(event["lower_level"])
        
        if upper <= lower:
            logger.warning("Skipping event %s: degenerate barriers (upper=%.6f <= lower=%.6f)",
                            event["event_id"], upper, lower)
            continue
            
        entry_close = float(event["close"])
        
        if entry_close <= 0.0:
            logger.warning("Skipping event %s: entry_close=%.6f <= 0", event["event_id"], entry_close)
            continue
            
        event_bar_index = int(event["bar_index"])
        label: int = 0
        barrier_hit: str = "vertical"
        matured_at: datetime = event["event_ts"]
        return_pct: float = 0.0
        found = False
        exit_bar = vertical_bar

        # Scan future bars from bar_index+1 to vertical_bar inclusive
        for bi in range(event_bar_index + 1, vertical_bar + 1):
            bar = bar_by_idx.get(bi)
            if bar is None:
                continue
            c = float(bar["close"])
            if c >= upper:
                label = 1
                barrier_hit = "upper"
                matured_at = bar["bar_end_time"]
                return_pct = (c - entry_close) / entry_close
                found = True
                exit_bar = bi
                break
            elif c <= lower:
                label = -1
                barrier_hit = "lower"
                matured_at = bar["bar_end_time"]
                return_pct = (c - entry_close) / entry_close
                found = True
                exit_bar = bi
                break

        if not found:  # vertical barrier hit
            last_bar = bar_by_idx.get(vertical_bar)
            if last_bar is not None:
                return_pct = (float(last_bar["close"]) - entry_close) / entry_close
                sigma = float(event["ewma_vol"])
                if sigma > 0 and abs(return_pct) < config.zero_label_threshold * sigma:
                    label = 0
                else:
                    label = 1 if return_pct > 0 else -1
                matured_at = last_bar["bar_end_time"]
            else:
                label = 0
                matured_at = event["event_ts"]
            barrier_hit = "vertical"

        labeled.append(
            {
                "_bar_index": event_bar_index,
            "_exit_bar": exit_bar,   # temp column for uniqueness
                "event_id": event["event_id"],
                "symbol": event["symbol"],
                "event_ts": event["event_ts"],
                "label": label,
                "barrier_hit": barrier_hit,
                "return_pct": return_pct,
                "upper_level": upper,
                "lower_level": lower,
                "vertical_bar": vertical_bar,
                "matured_at": matured_at,
                "avg_uniqueness": 1.0,  # overwritten below
            }
        )

    if not labeled:
        logger.warning("No matured events after labeling. Returning empty DF.")
        return _empty_labels()

    # Concurrency pass
    labeled = _fill_avg_uniqueness(labeled)

    return pl.DataFrame(
        [{k: v for k, v in row.items() if k not in ("_bar_index", "_exit_bar")} for row in labeled],
        schema=_LABELS_SCHEMA,
    )


def _fill_avg_uniqueness(labeled: list[dict]) -> list[dict]:
    """AFML Ch.4: concurrency array → avg_uniqueness per event."""
    if len(labeled) == 0:
        return labeled

    starts = np.array([row["_bar_index"] for row in labeled], dtype=int)
    ends = np.array([row["_exit_bar"] for row in labeled], dtype=int)

    min_bar = int(starts.min())
    max_bar = int(ends.max())
    span = max_bar - min_bar + 1
    concurrency = np.zeros(span, dtype=float)

    for s, e in zip(starts, ends, strict=True):
        concurrency[s - min_bar : e - min_bar + 1] += 1.0

    for i, row in enumerate(labeled):
        s = starts[i] - min_bar
        e = ends[i] - min_bar + 1
        span_slice = concurrency[s:e]
        # Avoid division by zero (shouldn't happen since the event itself adds 1)
        nonzero = span_slice[span_slice > 0]
        row["avg_uniqueness"] = float(np.mean(1.0 / nonzero)) if len(nonzero) > 0 else 1.0

    return labeled
