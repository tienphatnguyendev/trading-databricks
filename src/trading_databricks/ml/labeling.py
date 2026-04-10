from __future__ import annotations

import logging

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

    # --- Step 3: Forward-scan logic (Vectorized via join_where) ---
    # Filter valid events to process
    events_valid = events_with_close.filter(
        (pl.col("vertical_bar") <= max_bar_index)
        & (pl.col("upper_level") > pl.col("lower_level"))
        & (pl.col("close") > 0.0)
    )

    if events_valid.is_empty():
        logger.warning("No matured events after filtering. Returning empty DF.")
        return _empty_labels()

    # Rename bar columns for the non-equi join to avoid conflicts
    bars_for_join = bars.select(
        pl.col("bar_index").alias("future_bar_index"),
        pl.col("close").alias("future_close"),
        pl.col("bar_end_time").alias("future_time"),
    ).lazy()

    events_lazy = events_valid.lazy()

    # Find all future bars where the barrier is hit
    hits = events_lazy.join_where(
        bars_for_join,
        pl.col("future_bar_index") > pl.col("bar_index"),
        pl.col("future_bar_index") <= pl.col("vertical_bar"),
        (pl.col("future_close") >= pl.col("upper_level"))
        | (pl.col("future_close") <= pl.col("lower_level")),
    )

    # We only care about the FIRST hit
    first_hits = hits.group_by("event_id").agg(
        pl.col("future_bar_index").min().alias("hit_bar_index")
    )

    # Join the specific hit bar info back
    hits_data = first_hits.join(
        bars_for_join, left_on="hit_bar_index", right_on="future_bar_index", how="left"
    )

    # Re-join hits to the valid events
    final_events = events_lazy.join(hits_data, on="event_id", how="left")

    # Also join the vertical barrier bar data (used if no upper/lower hit occurs)
    vertical_bars = bars_for_join.rename(
        {
            "future_bar_index": "vert_bar_index",
            "future_close": "vert_close",
            "future_time": "vert_time",
        }
    )

    final_events = final_events.join(
        vertical_bars, left_on="vertical_bar", right_on="vert_bar_index", how="left"
    )

    # Determine hit_type (upper, lower, vertical)
    # Check upper hit BEFORE lower hit on the same bar to match original precedence
    final_events = final_events.with_columns(
        hit_type=pl.when(
            pl.col("hit_bar_index").is_not_null()
            & (pl.col("future_close") >= pl.col("upper_level"))
        )
        .then(pl.lit("upper"))
        .when(
            pl.col("hit_bar_index").is_not_null()
            & (pl.col("future_close") <= pl.col("lower_level"))
        )
        .then(pl.lit("lower"))
        .otherwise(pl.lit("vertical"))
    )

    # Compute outcomes (matured_at, _exit_bar, return_pct)
    final_events = final_events.with_columns(
        matured_at=pl.when(pl.col("hit_type") == "vertical")
        .then(
            pl.when(pl.col("vert_time").is_not_null())
            .then(pl.col("vert_time"))
            .otherwise(pl.col("event_ts"))
        )
        .otherwise(pl.col("future_time")),
        _exit_bar=pl.when(pl.col("hit_type") == "vertical")
        .then(pl.col("vertical_bar"))
        .otherwise(pl.col("hit_bar_index")),
        return_pct=pl.when(pl.col("hit_type") == "vertical")
        .then(
            pl.when(pl.col("vert_close").is_not_null())
            .then((pl.col("vert_close") - pl.col("close")) / pl.col("close"))
            .otherwise(pl.lit(0.0))
        )
        .otherwise((pl.col("future_close") - pl.col("close")) / pl.col("close")),
    )

    # Assign labels based on returns and hit types
    final_events = final_events.with_columns(
        label=pl.when(pl.col("hit_type") == "upper")
        .then(pl.lit(1))
        .when(pl.col("hit_type") == "lower")
        .then(pl.lit(-1))
        .otherwise(
            # Vertical barrier logic
            pl.when(
                (pl.col("ewma_vol") > 0)
                & (
                    pl.col("return_pct").abs()
                    < config.zero_label_threshold * pl.col("ewma_vol")
                )
            )
            .then(pl.lit(0))
            .when(pl.col("return_pct") > 0)
            .then(pl.lit(1))
            .when(pl.col("return_pct") < 0)
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))
        )
    )

    # Format dataframe to match expected schema + internal columns for uniqueness
    res_lazy = final_events.select(
        pl.col("bar_index").alias("_bar_index"),
        pl.col("_exit_bar"),
        pl.col("event_id"),
        pl.col("symbol"),
        pl.col("event_ts"),
        pl.col("label").cast(pl.Int32),
        pl.col("hit_type").alias("barrier_hit"),
        pl.col("return_pct"),
        pl.col("upper_level"),
        pl.col("lower_level"),
        pl.col("vertical_bar"),
        pl.col("matured_at"),
    )

    # Collect the results
    res_df = res_lazy.collect()

    # Use the existing concurrency logic
    dicts = res_df.to_dicts()
    labeled = _fill_avg_uniqueness(dicts)

    return pl.DataFrame(
        [
            {k: v for k, v in row.items() if k not in ("_bar_index", "_exit_bar")}
            for row in labeled
        ],
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
