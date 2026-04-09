from __future__ import annotations

from datetime import UTC, datetime

import polars as pl


def assemble_training_dataset(
    events: pl.LazyFrame,
    labels: pl.LazyFrame,
    feature_frames: list[pl.LazyFrame],
) -> pl.DataFrame:
    """Join events, labels, and all feature frames into a training dataset.

    Steps:
    1. Join all feature frames onto events via (event_id, symbol).
    2. Join labels onto the result via (event_id, symbol).
    3. Sort by event_ts, assign cpcv_group (1-6 chronological quantiles).
    4. Stamp dataset_version as ISO timestamp string.

    Returns an eager pl.DataFrame (collected once at the end).
    """
    result = events
    for ff in feature_frames:
        # Drop duplicate key columns from ff to avoid suffix clashes
        ff_clean = ff.select(
            [c for c in ff.columns if c not in ("symbol", "event_ts")]
            if "event_id" in ff.columns
            else ff.columns
        )
        # Join on event_id, keeping symbol/event_ts from events
        result = result.join(ff_clean, on="event_id", how="left")

    # Join labels
    label_cols = [
        c for c in labels.columns if c not in ("symbol", "event_ts", "bar_index")
    ]
    label_select_cols = (
        label_cols if "event_id" in label_cols else ["event_id"] + label_cols
    )
    result = result.join(
        labels.select(label_select_cols),
        on="event_id",
        how="left",
    )

    # Collect + sort + assign cpcv_group
    df = result.collect().sort("event_ts")
    n = len(df)

    if n > 0:
        df = df.with_columns(
            ((pl.arange(0, n, eager=True) * 6) // n + 1)
            .cast(pl.Int32)
            .alias("cpcv_group")
        )
    else:
        df = df.with_columns(pl.lit(1).cast(pl.Int32).alias("cpcv_group"))

    # Stamp dataset_version
    df = df.with_columns(pl.lit(datetime.now(UTC).isoformat()).alias("dataset_version"))

    return df
