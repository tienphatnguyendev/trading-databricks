from __future__ import annotations

from datetime import UTC, datetime

import polars as pl


def compute_positions(signals: pl.DataFrame) -> pl.DataFrame:
    """Compute current portfolio position from bet-sized signals.

    Phase 1A rules:
    - BTC-only, single position per symbol.
    - Latest signal wins (most recent event_ts).
    - primary_side=-1 → 'flat' (long-only Phase 1A constraint).
    - No passing signals → 'flat' with weight=0.0.

    Args:
        signals: Output of compute_bet_sizes() with raw_bet_size and
                 discretized_weight columns.

    Returns:
        pl.DataFrame with 1 row per symbol matching POSITIONS_SCHEMA:
        symbol, side, weight, confidence, signal_id, updated_at.
    """
    now = datetime.now(UTC)
    default_symbol = "BTCUSDT"

    _flat_position = pl.DataFrame(
        {
            "symbol": [default_symbol],
            "side": ["flat"],
            "weight": [0.0],
            "confidence": [0.0],
            "signal_id": [""],
            "updated_at": [now],
        },
        schema={
            "symbol": pl.Utf8,
            "side": pl.Utf8,
            "weight": pl.Float64,
            "confidence": pl.Float64,
            "signal_id": pl.Utf8,
            "updated_at": pl.Datetime("us", "UTC"),
        },
    )

    if signals.is_empty():
        return _flat_position

    is_long = pl.col("primary_side") == 1

    return (
        signals.sort("event_ts", descending=True)
        .group_by("symbol", maintain_order=True)
        .first()
        .with_columns(
            pl.when(is_long).then(pl.lit("long")).otherwise(pl.lit("flat")).alias("side"),
            pl.when(is_long)
            .then(pl.col("discretized_weight").cast(pl.Float64))
            .otherwise(pl.lit(0.0))
            .alias("weight"),
            pl.when(is_long)
            .then(pl.col("meta_confidence").cast(pl.Float64))
            .otherwise(pl.lit(0.0))
            .alias("confidence"),
            pl.lit(now).alias("updated_at"),
            pl.col("signal_id").cast(pl.Utf8),
        )
        .select(["symbol", "side", "weight", "confidence", "signal_id", "updated_at"])
    )
