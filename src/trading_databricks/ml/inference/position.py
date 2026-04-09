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
        }
    )

    if signals.is_empty():
        return _flat_position

    # Latest signal by event_ts
    latest = signals.sort("event_ts", descending=True).head(1)
    row = latest.row(0, named=True)

    primary_side = int(row["primary_side"])
    is_long = primary_side == 1
    side = "long" if is_long else "flat"

    return pl.DataFrame(
        {
            "symbol": [row["symbol"]],
            "side": [side],
            "weight": [float(row["discretized_weight"]) if is_long else 0.0],
            "confidence": [float(row["meta_confidence"])],
            "signal_id": [str(row["signal_id"])],
            "updated_at": [now],
        }
    )
