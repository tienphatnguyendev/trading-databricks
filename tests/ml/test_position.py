from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from trading_databricks.ml.inference.position import compute_positions


def _make_signals(
    meta_confidences: list[float],
    primary_sides: list[int],
    discretized_weights: list[float],
    event_ts_offsets_h: list[int] | None = None,
) -> pl.DataFrame:
    n = len(meta_confidences)
    base = datetime(2025, 1, 1)
    offsets = event_ts_offsets_h if event_ts_offsets_h else list(range(n))
    return pl.DataFrame(
        {
            "signal_id": [f"sig-{i}" for i in range(n)],
            "event_id": [f"evt-{i}" for i in range(n)],
            "symbol": ["BTCUSDT"] * n,
            "event_ts": [base + timedelta(hours=h) for h in offsets],
            "primary_side": primary_sides,
            "primary_confidence": [0.65] * n,
            "meta_label": [1] * n,
            "meta_confidence": meta_confidences,
            "raw_bet_size": [0.80] * n,
            "discretized_weight": discretized_weights,
            "primary_model_version": [1] * n,
            "meta_model_version": [1] * n,
        }
    )


def test_flat_position_when_no_signals() -> None:
    """Empty signals DataFrame → position side='flat', weight=0.0."""
    empty_signals = pl.DataFrame(
        schema={
            "signal_id": pl.Utf8,
            "event_id": pl.Utf8,
            "symbol": pl.Utf8,
            "event_ts": pl.Datetime,
            "primary_side": pl.Int32,
            "primary_confidence": pl.Float64,
            "meta_label": pl.Int32,
            "meta_confidence": pl.Float64,
            "raw_bet_size": pl.Float64,
            "discretized_weight": pl.Float64,
            "primary_model_version": pl.Int32,
            "meta_model_version": pl.Int32,
        }
    )
    result = compute_positions(empty_signals)

    assert len(result) == 1
    assert result["symbol"][0] == "BTCUSDT"
    assert result["side"][0] == "flat"
    assert result["weight"][0] == 0.0
    assert result["confidence"][0] == 0.0


def test_long_position_for_primary_side_1() -> None:
    """primary_side=1 → side='long', weight=discretized_weight."""
    signals = _make_signals([0.80], [1], [0.85])
    result = compute_positions(signals)

    assert result["side"][0] == "long"
    assert abs(result["weight"][0] - 0.85) < 1e-9
    assert abs(result["confidence"][0] - 0.80) < 1e-9


def test_long_only_constraint_short_signal_becomes_flat() -> None:
    """Phase 1A: primary_side=-1 → position side='flat', weight=0.0, confidence=0.0."""
    signals = _make_signals([0.85], [-1], [0.85])
    result = compute_positions(signals)

    assert result["side"][0] == "flat"
    assert result["weight"][0] == 0.0
    assert result["confidence"][0] == 0.0


def test_latest_signal_wins() -> None:
    """Multiple signals: latest event_ts determines the position."""
    signals = _make_signals(
        meta_confidences=[0.60, 0.90],
        primary_sides=[1, 1],
        discretized_weights=[0.35, 0.85],
        event_ts_offsets_h=[0, 5],  # second signal is 5h later
    )
    result = compute_positions(signals)

    # Latest signal: meta_confidence=0.90, discretized_weight=0.85
    assert abs(result["confidence"][0] - 0.90) < 1e-9
    assert abs(result["weight"][0] - 0.85) < 1e-9
    assert result["signal_id"][0] == "sig-1"


def test_output_columns_match_positions_schema() -> None:
    """Output has POSITIONS_SCHEMA columns: symbol, side, weight, confidence,
    signal_id, updated_at."""
    signals = _make_signals([0.80], [1], [0.85])
    result = compute_positions(signals)

    expected = {"symbol", "side", "weight", "confidence", "signal_id", "updated_at"}
    assert set(result.columns) == expected
    assert result.dtypes[result.columns.index("weight")] == pl.Float64
    assert result.dtypes[result.columns.index("updated_at")] == pl.Datetime
