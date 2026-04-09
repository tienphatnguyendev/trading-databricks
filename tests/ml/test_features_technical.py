from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from trading_databricks.ml.features.technical import compute_technical


def _make_bars_lf(closes: list[float], symbol: str = "BTCUSDT") -> pl.LazyFrame:
    n = len(closes)
    base = datetime(2025, 1, 1)
    return pl.DataFrame(
        {
            "bar_index": list(range(n)),
            "symbol": [symbol] * n,
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
            "bar_end_time": [base + timedelta(hours=i + 1) for i in range(n)],
        }
    ).lazy()


def _make_events_lf(bar_indices: list[int], symbol: str = "BTCUSDT") -> pl.LazyFrame:
    base = datetime(2025, 1, 1)
    return pl.DataFrame(
        {
            "event_id": [f"evt-{i}" for i in bar_indices],
            "symbol": [symbol] * len(bar_indices),
            "event_ts": [base + timedelta(hours=idx + 1) for idx in bar_indices],
            "bar_index": bar_indices,
            "ewma_vol": [0.02] * len(bar_indices),
            "avg_bars_per_day": [24.0] * len(bar_indices),
        }
    ).lazy()


def test_compute_technical_returns_lazyframe() -> None:
    """compute_technical returns a LazyFrame (not collected)."""
    bars = _make_bars_lf([100.0] * 100)
    events = _make_events_lf([50])
    result = compute_technical(events, bars)
    assert isinstance(result, pl.LazyFrame)


def test_compute_technical_columns_match_schema() -> None:
    """Output has all FEATURES_TECHNICAL_SCHEMA columns."""
    bars = _make_bars_lf([100.0] * 100)
    events = _make_events_lf([50])
    result = compute_technical(events, bars).collect()

    expected = {
        "event_id", "symbol", "event_ts", "computed_at",
        "rsi_14", "macd_signal", "bb_pct",
        "sma_ratio_5_20", "sma_ratio_10_50",
        "log_return_5", "log_return_20",
    }
    assert set(result.columns) == expected


def test_rsi_is_between_0_and_100() -> None:
    """RSI must always be in [0, 100] for valid price data."""
    import numpy as np
    prices = list(100.0 * np.cumprod(1 + np.random.default_rng(42).normal(0, 0.01, 200)))
    bars = _make_bars_lf(prices)
    events = _make_events_lf(list(range(20, 200, 10)))
    result = compute_technical(events, bars).collect()

    rsi_vals = result["rsi_14"].drop_nulls().to_list()
    assert all(0.0 <= v <= 100.0 for v in rsi_vals), f"RSI out of range: {rsi_vals}"


def test_pit_safety_no_future_bars_used() -> None:
    """Features at bar_index=50 must not use any bar > 50."""
    # Set bar 51 close to a very distinctive value. Features should not reflect it.
    closes = [100.0] * 51 + [999999.0] * 49
    bars = _make_bars_lf(closes)
    events = _make_events_lf([50])
    result = compute_technical(events, bars).collect()

    # If PIT-safe, features should be based on bars [0..50] all at 100.0
    # sma_ratio_5_20 should be ~1.0 (all same price)
    ratio = result["sma_ratio_5_20"][0]
    if ratio is not None:
        assert abs(ratio - 1.0) < 0.01, f"sma_ratio_5_20={ratio} suggests future leak"


def test_empty_events_returns_empty_lazyframe() -> None:
    """No events → empty LazyFrame with correct columns."""
    bars = _make_bars_lf([100.0] * 50)
    events = pl.DataFrame(
        schema={
            "event_id": pl.Utf8,
            "symbol": pl.Utf8,
            "event_ts": pl.Datetime,
            "bar_index": pl.Int64,
            "ewma_vol": pl.Float64,
            "avg_bars_per_day": pl.Float64,
        }
    ).lazy()
    result = compute_technical(events, bars).collect()
    assert result.is_empty()
    assert "rsi_14" in result.columns
