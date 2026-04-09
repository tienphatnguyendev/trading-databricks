from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from trading_databricks.ml.features.volume import compute_volume


def _make_bars_lf(
    closes: list[float], volumes: list[float] | None = None,
) -> pl.LazyFrame:
    n = len(closes)
    base = datetime(2025, 1, 1)
    vols = volumes if volumes else [1000.0] * n
    return pl.DataFrame(
        {
            "bar_index": list(range(n)),
            "symbol": ["BTCUSDT"] * n,
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": vols,
            "dollar_value": [c * v for c, v in zip(closes, vols, strict=True)],
            "bar_end_time": [base + timedelta(hours=i + 1) for i in range(n)],
        }
    ).lazy()


def _make_events_lf(bar_indices: list[int]) -> pl.LazyFrame:
    base = datetime(2025, 1, 1)
    return pl.DataFrame(
        {
            "event_id": [f"e{i}" for i in bar_indices],
            "symbol": ["BTCUSDT"] * len(bar_indices),
            "event_ts": [base + timedelta(hours=idx + 1) for idx in bar_indices],
            "bar_index": bar_indices,
            "ewma_vol": [0.02] * len(bar_indices),
            "avg_bars_per_day": [24.0] * len(bar_indices),
        }
    ).lazy()


def test_compute_volume_columns() -> None:
    bars = _make_bars_lf([100.0] * 100)
    events = _make_events_lf([50])
    result = compute_volume(events, bars).collect()
    expected = {
        "event_id", "symbol", "event_ts", "computed_at",
        "vwap_dev_20", "relative_volume_20", "obv_slope_20", "dollar_volume_ratio",
    }
    assert set(result.columns) == expected


def test_relative_volume_equals_one_for_constant_volume() -> None:
    """Constant volume → relative_volume_20 = 1.0 (current / rolling mean = 1)."""
    bars = _make_bars_lf([100.0] * 100, volumes=[500.0] * 100)
    events = _make_events_lf([50])
    result = compute_volume(events, bars).collect()
    val = result["relative_volume_20"][0]
    assert val is not None
    assert abs(val - 1.0) < 0.01


def test_vwap_dev_near_zero_for_flat_prices() -> None:
    """Flat prices → VWAP = close → deviation ≈ 0."""
    bars = _make_bars_lf([100.0] * 100)
    events = _make_events_lf([50])
    result = compute_volume(events, bars).collect()
    val = result["vwap_dev_20"][0]
    assert val is not None
    assert abs(val) < 0.01


def test_dollar_volume_ratio_equals_one_for_constant_data() -> None:
    """Constant dollar volume → dollar_volume_ratio = 1.0."""
    bars = _make_bars_lf([100.0] * 100, volumes=[1000.0] * 100)
    events = _make_events_lf([50])
    result = compute_volume(events, bars).collect()
    val = result["dollar_volume_ratio"][0]
    assert val is not None
    assert abs(val - 1.0) < 0.01
