from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from trading_databricks.ml.features.volatility import compute_volatility


def _make_bars_lf(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    symbol: str = "BTCUSDT",
) -> pl.LazyFrame:
    n = len(closes)
    base = datetime(2025, 1, 1)
    return pl.DataFrame(
        {
            "bar_index": list(range(n)),
            "symbol": [symbol] * n,
            "open": closes,
            "high": highs if highs else [c * 1.002 for c in closes],
            "low": lows if lows else [c * 0.998 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
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


def test_compute_volatility_columns() -> None:
    bars = _make_bars_lf([100.0] * 100)
    events = _make_events_lf([50])
    result = compute_volatility(events, bars).collect()
    expected = {
        "event_id", "symbol", "event_ts", "computed_at",
        "realized_vol_20", "realized_vol_50",
        "parkinson_vol_20", "garman_klass_20", "vol_ratio_20_50",
    }
    assert set(result.columns) == expected


def test_realized_vol_positive_for_volatile_prices() -> None:
    """Volatile prices → realized_vol_20 > 0."""
    import numpy as np
    prices = list(100.0 * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.02, 100)))
    bars = _make_bars_lf(prices)
    events = _make_events_lf([80])
    result = compute_volatility(events, bars).collect()
    val = result["realized_vol_20"][0]
    assert val is not None and val > 0.0


def test_parkinson_vol_positive() -> None:
    """Parkinson vol > 0 when high != low."""
    prices = [100.0 + i * 0.1 for i in range(100)]
    bars = _make_bars_lf(prices)  # default high=close*1.002, low=close*0.998
    events = _make_events_lf([80])
    result = compute_volatility(events, bars).collect()
    val = result["parkinson_vol_20"][0]
    assert val is not None and val > 0.0


def test_vol_ratio_gt_one_when_recent_more_volatile() -> None:
    """If vol_20 > vol_50, vol_ratio_20_50 > 1."""
    import numpy as np
    # First 50 bars: flat. Next 50 bars: very volatile.
    flat = [100.0] * 50
    volatile = list(100.0 * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.05, 50)))
    prices = flat + volatile
    bars = _make_bars_lf(prices)
    events = _make_events_lf([99])  # at end where recent is volatile
    result = compute_volatility(events, bars).collect()
    ratio = result["vol_ratio_20_50"][0]
    if ratio is not None:
        assert ratio > 1.0
