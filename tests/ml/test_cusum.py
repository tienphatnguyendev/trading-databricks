from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from trading_databricks.ml.config import CUSUMConfig
from trading_databricks.ml.cusum import compute_cusum_stats, detect_cusum_events


def _make_bars(
    closes: list[float],
    hours_per_bar: float = 4.5,
    symbol: str = "BTCUSDT",
) -> pl.DataFrame:
    """Synthetic bars. 200 bars × 4.5h = 37.5 days, enough to pass warm-up."""
    n = len(closes)
    base_dt = datetime(2025, 1, 1)
    return pl.DataFrame(
        {
            "bar_index": list(range(n)),
            "symbol": [symbol] * n,
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
            "bar_start_time": [
                base_dt + timedelta(hours=i * hours_per_bar) for i in range(n)
            ],
            "bar_end_time": [
                base_dt + timedelta(hours=(i + 1) * hours_per_bar) for i in range(n)
            ],
        }
    )


def test_warmup_guard_returns_empty_df() -> None:
    """Bars spanning only 10 days → warm-up guard → empty DataFrame."""
    # 50 bars × 4.5h = 225h ≈ 9.4 days < 30 days
    closes = [100.0] * 50
    bars = _make_bars(closes, hours_per_bar=4.5)
    result = detect_cusum_events(bars, config=CUSUMConfig())
    assert result.is_empty()
    assert "event_id" in result.columns
    assert "cusum_side" in result.columns


def test_step_function_produces_single_up_event() -> None:
    """200 bars: flat at 100, then jump to 110 at bar 100 → exactly 1 up event."""
    closes = [100.0] * 100 + [110.0] * 100
    bars = _make_bars(closes, hours_per_bar=4.5)  # spans 37.5 days
    result = detect_cusum_events(bars, config=CUSUMConfig())

    assert len(result) == 1
    assert result["cusum_side"][0] == "up"
    assert result["bar_index"][0] == 100
    assert result["event_id"][0] != ""
    assert result["ewma_vol"][0] > 0
    assert result["threshold_used"][0] > 0
    assert result["avg_bars_per_day"][0] > 0


def test_flat_market_produces_no_events() -> None:
    """300 bars all at the same price → zero events."""
    closes = [50000.0] * 300
    bars = _make_bars(closes, hours_per_bar=3.0)  # spans 37.5 days
    result = detect_cusum_events(bars, config=CUSUMConfig())
    assert result.is_empty()


def test_flash_crash_events_spaced_by_min_inter_event_bars() -> None:
    """Rapid -2% moves: events must be ≥ 5 bars apart."""
    import numpy as np

    # 200 flat bars then 20 bars of -2% each
    flat = [100.0] * 100
    crash = list(100.0 * np.cumprod([0.98] * 20))
    recovery = [crash[-1]] * 80
    closes = flat + crash + recovery
    bars = _make_bars(closes, hours_per_bar=4.5)
    config = CUSUMConfig(min_inter_event_bars=5)
    result = detect_cusum_events(bars, config=config)

    if len(result) >= 2:
        indices = result["bar_index"].to_list()
        for i in range(1, len(indices)):
            assert indices[i] - indices[i - 1] >= 5, (
                f"Events at bars {indices[i-1]} and {indices[i]} are only "
                f"{indices[i] - indices[i-1]} bars apart (min=5)"
            )


def test_down_event_detected() -> None:
    """Price drops sharply → cusum_side='down' event."""
    # 200 flat bars then steep drop to 90
    closes = [100.0] * 100 + [90.0] * 100
    bars = _make_bars(closes, hours_per_bar=4.5)
    result = detect_cusum_events(bars, config=CUSUMConfig())

    assert len(result) >= 1
    assert result["cusum_side"][0] == "down"


def test_output_columns_match_schema() -> None:
    """Output DataFrame has all CUSUM_EVENTS_SCHEMA columns with correct dtypes."""
    closes = [100.0] * 100 + [110.0] * 100
    bars = _make_bars(closes, hours_per_bar=4.5)
    result = detect_cusum_events(bars)

    expected_columns = {
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
    }
    assert set(result.columns) == expected_columns
    assert result.dtypes[result.columns.index("event_id")] == pl.Utf8
    assert result.dtypes[result.columns.index("bar_index")] == pl.Int64
    assert result.dtypes[result.columns.index("log_return")] == pl.Float64
    assert result.dtypes[result.columns.index("event_ts")] == pl.Datetime
    assert result.dtypes[result.columns.index("_ingested_at")] == pl.Datetime


def test_compute_cusum_stats_aggregates_by_day() -> None:
    """compute_cusum_stats groups events by symbol+date, counts per day."""
    # Build a synthetic events DataFrame (2 events on day 1, 1 on day 2)
    events = pl.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "symbol": ["BTCUSDT"] * 3,
            "event_ts": [
                datetime(2025, 1, 2, 10, 0),
                datetime(2025, 1, 2, 18, 0),
                datetime(2025, 1, 3, 8, 0),
            ],
            "bar_index": [10, 20, 30],
            "log_return": [0.01, 0.02, -0.01],
            "cusum_side": ["up", "up", "down"],
            "cusum_value": [0.05, 0.08, 0.04],
            "threshold_used": [0.03, 0.03, 0.03],
            "ewma_vol": [0.02, 0.02, 0.015],
            "avg_bars_per_day": [5.0, 5.0, 5.0],
            "_ingested_at": [datetime(2025, 1, 5)] * 3,
        }
    )

    stats = compute_cusum_stats(events)

    assert set(stats.columns) == {
        "symbol",
        "date",
        "event_count",
        "threshold_used",
        "ewma_vol",
        "avg_bars_per_day",
    }
    day1 = stats.filter(pl.col("date") == datetime(2025, 1, 2).date())
    assert day1["event_count"][0] == 2
    day2 = stats.filter(pl.col("date") == datetime(2025, 1, 3).date())
    assert day2["event_count"][0] == 1


def test_compute_cusum_stats_empty_events() -> None:
    """Empty events → empty stats DataFrame with correct columns."""
    events = pl.DataFrame(
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
        }
    )
    stats = compute_cusum_stats(events)
    assert stats.is_empty()
    assert "event_count" in stats.columns


def test_trending_market_produces_multiple_events() -> None:
    """500 bars trending up 0.1%/bar → multiple up events, all with side='up'."""
    import numpy as np

    closes = list(100.0 * np.cumprod(1 + np.full(500, 0.001)))
    bars = _make_bars(closes, hours_per_bar=1.5)  # 500*1.5h = 31.25 days
    result = detect_cusum_events(bars, config=CUSUMConfig())

    assert len(result) >= 2
    assert all(s == "up" for s in result["cusum_side"].to_list())


def test_unsorted_bars_are_handled() -> None:
    """detect_cusum_events sorts bars by bar_index internally."""
    closes = [100.0] * 100 + [110.0] * 100
    bars = _make_bars(closes, hours_per_bar=4.5)
    # Shuffle the bars
    bars_shuffled = bars.sample(fraction=1.0, shuffle=True, seed=42)
    result_sorted = detect_cusum_events(bars, config=CUSUMConfig())
    result_shuffled = detect_cusum_events(bars_shuffled, config=CUSUMConfig())
    assert result_sorted["bar_index"].to_list() == result_shuffled["bar_index"].to_list()


def test_sigma_probability_guard_for_zero_vol() -> None:
    """All identical prices: ewma_vol stays 0 → zero-vol guard prevents events
    (first bar skipped due to None log_return, others due to ewma_vol < 1e-10)."""
    closes = [50000.0] * 300
    bars = _make_bars(closes, hours_per_bar=3.0)
    result = detect_cusum_events(bars)
    assert result.is_empty()
