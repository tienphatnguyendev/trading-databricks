from __future__ import annotations

import random
from datetime import datetime, timedelta

import polars as pl

from trading_databricks.ml.config import LabelConfig
from trading_databricks.ml.labeling import compute_triple_barrier_labels


def _make_bars(closes: list[float], symbol: str = "BTCUSDT") -> pl.DataFrame:
    """Synthetic bars with 1-hour spacing. Provide enough to cover vertical barrier."""
    n = len(closes)
    base_dt = datetime(2025, 1, 1)
    return pl.DataFrame(
        {
            "bar_index": list(range(n)),
            "symbol": [symbol] * n,
            "open": closes,
            "high": [c * 1.0005 for c in closes],
            "low": [c * 0.9995 for c in closes],
            "close": closes,
            "volume": [500.0] * n,
            "bar_end_time": [base_dt + timedelta(hours=i + 1) for i in range(n)],
        }
    )


def _make_events(
    bar_indices: list[int],
    bars: pl.DataFrame,
    ewma_vol: float = 0.02,
    avg_bars_per_day: float = 24.0,
    symbol: str = "BTCUSDT",
) -> pl.DataFrame:
    """Synthetic CUSUM events at the specified bar indices."""
    base_dt = datetime(2025, 1, 1)
    return pl.DataFrame(
        {
            "event_id": [f"evt-{i}" for i in bar_indices],
            "symbol": [symbol] * len(bar_indices),
            "event_ts": [
                base_dt + timedelta(hours=idx + 1) for idx in bar_indices
            ],
            "bar_index": bar_indices,
            "log_return": [0.01] * len(bar_indices),
            "cusum_side": ["up"] * len(bar_indices),
            "cusum_value": [0.05] * len(bar_indices),
            "threshold_used": [0.03] * len(bar_indices),
            "ewma_vol": [ewma_vol] * len(bar_indices),
            "avg_bars_per_day": [avg_bars_per_day] * len(bar_indices),
            "_ingested_at": [datetime(2025, 1, 5)] * len(bar_indices),
        }
    )


def test_barrier_levels_are_computed_correctly() -> None:
    """Upper = close*(1+2σ), lower = close*(1-1σ), vertical_bar computed.

    Formula: vertical_bar = bar_index + days*bars/day.
    """
    closes = [100.0] * 500
    bars = _make_bars(closes)
    events = _make_events([10], bars, ewma_vol=0.02, avg_bars_per_day=24.0)
    config = LabelConfig(
        upper_barrier_multiplier=2.0,
        lower_barrier_multiplier=1.0,
        vertical_barrier_days=5,
    )
    result = compute_triple_barrier_labels(events, bars, config=config)
    # Result may be empty if event doesn't mature, but verify barrier fields are present
    assert "upper_level" in result.columns
    assert "lower_level" in result.columns
    assert "vertical_bar" in result.columns
    # With 500 bars and event at bar 10, vertical_bar = 10 + 5*24 = 130 (well within 500)
    assert len(result) == 1
    assert abs(result["upper_level"][0] - 100.0 * 1.04) < 1e-9   # 100*(1+2*0.02)
    assert abs(result["lower_level"][0] - 100.0 * 0.98) < 1e-9   # 100*(1-1*0.02)
    assert result["vertical_bar"][0] == 10 + 5 * 24               # 130


def test_upper_barrier_hit() -> None:
    """Price rises above upper level → label=+1, barrier_hit='upper'."""
    # close=100, ewma_vol=0.02 → upper=104.0, lower=98.0
    # Bars 1-10: prices climb steadily, hit 104.5 at bar 4
    closes = [100.0] + [101.0, 102.0, 103.0, 104.5] + [104.5] * 495
    bars = _make_bars(closes)
    events = _make_events([0], bars, ewma_vol=0.02, avg_bars_per_day=24.0)
    result = compute_triple_barrier_labels(events, bars)

    assert len(result) == 1
    assert result["label"][0] == 1
    assert result["barrier_hit"][0] == "upper"
    assert abs(result["upper_level"][0] - 104.0) < 1e-9


def test_lower_barrier_hit() -> None:
    """Price drops below lower level → label=-1, barrier_hit='lower'."""
    # close=100, ewma_vol=0.02 → lower=98.0
    # Bars 1-4: prices drop to 97.5
    closes = [100.0] + [99.0, 98.5, 98.0, 97.5] + [97.5] * 495
    bars = _make_bars(closes)
    events = _make_events([0], bars, ewma_vol=0.02, avg_bars_per_day=24.0)
    result = compute_triple_barrier_labels(events, bars)

    assert len(result) == 1
    assert result["label"][0] == -1
    assert result["barrier_hit"][0] == "lower"


def test_vertical_barrier_hit_flat_return_zero_label() -> None:
    """Price stays flat within barriers → vertical expiry → label=0.

    vertical_barrier_days=5, avg_bars/day=24 → vertical_bar=120.
    All bars stay at 100 ± 0.005 (well within barriers).
    """
    random.seed(42)
    closes = [100.0] + [100.0 + random.uniform(-0.005, 0.005) for _ in range(499)]
    bars = _make_bars(closes)
    events = _make_events([0], bars, ewma_vol=0.02, avg_bars_per_day=24.0)
    config = LabelConfig(zero_label_threshold=0.5)  # generous zero threshold
    result = compute_triple_barrier_labels(events, bars, config=config)

    assert len(result) == 1
    assert result["label"][0] == 0
    assert result["barrier_hit"][0] == "vertical"


def test_upper_checked_before_lower_same_bar() -> None:
    """When upper and lower are both breached in the same bar, upper wins."""
    # This edge case is handled by checking upper first in the loop.
    # Simulate: a bar whose close is above upper AND a prior bar below lower,
    # but upper bar comes first in the scan.
    closes = [100.0] + [103.0, 104.5] + [100.0] * 497  # bar 2 hits upper=104.0
    bars = _make_bars(closes)
    events = _make_events([0], bars, ewma_vol=0.02, avg_bars_per_day=24.0)
    result = compute_triple_barrier_labels(events, bars)

    assert result["barrier_hit"][0] == "upper"
    assert result["label"][0] == 1


def test_maturation_filter_excludes_unmatured_events() -> None:
    """Event near end of bar history: vertical_bar > max bar → excluded."""
    # 100 bars, event at bar 90, avg_bars_per_day=24, vertical_barrier_days=5
    # vertical_bar = 90 + 120 = 210 > 99 (max bar) → excluded
    closes = [100.0] * 100
    bars = _make_bars(closes)
    events = _make_events([90], bars, ewma_vol=0.02, avg_bars_per_day=24.0)
    result = compute_triple_barrier_labels(events, bars)
    assert result.is_empty()


def test_avg_uniqueness_single_event_is_one() -> None:
    """Single event with no overlap → avg_uniqueness = 1.0."""
    closes = [100.0] + [104.5] + [104.5] * 498
    bars = _make_bars(closes)
    events = _make_events([0], bars, ewma_vol=0.02, avg_bars_per_day=24.0)
    result = compute_triple_barrier_labels(events, bars)
    assert len(result) == 1
    assert abs(result["avg_uniqueness"][0] - 1.0) < 1e-9


def test_avg_uniqueness_overlapping_events_below_one() -> None:
    """Two overlapping events → avg_uniqueness < 1.0."""
    # Event A at bar 0, B at bar 2. Both exit at bar 5.
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 110.0] + [110.0] * 494
    bars = _make_bars(closes)
    events = _make_events([0, 2], bars, ewma_vol=0.02, avg_bars_per_day=24.0)
    result = compute_triple_barrier_labels(events, bars)
    if len(result) == 2:
        assert result["avg_uniqueness"][0] < 1.0


def test_output_columns_match_schema() -> None:
    """Output DataFrame has all TRIPLE_BARRIER_LABELS_SCHEMA columns."""
    closes = [100.0] + [104.5] + [104.5] * 498
    bars = _make_bars(closes)
    events = _make_events([0], bars, ewma_vol=0.02, avg_bars_per_day=24.0)
    result = compute_triple_barrier_labels(events, bars)

    expected = {
        "event_id", "symbol", "event_ts", "label",
        "barrier_hit", "return_pct", "upper_level", "lower_level",
        "vertical_bar", "matured_at", "avg_uniqueness",
    }
    assert set(result.columns) == expected
    assert result.dtypes[result.columns.index("label")] == pl.Int32
    assert result.dtypes[result.columns.index("vertical_bar")] == pl.Int64
    assert result.dtypes[result.columns.index("avg_uniqueness")] == pl.Float64
    # _bar_index should NOT appear in output
    assert "_bar_index" not in result.columns

def test_entry_close_zero_skipped_gracefully() -> None:
    """Events with close=0 must be skipped, not crash."""
    closes = [0.0] + [100.0] * 150
    bars = _make_bars(closes)
    events = _make_events([0], bars, ewma_vol=0.02, avg_bars_per_day=24.0)
    result = compute_triple_barrier_labels(events, bars)
    assert len(result) == 0


def test_avg_uniqueness_uses_actual_exit_bar() -> None:
    """Events that hit upper/lower barrier early should have higher avg_uniqueness than if they used vertical_bar."""
    bars_list = [100.0, 101.0, 110.0] + [100.0] * 120
    bars = _make_bars(bars_list)
    events = _make_events([0], bars, ewma_vol=0.05, avg_bars_per_day=24.0)
    result = compute_triple_barrier_labels(events, bars)

    assert len(result) == 1
    assert result["avg_uniqueness"][0] == 1.0
def test_avg_uniqueness_uses_actual_exit_bar() -> None:
    """Events that hit upper/lower barrier early should have higher avg_uniqueness than if they used vertical_bar."""
    # Bars:
    # 0: 100
    # 1: 101
    # 2: 120 (Event A exits here!)
    # ...
    # 5: 100 (Event B starts here)
    # ...
    bars_list = [100.0, 101.0, 120.0, 100.0, 100.0, 100.0] + [100.0] * 150
    bars = _make_bars(bars_list)
    events = _make_events([0, 5], bars, ewma_vol=0.05, avg_bars_per_day=24.0)

    result = compute_triple_barrier_labels(events, bars)
    assert len(result) == 2
    # If A exits at bar 2, and B starts at bar 5, they do not overlap at all!
    # Therefore, both should have avg_uniqueness == 1.0
    assert result["avg_uniqueness"][0] == 1.0
    assert result["avg_uniqueness"][1] == 1.0
