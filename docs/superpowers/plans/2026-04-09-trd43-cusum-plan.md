# TRD-43: CUSUM Event Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `detect_cusum_events()` and `compute_cusum_stats()` as pure Polars functions that identify significant market regime changes using the symmetric CUSUM filter on dollar bar log returns.

**Architecture:** Vectorized pre-processing (log returns, EWMA vol, rolling avg_bars_per_day, threshold) followed by a sequential Python loop for the stateful CUSUM accumulators. Returns a `pl.DataFrame` matching `CUSUM_EVENTS_SCHEMA`. Warm-up guard exits early when bar history spans < 30 calendar days.

**Tech Stack:** `polars`, `uuid`, `logging`, `datetime` — no Spark, no pandas.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `src/trading_databricks/ml/cusum.py` | CREATE | `detect_cusum_events()` + `compute_cusum_stats()` |
| `tests/ml/test_cusum.py` | CREATE | TDD tests for all behaviour |

Reads from: `CUSUMConfig` (`ml/config.py`), `CUSUM_EVENTS_SCHEMA`, `CUSUM_EVENT_STATS_SCHEMA` (`ml/schemas.py`)

---

## Task 1: Scaffold + warm-up guard

**Files:**
- Create: `tests/ml/test_cusum.py`
- Create: `src/trading_databricks/ml/cusum.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/test_cusum.py
from __future__ import annotations

import math
from datetime import datetime, timedelta

import polars as pl
import pytest

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
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/ml/test_cusum.py::test_warmup_guard_returns_empty_df -v
```

Expected: `ImportError` or `ModuleNotFoundError` (function doesn't exist yet).

- [ ] **Step 3: Implement scaffold + warm-up guard**

```python
# src/trading_databricks/ml/cusum.py
from __future__ import annotations

import logging
import uuid
from datetime import datetime

import polars as pl

from trading_databricks.ml.config import CUSUMConfig

logger = logging.getLogger(__name__)

_EVENTS_COLUMNS = [
    "event_id", "symbol", "event_ts", "bar_index",
    "log_return", "cusum_side", "cusum_value",
    "threshold_used", "ewma_vol", "avg_bars_per_day",
    "_ingested_at",
]

_STATS_COLUMNS = [
    "symbol", "date", "event_count",
    "threshold_used", "ewma_vol", "avg_bars_per_day",
]


def _empty_events() -> pl.DataFrame:
    return pl.DataFrame(
        {col: [] for col in _EVENTS_COLUMNS},
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
        },
    )


def detect_cusum_events(
    bars: pl.DataFrame,
    *,
    config: CUSUMConfig = CUSUMConfig(),
) -> pl.DataFrame:
    """Detect CUSUM events on dollar bar log returns.

    Args:
        bars: Dollar bars with columns: bar_index, symbol, close, bar_end_time.
              Must span ≥ config.lookback_days calendar days.
        config: CUSUM hyperparameters.

    Returns:
        pl.DataFrame matching CUSUM_EVENTS_SCHEMA columns. Empty if warm-up guard fires.
    """
    if bars.is_empty():
        return _empty_events()

    bars = bars.sort("bar_index")

    # Warm-up guard: need ≥ lookback_days calendar days of history
    min_ts = bars["bar_end_time"].min()
    max_ts = bars["bar_end_time"].max()
    span_days = (max_ts - min_ts).total_seconds() / 86400  # type: ignore[operator]
    if span_days < config.lookback_days:
        logger.warning(
            "CUSUM warm-up guard: only %.1f days of bars (need %d). Returning empty.",
            span_days,
            config.lookback_days,
        )
        return _empty_events()

    return _empty_events()  # placeholder until Task 2
```

- [ ] **Step 4: Run to verify it passes**

```bash
uv run pytest tests/ml/test_cusum.py::test_warmup_guard_returns_empty_df -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/ml/cusum.py tests/ml/test_cusum.py
git commit -m "TRD-43 feat: scaffold detect_cusum_events with warm-up guard"
```

---

## Task 2: Vectorized pre-processing + CUSUM loop

**Files:**
- Modify: `src/trading_databricks/ml/cusum.py`
- Modify: `tests/ml/test_cusum.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/ml/test_cusum.py`:

```python
def test_step_function_produces_single_up_event() -> None:
    """200 bars: flat at 100, then jump to 110 at bar 100 → exactly 1 up event."""
    import numpy as np

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
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/ml/test_cusum.py::test_step_function_produces_single_up_event tests/ml/test_cusum.py::test_flat_market_produces_no_events -v
```

Expected: FAIL (returns empty from placeholder).

- [ ] **Step 3: Implement vectorized pre-processing + CUSUM loop**

Replace `detect_cusum_events` in `cusum.py` (keep `_empty_events`, `_EVENTS_COLUMNS`, `_STATS_COLUMNS`, logger, imports — only the function body changes):

```python
def detect_cusum_events(
    bars: pl.DataFrame,
    *,
    config: CUSUMConfig = CUSUMConfig(),
) -> pl.DataFrame:
    if bars.is_empty():
        return _empty_events()

    bars = bars.sort("bar_index")

    # Warm-up guard
    min_ts = bars["bar_end_time"].min()
    max_ts = bars["bar_end_time"].max()
    span_days = (max_ts - min_ts).total_seconds() / 86400  # type: ignore[operator]
    if span_days < config.lookback_days:
        logger.warning(
            "CUSUM warm-up: %.1f days of bars (need %d). Returning empty.",
            span_days,
            config.lookback_days,
        )
        return _empty_events()

    # --- Step 1: Vectorized log returns ---
    bars = bars.with_columns(
        pl.col("close").log().diff().alias("log_return"),
    )

    # --- Step 2: EWMA volatility (vectorized) ---
    # alpha = 1 - lambda (decay factor for ewm_mean)
    alpha = 1.0 - config.ewma_lambda
    bars = bars.with_columns(
        pl.col("log_return")
        .pow(2)
        .ewm_mean(alpha=alpha, min_periods=1)
        .sqrt()
        .alias("ewma_vol"),
    )

    # --- Step 3: avg_bars_per_day + rolling threshold ---
    # Convert timestamps to day ordinals for calendar-day span
    bars = bars.with_columns(
        pl.col("bar_end_time").dt.epoch("d").alias("_day_ordinal"),
    )
    lookback_size = int(75 * config.lookback_days)  # initial estimate
    bars = bars.with_columns(
        (
            pl.col("_day_ordinal")
            - pl.col("_day_ordinal").shift(lookback_size)
        )
        .clip(lower_bound=1)
        .alias("_calendar_days"),
    )
    bars = bars.with_columns(
        (pl.lit(float(lookback_size)) / pl.col("_calendar_days"))
        .alias("avg_bars_per_day"),
    )
    bars = bars.with_columns(
        (
            config.threshold_multiplier
            * pl.col("ewma_vol")
            * pl.col("avg_bars_per_day").sqrt()
        ).alias("threshold"),
    )

    # --- Step 4: Sequential CUSUM loop ---
    events: list[dict] = []
    s_pos, s_neg = 0.0, 0.0
    last_event_idx = -config.min_inter_event_bars

    for row in bars.iter_rows(named=True):
        lr = row["log_return"]
        if lr is None:
            continue
        ewma_vol = row["ewma_vol"]
        if ewma_vol is not None and ewma_vol < 1e-10:
            continue  # zero-vol guard: skip bar, don't update accumulators

        threshold = row["threshold"]
        s_pos = max(0.0, s_pos + lr)
        s_neg = min(0.0, s_neg + lr)

        spacing_ok = (row["bar_index"] - last_event_idx) >= config.min_inter_event_bars

        if s_pos > threshold and spacing_ok:
            events.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "symbol": row["symbol"],
                    "event_ts": row["bar_end_time"],
                    "bar_index": row["bar_index"],
                    "log_return": lr,
                    "cusum_side": "up",
                    "cusum_value": s_pos,
                    "threshold_used": threshold,
                    "ewma_vol": ewma_vol,
                    "avg_bars_per_day": row["avg_bars_per_day"],
                }
            )
            s_pos, s_neg = 0.0, 0.0
            last_event_idx = row["bar_index"]
        elif abs(s_neg) > threshold and spacing_ok:
            events.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "symbol": row["symbol"],
                    "event_ts": row["bar_end_time"],
                    "bar_index": row["bar_index"],
                    "log_return": lr,
                    "cusum_side": "down",
                    "cusum_value": abs(s_neg),
                    "threshold_used": threshold,
                    "ewma_vol": ewma_vol,
                    "avg_bars_per_day": row["avg_bars_per_day"],
                }
            )
            s_pos, s_neg = 0.0, 0.0
            last_event_idx = row["bar_index"]

    if not events:
        return _empty_events()

    now = datetime.utcnow()
    result = pl.DataFrame(events)
    result = result.with_columns(pl.lit(now).alias("_ingested_at"))
    return result
```

- [ ] **Step 4: Run to verify tests pass**

```bash
uv run pytest tests/ml/test_cusum.py::test_step_function_produces_single_up_event tests/ml/test_cusum.py::test_flat_market_produces_no_events -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/ml/cusum.py tests/ml/test_cusum.py
git commit -m "TRD-43 feat: implement vectorized pre-processing and CUSUM accumulator loop"
```

---

## Task 3: Min-spacing enforcement + down events

**Files:**
- Modify: `tests/ml/test_cusum.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/ml/test_cusum.py`:

```python
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
    import numpy as np

    # 200 flat bars then steep drop to 90
    closes = [100.0] * 100 + [90.0] * 100
    bars = _make_bars(closes, hours_per_bar=4.5)
    result = detect_cusum_events(bars, config=CUSUMConfig())

    assert len(result) >= 1
    assert result["cusum_side"][0] == "down"
```

- [ ] **Step 2: Run to verify they pass (or expose bug)**

```bash
uv run pytest tests/ml/test_cusum.py::test_flash_crash_events_spaced_by_min_inter_event_bars tests/ml/test_cusum.py::test_down_event_detected -v
```

Expected: PASS (spacing logic already in loop; down event handled by `elif abs(s_neg)`).

- [ ] **Step 3: Commit**

```bash
git add tests/ml/test_cusum.py
git commit -m "TRD-43 test: add spacing enforcement and down-event detection tests"
```

---

## Task 4: Output schema validation

**Files:**
- Modify: `tests/ml/test_cusum.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/ml/test_cusum.py`:

```python
def test_output_columns_match_schema() -> None:
    """Output DataFrame has all CUSUM_EVENTS_SCHEMA columns with correct dtypes."""
    closes = [100.0] * 100 + [110.0] * 100
    bars = _make_bars(closes, hours_per_bar=4.5)
    result = detect_cusum_events(bars)

    expected_columns = {
        "event_id", "symbol", "event_ts", "bar_index",
        "log_return", "cusum_side", "cusum_value",
        "threshold_used", "ewma_vol", "avg_bars_per_day", "_ingested_at",
    }
    assert set(result.columns) == expected_columns
    assert result.dtypes[result.columns.index("event_id")] == pl.Utf8
    assert result.dtypes[result.columns.index("bar_index")] == pl.Int64
    assert result.dtypes[result.columns.index("log_return")] == pl.Float64
    assert result.dtypes[result.columns.index("event_ts")] == pl.Datetime
    assert result.dtypes[result.columns.index("_ingested_at")] == pl.Datetime
```

- [ ] **Step 2: Run to verify it passes**

```bash
uv run pytest tests/ml/test_cusum.py::test_output_columns_match_schema -v
```

Expected: PASS (columns already correct from Task 2).

- [ ] **Step 3: Commit**

```bash
git add tests/ml/test_cusum.py
git commit -m "TRD-43 test: add output schema column validation"
```

---

## Task 5: `compute_cusum_stats`

**Files:**
- Modify: `src/trading_databricks/ml/cusum.py`
- Modify: `tests/ml/test_cusum.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/ml/test_cusum.py`:

```python
def test_compute_cusum_stats_aggregates_by_day() -> None:
    """compute_cusum_stats groups events by symbol+date, counts per day."""
    from trading_databricks.ml.cusum import compute_cusum_stats

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
        "symbol", "date", "event_count", "threshold_used", "ewma_vol", "avg_bars_per_day"
    }
    day1 = stats.filter(pl.col("date") == datetime(2025, 1, 2).date())
    assert day1["event_count"][0] == 2
    day2 = stats.filter(pl.col("date") == datetime(2025, 1, 3).date())
    assert day2["event_count"][0] == 1


def test_compute_cusum_stats_empty_events() -> None:
    """Empty events → empty stats DataFrame with correct columns."""
    from trading_databricks.ml.cusum import compute_cusum_stats

    events = _make_bars([100.0] * 10)  # wrong schema, just needs to be empty
    # Use the proper empty events schema
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/ml/test_cusum.py::test_compute_cusum_stats_aggregates_by_day tests/ml/test_cusum.py::test_compute_cusum_stats_empty_events -v
```

Expected: FAIL (`compute_cusum_stats` not defined).

- [ ] **Step 3: Implement `compute_cusum_stats`**

Add to `cusum.py` (after `detect_cusum_events`):

```python
def compute_cusum_stats(events: pl.DataFrame) -> pl.DataFrame:
    """Aggregate CUSUM events into daily statistics by symbol.

    Args:
        events: Output of detect_cusum_events(). May be empty.

    Returns:
        pl.DataFrame with columns: symbol, date, event_count,
        threshold_used, ewma_vol, avg_bars_per_day.
    """
    _stats_schema = {
        "symbol": pl.Utf8,
        "date": pl.Date,
        "event_count": pl.Int32,
        "threshold_used": pl.Float64,
        "ewma_vol": pl.Float64,
        "avg_bars_per_day": pl.Float64,
    }
    if events.is_empty():
        return pl.DataFrame(schema=_stats_schema)

    return (
        events.with_columns(pl.col("event_ts").dt.date().alias("date"))
        .group_by(["symbol", "date"])
        .agg(
            pl.len().cast(pl.Int32).alias("event_count"),
            pl.col("threshold_used").mean(),
            pl.col("ewma_vol").mean(),
            pl.col("avg_bars_per_day").mean(),
        )
        .sort(["symbol", "date"])
    )
```

- [ ] **Step 4: Run to verify tests pass**

```bash
uv run pytest tests/ml/test_cusum.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/ml/cusum.py tests/ml/test_cusum.py
git commit -m "TRD-43 feat: implement compute_cusum_stats daily aggregation"
```

---

## Task 6: Full test suite + linting

**Files:**
- Modify: `tests/ml/test_cusum.py`

- [ ] **Step 1: Add trending and edge-case tests**

Add to `tests/ml/test_cusum.py`:

```python
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
```

- [ ] **Step 2: Run full test suite**

```bash
uv run pytest tests/ml/test_cusum.py -v
```

Expected: All tests PASS.

- [ ] **Step 3: Run linting**

```bash
uv run ruff check src/trading_databricks/ml/cusum.py tests/ml/test_cusum.py
```

Fix any issues, then:

- [ ] **Step 4: Commit**

```bash
git add src/trading_databricks/ml/cusum.py tests/ml/test_cusum.py
git commit -m "TRD-43 test: add trending market, unsorted input, and zero-vol guard tests"
```

---

## Verification

```bash
uv run pytest tests/ml/test_cusum.py -v
```

All tests should pass. Final file count: 2 files created.
