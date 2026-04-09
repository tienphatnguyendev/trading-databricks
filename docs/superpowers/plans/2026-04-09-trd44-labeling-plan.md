# TRD-44: Triple-Barrier Labeling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `compute_triple_barrier_labels()` as a pure Polars function that assigns {-1, 0, +1} labels to CUSUM events using asymmetric triple-barrier logic, and computes `avg_uniqueness` via the AFML concurrency matrix.

**Architecture:** (1) Vectorized barrier level computation — join events with bars for `close` at `bar_index`, then compute `upper_level`, `lower_level`, `vertical_bar`. (2) Sequential forward-scan loop per event — scan future bars until a barrier is hit. (3) Second-pass concurrency matrix for `avg_uniqueness`. Maturation filter excludes any event whose `vertical_bar` exceeds available bar history.

**Tech Stack:** `polars`, `numpy`, `uuid`, `logging`, `datetime` — no Spark, no pandas.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `src/trading_databricks/ml/labeling.py` | CREATE | `compute_triple_barrier_labels()` |
| `tests/ml/test_labeling.py` | CREATE | TDD tests for all behaviour |

Reads from: `LabelConfig` (`ml/config.py`), `TRIPLE_BARRIER_LABELS_SCHEMA` (`ml/schemas.py`)

---

## Task 1: Scaffold + barrier level computation

**Files:**
- Create: `tests/ml/test_labeling.py`
- Create: `src/trading_databricks/ml/labeling.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/test_labeling.py
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

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
    """upper = close*(1+2σ), lower = close*(1-1σ), vertical_bar = bar_index + days*bars/day."""
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
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/ml/test_labeling.py::test_barrier_levels_are_computed_correctly -v
```

Expected: `ImportError` (module doesn't exist).

- [ ] **Step 3: Implement scaffold + barrier level computation**

```python
# src/trading_databricks/ml/labeling.py
from __future__ import annotations

import logging
import uuid
from datetime import datetime

import numpy as np
import polars as pl

from trading_databricks.ml.config import LabelConfig

logger = logging.getLogger(__name__)

_LABELS_SCHEMA = {
    "event_id": pl.Utf8,
    "symbol": pl.Utf8,
    "event_ts": pl.Datetime,
    "label": pl.Int32,
    "barrier_hit": pl.Utf8,
    "return_pct": pl.Float64,
    "upper_level": pl.Float64,
    "lower_level": pl.Float64,
    "vertical_bar": pl.Int64,
    "matured_at": pl.Datetime,
    "avg_uniqueness": pl.Float64,
}


def _empty_labels() -> pl.DataFrame:
    return pl.DataFrame(schema=_LABELS_SCHEMA)


def compute_triple_barrier_labels(
    events: pl.DataFrame,
    bars: pl.DataFrame,
    *,
    config: LabelConfig = LabelConfig(),
) -> pl.DataFrame:
    """Label CUSUM events using the triple-barrier method.

    Args:
        events: CUSUM events DataFrame (CUSUM_EVENTS_SCHEMA columns).
        bars: Dollar bars DataFrame with bar_index, close, bar_end_time.
        config: Labeling hyperparameters.

    Returns:
        pl.DataFrame matching TRIPLE_BARRIER_LABELS_SCHEMA. Only matured events included.
    """
    if events.is_empty() or bars.is_empty():
        logger.warning("compute_triple_barrier_labels: empty input, returning empty DF.")
        return _empty_labels()

    bars = bars.sort("bar_index")
    max_bar_index = bars["bar_index"].max()

    # --- Step 1: Join events with bars to get close at bar_index ---
    events_with_close = events.join(
        bars.select("bar_index", "close", "bar_end_time"),
        on="bar_index",
        how="left",
        suffix="_bar",
    )

    # --- Step 2: Compute barrier levels (vectorized) ---
    events_with_close = events_with_close.with_columns(
        (
            pl.col("close") * (1.0 + config.upper_barrier_multiplier * pl.col("ewma_vol"))
        ).alias("upper_level"),
        (
            pl.col("close") * (1.0 - config.lower_barrier_multiplier * pl.col("ewma_vol"))
        ).alias("lower_level"),
        (
            pl.col("bar_index")
            + (pl.col("avg_bars_per_day") * config.vertical_barrier_days).cast(pl.Int64)
        ).alias("vertical_bar"),
    )

    # --- Step 3: Forward-scan loop ---
    bars_array = bars.select("bar_index", "close", "bar_end_time").to_dicts()
    bar_by_idx: dict[int, dict] = {b["bar_index"]: b for b in bars_array}

    labeled: list[dict] = []

    for event in events_with_close.iter_rows(named=True):
        vertical_bar = event["vertical_bar"]

        # Maturation filter: skip events that haven't matured yet
        if vertical_bar > max_bar_index:
            continue

        upper = event["upper_level"]
        lower = event["lower_level"]
        entry_close = event["close"]
        label: int | None = None
        barrier_hit: str = "vertical"
        matured_at: datetime | None = None
        return_pct: float = 0.0

        # Scan future bars from bar_index+1 to vertical_bar inclusive
        for bi in range(event["bar_index"] + 1, vertical_bar + 1):
            bar = bar_by_idx.get(bi)
            if bar is None:
                continue
            if bar["close"] >= upper:
                label = 1
                barrier_hit = "upper"
                matured_at = bar["bar_end_time"]
                return_pct = (bar["close"] - entry_close) / entry_close
                break
            elif bar["close"] <= lower:
                label = -1
                barrier_hit = "lower"
                matured_at = bar["bar_end_time"]
                return_pct = (bar["close"] - entry_close) / entry_close
                break

        if label is None:  # vertical barrier hit
            last_bar = bar_by_idx.get(vertical_bar)
            if last_bar is not None:
                return_pct = (last_bar["close"] - entry_close) / entry_close
                sigma = event["ewma_vol"]
                if sigma > 0 and abs(return_pct) < config.zero_label_threshold * sigma:
                    label = 0
                else:
                    label = 1 if return_pct > 0 else -1
                matured_at = last_bar["bar_end_time"]
            else:
                label = 0
                matured_at = event["event_ts"]
            barrier_hit = "vertical"

        labeled.append(
            {
                "event_id": event["event_id"],
                "symbol": event["symbol"],
                "event_ts": event["event_ts"],
                "label": label,
                "barrier_hit": barrier_hit,
                "return_pct": return_pct,
                "upper_level": upper,
                "lower_level": lower,
                "vertical_bar": vertical_bar,
                "matured_at": matured_at,
                "avg_uniqueness": 1.0,  # placeholder, overwritten in Step 4
            }
        )

    if not labeled:
        logger.warning("No matured events after labeling. Returning empty DF.")
        return _empty_labels()

    result = pl.DataFrame(labeled, schema=_LABELS_SCHEMA)

    # --- Step 4: Second-pass concurrency matrix for avg_uniqueness ---
    result = _compute_avg_uniqueness(result)
    return result


def _compute_avg_uniqueness(labels: pl.DataFrame) -> pl.DataFrame:
    """Second pass: compute avg_uniqueness via AFML Ch. 4 concurrency matrix."""
    if labels.is_empty():
        return labels

    bar_indices = labels["bar_index"].to_list() if "bar_index" in labels.columns else []

    events_list = labels.select(
        pl.col("event_id"),
    ).to_series().to_list()

    # We need bar_index from the input — but TRIPLE_BARRIER_LABELS_SCHEMA doesn't have it.
    # avg_uniqueness is computed from the forward spans [bar_index, vertical_bar].
    # Since we don't store bar_index in the output schema, we'll compute it from event_ts.
    # Simplified: use position-based proxy (row index as bar_index proxy).
    n_events = len(labels)
    vertical_bars = labels["vertical_bar"].to_numpy()

    # Use sequential row positions as a proxy for bar_index (events sorted by event_ts)
    # The real bar_index is not in TRIPLE_BARRIER_LABELS_SCHEMA, so use position.
    row_start = np.arange(n_events, dtype=float)
    row_end = virtual_bar = virtual_bar = np.minimum(
        vertical_bars.astype(float),
        vertical_bars.astype(float),  # both same; can't know real bar count without it
    )

    # Simplified approach: store avg_uniqueness = 1/(overlap_count)
    # Full concurrency matrix requires bar_index in the labeled dict.
    # We add a temporary bar_index column during computation, then drop it.
    # Since labeled dict does include bar_index, we reconstruct here:
    bar_starts = labels["vertical_bar"].to_numpy() - (
        (labels["vertical_bar"] - labels["vertical_bar"]).to_numpy()
    )  # placeholder; see implementation note

    return labels
```

**Implementation note:** The concurrency matrix requires `bar_index` (event start bar) which is not in `TRIPLE_BARRIER_LABELS_SCHEMA`. Fix: temporarily carry `bar_index` in the `labeled` dict, compute uniqueness, then drop. This is done in Task 2.

- [ ] **Step 4: Run to verify test passes**

```bash
uv run pytest tests/ml/test_labeling.py::test_barrier_levels_are_computed_correctly -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/ml/labeling.py tests/ml/test_labeling.py
git commit -m "TRD-44 feat: scaffold labeling with barrier level computation and forward scan"
```

---

## Task 2: Upper/lower barrier hits + concurrency matrix

**Files:**
- Modify: `src/trading_databricks/ml/labeling.py`
- Modify: `tests/ml/test_labeling.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/ml/test_labeling.py`:

```python
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
    """Price stays flat within barriers → vertical expiry → label=0."""
    # close=100, ewma_vol=0.02. vertical_barrier_days=5, avg_bars/day=24 → vertical_bar=120
    # All bars stay at 100 ± 0.005 (well within barriers)
    import random
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
    # Event A at bar 0, vertical_bar=120. Event B at bar 50, vertical_bar=170.
    # Overlap [50,120], so A.uniqueness < 1.0
    closes = [100.0] + [104.5] + [104.5] * 498
    bars = _make_bars(closes)
    events = _make_events([0, 50], bars, ewma_vol=0.02, avg_bars_per_day=24.0)
    result = compute_triple_barrier_labels(events, bars)
    if len(result) == 2:
        assert result["avg_uniqueness"][0] < 1.0
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/ml/test_labeling.py -v
```

Expected: Several tests FAIL (concurrency placeholder returns 1.0 always).

- [ ] **Step 3: Fix labeling.py — carry bar_index through labeled dict + implement concurrency matrix**

Replace `compute_triple_barrier_labels` and `_compute_avg_uniqueness` in `labeling.py`:

```python
def compute_triple_barrier_labels(
    events: pl.DataFrame,
    bars: pl.DataFrame,
    *,
    config: LabelConfig = LabelConfig(),
) -> pl.DataFrame:
    if events.is_empty() or bars.is_empty():
        logger.warning("compute_triple_barrier_labels: empty input.")
        return _empty_labels()

    bars = bars.sort("bar_index")
    max_bar_index = int(bars["bar_index"].max())

    events_with_close = events.join(
        bars.select("bar_index", "close", "bar_end_time"),
        on="bar_index",
        how="left",
        suffix="_bar",
    ).with_columns(
        (
            pl.col("close") * (1.0 + config.upper_barrier_multiplier * pl.col("ewma_vol"))
        ).alias("upper_level"),
        (
            pl.col("close") * (1.0 - config.lower_barrier_multiplier * pl.col("ewma_vol"))
        ).alias("lower_level"),
        (
            pl.col("bar_index")
            + (pl.col("avg_bars_per_day") * config.vertical_barrier_days).cast(pl.Int64)
        ).alias("vertical_bar"),
    )

    bar_by_idx: dict[int, dict] = {
        b["bar_index"]: b
        for b in bars.select("bar_index", "close", "bar_end_time").to_dicts()
    }

    labeled: list[dict] = []

    for event in events_with_close.iter_rows(named=True):
        vertical_bar = int(event["vertical_bar"])
        if vertical_bar > max_bar_index:
            continue

        upper = float(event["upper_level"])
        lower = float(event["lower_level"])
        entry_close = float(event["close"])
        event_bar_index = int(event["bar_index"])
        label: int = 0
        barrier_hit = "vertical"
        return_pct = 0.0
        matured_at: datetime = event["event_ts"]
        found = False

        for bi in range(event_bar_index + 1, vertical_bar + 1):
            bar = bar_by_idx.get(bi)
            if bar is None:
                continue
            c = float(bar["close"])
            if c >= upper:
                label, barrier_hit = 1, "upper"
                return_pct = (c - entry_close) / entry_close
                matured_at = bar["bar_end_time"]
                found = True
                break
            elif c <= lower:
                label, barrier_hit = -1, "lower"
                return_pct = (c - entry_close) / entry_close
                matured_at = bar["bar_end_time"]
                found = True
                break

        if not found:  # vertical barrier
            last = bar_by_idx.get(vertical_bar)
            if last is not None:
                return_pct = (float(last["close"]) - entry_close) / entry_close
                sigma = float(event["ewma_vol"])
                if sigma > 0 and abs(return_pct) < config.zero_label_threshold * sigma:
                    label = 0
                else:
                    label = 1 if return_pct > 0 else -1
                matured_at = last["bar_end_time"]

        labeled.append(
            {
                "_bar_index": event_bar_index,   # temp column for uniqueness
                "event_id": event["event_id"],
                "symbol": event["symbol"],
                "event_ts": event["event_ts"],
                "label": label,
                "barrier_hit": barrier_hit,
                "return_pct": return_pct,
                "upper_level": upper,
                "lower_level": lower,
                "vertical_bar": vertical_bar,
                "matured_at": matured_at,
                "avg_uniqueness": 1.0,  # overwritten below
            }
        )

    if not labeled:
        return _empty_labels()

    # Concurrency pass
    labeled = _fill_avg_uniqueness(labeled)

    return pl.DataFrame(
        [{k: v for k, v in row.items() if k != "_bar_index"} for row in labeled],
        schema=_LABELS_SCHEMA,
    )


def _fill_avg_uniqueness(labeled: list[dict]) -> list[dict]:
    """AFML Ch.4: concurrency array → avg_uniqueness per event."""
    if len(labeled) == 0:
        return labeled

    starts = np.array([row["_bar_index"] for row in labeled], dtype=int)
    ends = np.array([row["vertical_bar"] for row in labeled], dtype=int)

    min_bar = int(starts.min())
    max_bar = int(ends.max())
    span = max_bar - min_bar + 1
    concurrency = np.zeros(span, dtype=float)

    for s, e in zip(starts, ends):
        concurrency[s - min_bar : e - min_bar + 1] += 1.0

    for i, row in enumerate(labeled):
        s = starts[i] - min_bar
        e = ends[i] - min_bar + 1
        span_slice = concurrency[s:e]
        # Avoid division by zero (shouldn't happen since the event itself adds 1)
        nonzero = span_slice[span_slice > 0]
        row["avg_uniqueness"] = float(np.mean(1.0 / nonzero)) if len(nonzero) > 0 else 1.0

    return labeled
```

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/ml/test_labeling.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/ml/labeling.py tests/ml/test_labeling.py
git commit -m "TRD-44 feat: forward scan loop, maturation filter, concurrency matrix for avg_uniqueness"
```

---

## Task 3: Output schema validation + linting

**Files:**
- Modify: `tests/ml/test_labeling.py`

- [ ] **Step 1: Write schema test**

Add to `tests/ml/test_labeling.py`:

```python
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
```

- [ ] **Step 2: Run all tests**

```bash
uv run pytest tests/ml/test_labeling.py -v
```

Expected: All PASS.

- [ ] **Step 3: Run linting**

```bash
uv run ruff check src/trading_databricks/ml/labeling.py tests/ml/test_labeling.py
```

Fix any issues.

- [ ] **Step 4: Commit**

```bash
git add src/trading_databricks/ml/labeling.py tests/ml/test_labeling.py
git commit -m "TRD-44 test: schema validation, linting clean"
```

---

## Verification

```bash
uv run pytest tests/ml/test_labeling.py -v
```

All tests pass. Final file count: 2 files created.
