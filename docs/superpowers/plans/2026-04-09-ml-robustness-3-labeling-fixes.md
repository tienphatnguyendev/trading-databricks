# ML Robustness Labeling Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three critical bugs in the triple barrier labeling logic to ensure robustness during zero-division, early exits, and zero-volatility scenarios.

**Architecture:** The fixes will be applied to the core Polars/NumPy `compute_triple_barrier_labels` function in `labeling.py`. Tests will be added to `test_labeling.py` using synthetic bar and event data to ensure behavioral correctness.

**Tech Stack:** Python, Polars, NumPy, pytest

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `src/trading_databricks/ml/labeling.py` | MODIFY | Fix 3 bugs related to 0-division, exit bars, and zero-volatility. |
| `tests/ml/test_labeling.py` | MODIFY | Add edge case tests for the bugs. |

---

### Task 1: Fix entry_close=0 Division by Zero

**Files:**
- Modify: `tests/ml/test_labeling.py`
- Modify: `src/trading_databricks/ml/labeling.py`

- [ ] **Step 1: Write the failing test**

Open `tests/ml/test_labeling.py` and add the following test function:

```python
def test_entry_close_zero_skipped_gracefully() -> None:
    """Events with close=0 must be skipped, not crash."""
    bars = _make_bars([0.0, 100.0, 101.0, 102.0])
    events = _make_events([0], bars, ewma_vol=0.02, avg_bars_per_day=24.0)
    result = compute_triple_barrier_labels(events, bars)
    assert len(result) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_labeling.py::test_entry_close_zero_skipped_gracefully -v`
Expected: FAIL with "ZeroDivisionError"

- [ ] **Step 3: Write minimal implementation**

Open `src/trading_databricks/ml/labeling.py`. Find the loop where `entry_close = float(event["close"])` is extracted (around line 95). Add the following guard immediately after:

```python
        entry_close = float(event["close"])
        if entry_close <= 0.0:
            logger.warning("Skipping event %s: entry_close=%.6f <= 0", event["event_id"], entry_close)
            continue
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_labeling.py::test_entry_close_zero_skipped_gracefully -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_labeling.py src/trading_databricks/ml/labeling.py
git commit -m "fix(ml): prevent division by zero when entry_close is 0 in labeling"
```

---

### Task 2: Fix concurrency exit bar

**Files:**
- Modify: `tests/ml/test_labeling.py`
- Modify: `src/trading_databricks/ml/labeling.py`

- [ ] **Step 1: Write the failing test**

Open `tests/ml/test_labeling.py` and add the following test function:

```python
def test_avg_uniqueness_uses_actual_exit_bar() -> None:
    """Events that hit upper/lower barrier early should have higher avg_uniqueness than if they used vertical_bar."""
    bars_list = [100.0, 101.0, 110.0] + [100.0] * 20
    bars = _make_bars(bars_list)
    events = _make_events([0], bars, ewma_vol=0.05, avg_bars_per_day=24.0)
    import polars as pl
    events = events.with_columns(pl.lit(20).alias("vertical_bar"))

    result = compute_triple_barrier_labels(events, bars)

    assert len(result) == 1
    assert result["avg_uniqueness"][0] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_labeling.py::test_avg_uniqueness_uses_actual_exit_bar -v`
Expected: FAIL because uniqueness uses vertical_bar (20) and overlaps incorrectly.

- [ ] **Step 3: Write minimal implementation**

Open `src/trading_databricks/ml/labeling.py`. 

First, inside the `for event in events_dicts:` loop, initialize `exit_bar`:
```python
        exit_bar = vertical_bar
```

Next, update the barrier hit conditions to record the actual `exit_bar`:
```python
            if c >= upper:
                label = 1
                barrier_hit = "upper"
                found = True
                exit_bar = bi
                break
            elif c <= lower:
                label = -1
                barrier_hit = "lower"
                found = True
                exit_bar = bi
                break
```

Next, add `_exit_bar` to the dictionary appended to `labeled`:
```python
        labeled.append({
            "event_id": event["event_id"],
            "bar_index": event["bar_index"],
            "label": label,
            "barrier_hit": barrier_hit,
            "return_pct": return_pct,
            "_bar_index": event["bar_index"],
            "_exit_bar": exit_bar,
            "vertical_bar": vertical_bar
        })
```

Next, inside `_fill_avg_uniqueness`, change the `ends` calculation to use `_exit_bar`:
```python
    starts = np.array([row["_bar_index"] for row in labeled], dtype=int)
    ends = np.array([row["_exit_bar"] for row in labeled], dtype=int)
```

Finally, strip `_exit_bar` before returning:
```python
    return [{k: v for k, v in row.items() if k not in ("_bar_index", "_exit_bar")} for row in labeled]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_labeling.py::test_avg_uniqueness_uses_actual_exit_bar -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_labeling.py src/trading_databricks/ml/labeling.py
git commit -m "fix(ml): use actual exit bar instead of vertical bar for concurrency span"
```

---

### Task 3: Fix `ewma_vol=0` collapsing barriers

**Files:**
- Modify: `tests/ml/test_labeling.py`
- Modify: `src/trading_databricks/ml/labeling.py`

- [ ] **Step 1: Write the failing test**

Open `tests/ml/test_labeling.py` and add the following test function:

```python
def test_zero_ewma_vol_does_not_always_label_positive() -> None:
    """Events with ewma_vol=0 should be skipped or labeled 0, not always 1."""
    bars = _make_bars([100.0, 101.0, 99.0, 100.0, 100.0])
    events = _make_events([0], bars, ewma_vol=0.0, avg_bars_per_day=24.0)
    result = compute_triple_barrier_labels(events, bars)
    if len(result) > 0:
        assert result["label"][0] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_labeling.py::test_zero_ewma_vol_does_not_always_label_positive -v`
Expected: FAIL because ewma_vol=0 collapses barriers resulting in a false label of 1.

- [ ] **Step 3: Write minimal implementation**

Open `src/trading_databricks/ml/labeling.py`. Find where `upper` and `lower` are extracted (around line 93-94) and add a guard:

```python
        upper = float(event["upper_level"])
        lower = float(event["lower_level"])
        
        if upper <= lower:
            logger.warning("Skipping event %s: degenerate barriers (upper=%.6f <= lower=%.6f)",
                            event["event_id"], upper, lower)
            continue
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_labeling.py::test_zero_ewma_vol_does_not_always_label_positive -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_labeling.py src/trading_databricks/ml/labeling.py
git commit -m "fix(ml): skip events with degenerate barriers when ewma_vol is 0"
```

---

### Task 4: Add null close guard after left join

**Files:**
- Modify: `tests/ml/test_labeling.py`
- Modify: `src/trading_databricks/ml/labeling.py`

- [ ] **Step 1: Write the failing test**

Open `tests/ml/test_labeling.py` and add the following test function:

```python
def test_event_with_nonexistent_bar_index_skipped() -> None:
    """Events whose bar_index doesn't exist in bars should be skipped."""
    bars = _make_bars([100.0, 101.0, 102.0])
    events = _make_events([999], bars, ewma_vol=0.02, avg_bars_per_day=24.0)
    result = compute_triple_barrier_labels(events, bars)
    assert len(result) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_labeling.py::test_event_with_nonexistent_bar_index_skipped -v`
Expected: FAIL with TypeError or KeyError due to null close value after a failed left join.

- [ ] **Step 3: Write minimal implementation**

Open `src/trading_databricks/ml/labeling.py`. Locate the join operation where `events` is joined with `bars` (around line 63) and append a filter to drop null close values:

```python
    events_with_close = events.join(
        bars.select(["bar_index", "close"]), on="bar_index", how="left"
    ).filter(pl.col("close").is_not_null())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_labeling.py::test_event_with_nonexistent_bar_index_skipped -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_labeling.py src/trading_databricks/ml/labeling.py
git commit -m "fix(ml): guard against null close values resulting from left joins"
```

---

### Task 5: Final verification

- [ ] Run full test suite: `uv run pytest tests/ml/test_labeling.py -v`
- [ ] Run overall tests: `uv run pytest`
- [ ] Run linter: `uv run ruff check src/trading_databricks/ml/labeling.py`
