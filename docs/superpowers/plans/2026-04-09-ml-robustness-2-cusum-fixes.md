# ML Robustness Plan 2/8: CUSUM Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three silent data corruption bugs in the CUSUM event detection pipeline to ensure robustness against zeroes, offset indices, and missing thresholds.

**Architecture:** The approach is to add precise edge-case handling into the existing Polars dataframe transformations (`cusum.py`). We will guard `log(0)` with a `.when().then()` expression, correct `avg_bars_per_day` to use sequence row numbers (`pl.int_range()`) instead of the raw `bar_index`, and introduce an explicit `None` and `NaN` loop guard for the threshold variable. This adheres to rigorous AFML event-sampling requirements.

**Tech Stack:** Python, Polars, Pytest

---

### Task 1: Fix log(0) producing -inf

**Files:**
- Modify: `src/trading_databricks/ml/cusum.py`
- Modify: `tests/ml/test_cusum.py`

- [ ] **Step 1: Write the failing test**

```python
def test_close_zero_does_not_corrupt_pipeline() -> None:
    """A single close=0 bar must not produce NaN in any output column."""
    closes = [100.0] * 200
    bars = _make_bars(closes, hours_per_bar=4.5)
    # Inject a zero close
    bars = bars.with_columns(
        pl.when(pl.col("bar_index") == 50)
        .then(pl.lit(0.0))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    result = detect_cusum_events(bars)
    # Should either skip the bad bar or return empty, not produce NaN in the output
    assert result.select(pl.all().is_nan().any()).sum_horizontal().item() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_cusum.py::test_close_zero_does_not_corrupt_pipeline -v`
Expected: FAIL or ERROR

- [ ] **Step 3: Write minimal implementation**

Modify `src/trading_databricks/ml/cusum.py`:
Replace lines ~65-72 (`# --- Step 1: Vectorized log returns ---` block):
```python
    # --- Step 1: Vectorized log returns ---
    bars = bars.with_columns(
        pl.col("close").log().diff().alias("log_return"),
    )
```
With:
```python
    # --- Step 1: Vectorized log returns ---
    bars = bars.with_columns(
        pl.when(pl.col("close") > 0)
        .then(pl.col("close").log())
        .otherwise(None)
        .diff()
        .alias("log_return"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_cusum.py::test_close_zero_does_not_corrupt_pipeline -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_cusum.py src/trading_databricks/ml/cusum.py
git commit -m "fix(ml): prevent log(0) from corrupting cusum accumulators"
```

### Task 2: Fix avg_bars_per_day using raw bar_index

**Files:**
- Modify: `src/trading_databricks/ml/cusum.py`
- Modify: `tests/ml/test_cusum.py`

- [ ] **Step 1: Write the failing test**

```python
def test_nonzero_based_bar_index_produces_same_events() -> None:
    """bar_index starting at 50000 should produce same events as starting at 0."""
    closes = [100.0] * 100 + [110.0] * 100
    bars_zero = _make_bars(closes, hours_per_bar=4.5)
    bars_offset = bars_zero.with_columns(
        (pl.col("bar_index") + 50000).alias("bar_index")
    )
    events_zero = detect_cusum_events(bars_zero)
    events_offset = detect_cusum_events(bars_offset)
    assert len(events_zero) == len(events_offset)
    assert len(events_zero) > 0  # ensure we're actually testing something
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_cusum.py::test_nonzero_based_bar_index_produces_same_events -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Modify `src/trading_databricks/ml/cusum.py`:
Replace lines ~100-110 (`# Rolling calculation...` block):
```python
    # Rolling calculation: for each bar, count bars as min(bar_index, lookback_size)
    bars = bars.with_columns(
        (
            pl.min_horizontal(
                pl.col("bar_index").cast(pl.Float64),
                pl.lit(float(lookback_size)),
            )
            / pl.col("_days_since_start")
        ).alias("avg_bars_per_day"),
    )
```
With:
```python
    # Rolling calculation: for each bar, count bars as min(row_number, lookback_size)
    bars = bars.with_columns(
        (
            pl.min_horizontal(
                (pl.int_range(pl.len()) + 1).cast(pl.Float64),
                pl.lit(float(lookback_size)),
            )
            / pl.col("_days_since_start")
        ).alias("avg_bars_per_day"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_cusum.py::test_nonzero_based_bar_index_produces_same_events -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_cusum.py src/trading_databricks/ml/cusum.py
git commit -m "fix(ml): use row count instead of bar_index for avg_bars_per_day"
```

### Task 3: Fix missing threshold guard

**Files:**
- Modify: `src/trading_databricks/ml/cusum.py`
- Modify: `tests/ml/test_cusum.py`

- [ ] **Step 1: Write the failing test**

```python
def test_cusum_does_not_crash_on_degenerate_threshold() -> None:
    """Bars that produce NaN/None threshold must not crash."""
    closes = [100.0] * 200
    bars = _make_bars(closes, hours_per_bar=4.5)
    
    # Force threshold to be NaN for a specific row to test the loop resilience
    bars = bars.with_columns(
        pl.when(pl.col("bar_index") == 100)
        .then(pl.lit(float("nan")))
        .otherwise(pl.col("ewma_vol"))
        .alias("ewma_vol")
    ).with_columns(
        (
            CUSUMConfig().threshold_multiplier
            * pl.col("ewma_vol")
            * pl.col("avg_bars_per_day").sqrt()
        ).alias("threshold")
    )
    
    # Should handle this gracefully and not crash
    result = detect_cusum_events(bars)
    assert isinstance(result, pl.DataFrame)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_cusum.py::test_cusum_does_not_crash_on_degenerate_threshold -v`
Expected: FAIL or ERROR

- [ ] **Step 3: Write minimal implementation**

Modify `src/trading_databricks/ml/cusum.py`:
Replace lines ~125-135 (inside `for row in bars.iter_rows` block):
```python
        threshold = row["threshold"]
        s_pos = max(0.0, s_pos + lr)
        s_neg = min(0.0, s_neg + lr)
```
With:
```python
        threshold = row["threshold"]
        if threshold is None or (isinstance(threshold, float) and threshold != threshold):
            continue
            
        s_pos = max(0.0, s_pos + lr)
        s_neg = min(0.0, s_neg + lr)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_cusum.py::test_cusum_does_not_crash_on_degenerate_threshold -v`
Run all tests: `uv run pytest tests/ml/test_cusum.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_cusum.py src/trading_databricks/ml/cusum.py
git commit -m "fix(ml): add robust NaN and None guard for CUSUM threshold"
```
