# ML Robustness Plan 1/8: Input Validation Framework

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
>
> **Parallelism:** This plan is independent of plans 2-6. Execute concurrently.
> **Depends on:** Nothing (first wave).
> **Blocks:** Plan 7 (sklearn transformers).

**Goal:** Create a validation module that enforces data contracts at stage boundaries. Currently, the pipeline has ZERO validation between stages — left joins silently produce NULLs, missing columns crash with cryptic KeyErrors, and corrupt data (close=0, null timestamps) propagates silently.

**Tech Stack:** Polars only. No Spark, no pandas.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `src/trading_databricks/ml/validation.py` | CREATE | Stage-boundary validation functions |
| `tests/ml/test_validation.py` | CREATE | Tests for all validators |

Reads from: `ml/schemas.py` (column name contracts)

---

## Task 1: `validate_bars()`

**Files:** `src/trading_databricks/ml/validation.py`, `tests/ml/test_validation.py`

- [ ] **Step 1: Write failing tests**
  - Test valid bars pass silently
  - Test missing `close` column raises `ValueError`
  - Test `close <= 0` raises `ValueError` (identifies the offending rows)
  - Test `bar_end_time` with nulls raises `ValueError`
  - Test non-monotonic `bar_index` raises `ValueError`
  - Test duplicate `bar_index` raises `ValueError`

- [ ] **Step 2: Implement `validate_bars(bars: pl.DataFrame) -> None`**
  ```python
  def validate_bars(bars: pl.DataFrame) -> None:
      """Assert bars meet CUSUM/feature pipeline contracts.
      Raises ValueError with descriptive message on failure."""
  ```
  Required columns: `bar_index`, `close`, `bar_end_time`, `symbol`
  Checks:
  - All required columns exist
  - `close > 0` (all rows)
  - `bar_end_time` non-null (all rows)
  - `bar_index` strictly monotonically increasing after sort (no gaps allowed for PIT safety)
  - Report: which rows fail, first 5 offending values

- [ ] **Step 3: Run tests, verify green**

---

## Task 2: `validate_events()`

- [ ] **Step 1: Write failing tests**
  - Valid events pass
  - Missing `event_id` raises
  - `ewma_vol <= 0` raises (identifies rows)
  - `cusum_side` not in `{'up', 'down'}` raises
  - `threshold_used <= 0` or null raises
  - `event_ts` null raises

- [ ] **Step 2: Implement `validate_events(events: pl.DataFrame) -> None`**
  Required columns: `event_id`, `symbol`, `event_ts`, `bar_index`, `log_return`, `cusum_side`, `cusum_value`, `threshold_used`, `ewma_vol`

- [ ] **Step 3: Run tests, verify green**

---

## Task 3: `validate_labels()`

- [ ] **Step 1: Write failing tests**
  - Valid labels pass
  - `label` not in `{-1, 0, 1}` raises
  - `barrier_hit` not in `{'upper', 'lower', 'vertical'}` raises
  - `avg_uniqueness <= 0` or `> 1` raises
  - `avg_uniqueness` NaN raises

- [ ] **Step 2: Implement `validate_labels(labels: pl.DataFrame) -> None`**
  Required columns: from `_LABELS_SCHEMA` keys in `labeling.py`

- [ ] **Step 3: Run tests, verify green**

---

## Task 4: `validate_training_dataset()`

- [ ] **Step 1: Write failing tests**
  - Valid dataset passes
  - All-null feature column raises (indicates upstream stage failure masked by left join)
  - `label` column null raises
  - `cpcv_group` outside `1..n_groups` raises
  - Empty DataFrame raises

- [ ] **Step 2: Implement `validate_training_dataset(df: pl.DataFrame, *, n_groups: int = 6) -> None`**

- [ ] **Step 3: Run tests, verify green**

---

## Task 5: Final verification

- [ ] Run `uv run pytest tests/ml/test_validation.py -v` — all pass
- [ ] Run `uv run pytest` — all 48+ tests pass (no regressions)
- [ ] Run `uv run ruff check src/trading_databricks/ml/validation.py`
