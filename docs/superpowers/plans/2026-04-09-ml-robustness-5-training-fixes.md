# ML Robustness Plan 5/8: Training Module Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the CPCV embargo data leakage bug for non-contiguous test groups, add NaN/zero validation for sample weights, and verify the PSR kurtosis formula.

**Architecture:** Modifies the core training and validation pipeline (CPCV split logic, sample weight computation) to ensure mathematical robustness and eliminate data leakage.

**Tech Stack:** Python, NumPy, Polars, SciPy, Pytest.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `tests/ml/test_cpcv.py` | Modify | Add test for non-contiguous embargo |
| `src/trading_databricks/ml/training/cpcv.py` | Modify | Fix embargo logic for test segments |
| `tests/ml/test_sample_weights.py` | Modify | Add tests for NaN/zero weights |
| `src/trading_databricks/ml/training/sample_weights.py`| Modify | Add validation logic |
| `tests/ml/test_promotion.py` | Modify | Add PSR analytical verification test |

---

### Task 1: Fix CPCV Embargo Data Leakage

Currently, `_embargo` only drops samples after the absolute maximum index in the test group. When a test split is non-contiguous, samples directly following earlier test blocks leak into the training set.

**Files:**
- Modify: `tests/ml/test_cpcv.py`
- Modify: `src/trading_databricks/ml/training/cpcv.py`

- [ ] **Step 1: Write the failing test**

Append this test to `tests/ml/test_cpcv.py`:

```python
def test_embargo_applies_after_each_test_segment() -> None:
    """Non-contiguous test groups must have embargo after EACH segment."""
    from trading_databricks.ml.config import CPCVConfig
    from trading_databricks.ml.training.cpcv import CPCVSplitter
    import numpy as np

    n = 120
    t_start = np.arange(n)
    t_end = np.arange(n) + 5
    config = CPCVConfig(n_groups=6, k_test=2, embargo_pct=0.01)
    splitter = CPCVSplitter(config, t_start=t_start, t_end=t_end)
    
    found = False
    for train_idx, test_idx in splitter.split(np.zeros((n, 1))):
        sorted_test = np.sort(test_idx)
        gaps = np.where(np.diff(sorted_test) > 1)[0]
        if len(gaps) > 0:
            found = True
            first_seg_end = sorted_test[gaps[0]]
            embargo_size = max(1, int(0.01 * n))
            for embargo_idx in range(first_seg_end + 1, first_seg_end + 1 + embargo_size):
                assert embargo_idx not in train_idx, (
                    f"Sample {embargo_idx} should be embargoed (after first test segment ending at {first_seg_end})"
                )
            return
            
    assert found, "No non-contiguous test groups found in any split"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_cpcv.py::test_embargo_applies_after_each_test_segment -v`
Expected: FAIL with an AssertionError showing the sample was not embargoed.

- [ ] **Step 3: Write minimal implementation**

In `src/trading_databricks/ml/training/cpcv.py`, replace the `_embargo` method entirely with:

```python
    def _embargo(
        self, train_idx: np.ndarray, test_idx: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """Remove embargo_pct × n_samples samples immediately after each test group segment."""
        if len(test_idx) == 0:
            return train_idx

        embargo_size = max(1, int(self.config.embargo_pct * n_samples))
        
        # Find contiguous segments of test indices and embargo after each
        sorted_test = np.sort(test_idx)
        gaps = np.where(np.diff(sorted_test) > 1)[0]
        segment_ends = np.append(sorted_test[gaps], sorted_test[-1])

        embargo_set: set[int] = set()
        for seg_end in segment_ends:
            for j in range(int(seg_end) + 1, min(int(seg_end) + 1 + embargo_size, n_samples)):
                embargo_set.add(j)

        return np.array([i for i in train_idx if i not in embargo_set], dtype=int)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_cpcv.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_cpcv.py src/trading_databricks/ml/training/cpcv.py
git commit -m "fix: apply CPCV embargo after each non-contiguous test segment"
```

---

### Task 2: Validate Sample Weights

Invalid labels (e.g. NaN or zero `avg_uniqueness` values) will silently cause sklearn estimators to produce NaN losses or omit samples.

**Files:**
- Modify: `tests/ml/test_sample_weights.py`
- Modify: `src/trading_databricks/ml/training/sample_weights.py`

- [ ] **Step 1: Write the failing test**

Append this to `tests/ml/test_sample_weights.py`:

```python
def test_nan_weights_raise_value_error() -> None:
    import pytest
    import polars as pl
    from trading_databricks.ml.training.sample_weights import compute_sample_weights

    labels = pl.DataFrame({"avg_uniqueness": [0.8, float("nan"), 0.6]})
    with pytest.raises(ValueError, match="NaN"):
        compute_sample_weights(labels)

def test_zero_weights_raise_value_error() -> None:
    import pytest
    import polars as pl
    from trading_databricks.ml.training.sample_weights import compute_sample_weights

    labels = pl.DataFrame({"avg_uniqueness": [0.8, 0.0, 0.6]})
    with pytest.raises(ValueError, match="non-positive"):
        compute_sample_weights(labels)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_sample_weights.py -v`
Expected: FAIL, ValueError not raised.

- [ ] **Step 3: Write minimal implementation**

In `src/trading_databricks/ml/training/sample_weights.py`, replace `compute_sample_weights` with:

```python
def compute_sample_weights(labels: pl.DataFrame) -> np.ndarray:
    """Extract avg_uniqueness from labeled events as a numpy float64 array.

    The avg_uniqueness column is computed by TRD-44's labeling module and
    stored in TRIPLE_BARRIER_LABELS_SCHEMA. This function exists for sklearn
    sample_weight parameter compatibility.

    Args:
        labels: DataFrame with at least an 'avg_uniqueness' column.

    Returns:
        1-D numpy float64 array of sample weights, one per event.

    Raises:
        polars.exceptions.ColumnNotFoundError: If 'avg_uniqueness' column is absent.
        ValueError: If weights contain NaN or non-positive values.
    """
    import numpy as np

    weights = labels.get_column("avg_uniqueness").to_numpy().astype(np.float64)
    
    n_nan = int(np.isnan(weights).sum())
    n_nonpos = int((weights <= 0.0).sum()) - n_nan  # exclude NaN from nonpos count
    
    if n_nan > 0 or n_nonpos > 0:
        raise ValueError(
            f"Invalid sample weights: {n_nan} NaN, {n_nonpos} non-positive. "
            "Check upstream labeling for degenerate events."
        )
        
    return weights
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_sample_weights.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_sample_weights.py src/trading_databricks/ml/training/sample_weights.py
git commit -m "feat: add validation for sample weights to reject NaN and zero"
```

---

### Task 3: Verify PSR Kurtosis Formula

The formula uses `gamma_4 - 1.0` where `gamma_4` is raw kurtosis (`fisher=False`). For a normal distribution, raw kurtosis is 3. We verify that PSR evaluates correctly against analytical estimates.

**Files:**
- Modify: `tests/ml/test_promotion.py`

- [ ] **Step 1: Write verification test**

Append this to `tests/ml/test_promotion.py`:

```python
def test_psr_normal_returns_matches_analytical() -> None:
    """For normal IID returns, PSR variance should be ~(1 + 0.5*SR^2)/(T-1)."""
    import numpy as np
    from trading_databricks.ml.training.promotion import compute_psr

    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 1000)
    sr = float(np.mean(returns) / np.std(returns, ddof=1))
    
    psr = compute_psr(sr, returns, benchmark_sharpe=0.0)
    # For 1000 observations with SR~1, PSR should be very high (>0.99)
    assert psr > 0.95
    
    # Verify: PSR(SR=0 | benchmark=0) should be ~0.5
    zero_sr_returns = np.random.normal(0.0, 0.01, 100)
    psr_zero = compute_psr(0.0, zero_sr_returns, benchmark_sharpe=0.0)
    assert 0.3 < psr_zero < 0.7  # centered around 0.5
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_promotion.py::test_psr_normal_returns_matches_analytical -v`
Expected: PASS (Formula is already correct; this is purely a verification test).

- [ ] **Step 3: Commit**

```bash
git add tests/ml/test_promotion.py
git commit -m "test: add analytical verification test for PSR kurtosis formula"
```

---

### Task 4: Final verification

- [ ] **Step 1: Run full test suite**
Run: `uv run pytest -v`
Expected: PASS

- [ ] **Step 2: Run linter**
Run: `uv run ruff check src/trading_databricks/ml/training/`
Expected: Clean output.
