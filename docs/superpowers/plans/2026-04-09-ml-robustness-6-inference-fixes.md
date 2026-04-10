# ML Robustness Plan 6/8: Inference Module Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the confidence leak on flat positions and add NaN/Null prediction observability to the inference pipeline.

**Architecture:** Enforce strict zero-confidence constraints on flat positions produced from short signals (in Phase 1A long-only mode). Add an explicit validation step before the threshold filters in `bet_sizing.py` to count, log, and filter out any IEEE NaNs or Nulls in the confidence columns, alerting us to upstream model degradation.

**Tech Stack:** Python, Polars, Pytest

---

### Task 1: Fix confidence on flat positions

**Files:**
- Modify: `tests/ml/test_position.py`
- Modify: `src/trading_databricks/ml/inference/position.py`

- [ ] **Step 1: Write the failing test**
In `tests/ml/test_position.py`, modify the existing `test_long_only_constraint_short_signal_becomes_flat` function to assert that both weight and confidence are precisely 0.0:

```python
def test_long_only_constraint_short_signal_becomes_flat() -> None:
    """Phase 1A: primary_side=-1 → position side='flat', weight=0.0, confidence=0.0."""
    signals = _make_signals([0.85], [-1], [0.85])
    result = compute_positions(signals)

    assert result["side"][0] == "flat"
    assert result["weight"][0] == 0.0
    assert result["confidence"][0] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_position.py::test_long_only_constraint_short_signal_becomes_flat -v`
Expected: FAIL with an assertion error like `assert 0.85 == 0.0`.

- [ ] **Step 3: Write minimal implementation**
In `src/trading_databricks/ml/inference/position.py`, locate the return statement at the end of the `compute_positions` function and update the `"confidence"` key to apply the `if is_long else 0.0` condition:

```python
    return pl.DataFrame(
        {
            "symbol": [row["symbol"]],
            "side": [side],
            "weight": [float(row["discretized_weight"]) if is_long else 0.0],
            "confidence": [float(row["meta_confidence"]) if is_long else 0.0],
            "signal_id": [str(row["signal_id"])],
            "updated_at": [now],
        }
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_position.py::test_long_only_constraint_short_signal_becomes_flat -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_position.py src/trading_databricks/ml/inference/position.py
git commit -m "fix(inference): enforce zero confidence on flat positions"
```

---

### Task 2: Add NaN/Null prediction logging

**Files:**
- Modify: `tests/ml/test_bet_sizing.py`
- Modify: `src/trading_databricks/ml/inference/bet_sizing.py`

- [ ] **Step 1: Write the failing test**
In `tests/ml/test_bet_sizing.py`, add `import logging` to the top of the file (under `from datetime import datetime`), and then append this new test function at the bottom:

```python
import logging

def test_nan_or_null_confidence_logged_and_dropped(caplog) -> None:
    """NaN or Null predictions should be logged as warnings and excluded."""
    preds = pl.DataFrame(
        {
            "signal_id": ["sig-0", "sig-1", "sig-2"],
            "event_id": ["evt-0", "evt-1", "evt-2"],
            "symbol": ["BTCUSDT"] * 3,
            "event_ts": [datetime(2025, 1, 1)] * 3,
            "primary_side": [1, 1, 1],
            "primary_confidence": [0.70, float("nan"), 0.80],
            "meta_label": [1] * 3,
            "meta_confidence": [0.80, 0.90, None],
            "primary_model_version": [1] * 3,
            "meta_model_version": [1] * 3,
        }
    )
    with caplog.at_level(logging.WARNING):
        result = compute_bet_sizes(preds, sigma_probability=0.10)

    # Only sig-0 should pass (both confidences are valid and > thresholds)
    assert len(result) == 1
    assert result["signal_id"][0] == "sig-0"
    assert "Dropping 2 predictions with NaN/Null confidence values" in caplog.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_bet_sizing.py::test_nan_or_null_confidence_logged_and_dropped -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**
In `src/trading_databricks/ml/inference/bet_sizing.py`:

First, add the logging imports under `from scipy.stats import norm`:
```python
import logging

logger = logging.getLogger(__name__)
```

Then, replace the `filtered = predictions.filter(...)` block in `compute_bet_sizes` (around lines 85-88) with the following explicit validation step:

```python
    # Step 1a: Check for NaN or Null confidences and drop them with a warning
    nan_mask = (
        pl.col("meta_confidence").is_nan() | pl.col("meta_confidence").is_null() |
        pl.col("primary_confidence").is_nan() | pl.col("primary_confidence").is_null()
    )
    nan_count = predictions.select(nan_mask.sum()).item()
    if nan_count > 0:
        logger.warning(
            "Dropping %d predictions with NaN/Null confidence values (possible upstream model failure)",
            nan_count,
        )
        predictions = predictions.filter(~nan_mask)

    # Step 1b: Filter by both thresholds (strictly greater than)
    filtered = predictions.filter(
        (pl.col("meta_confidence") > config.meta_threshold)
        & (pl.col("primary_confidence") > config.primary_threshold)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_bet_sizing.py::test_nan_or_null_confidence_logged_and_dropped -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_bet_sizing.py src/trading_databricks/ml/inference/bet_sizing.py
git commit -m "feat(inference): log and drop NaN/Null predictions before bet sizing"
```

---

### Task 3: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest`
Expected: All tests pass.

- [ ] **Step 2: Run linter**

Run: `uv run ruff check src/trading_databricks/ml/inference/ tests/ml/`
Expected: No errors. Fix any unused imports or formatting issues if present.
