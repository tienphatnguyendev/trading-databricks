# ML Robustness Plan 4/8: Feature Engineering Bug Fixes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix numerical robustness bugs in volatility, technical, and news feature extraction that cause silent NaN propagation and data leakage.

**Architecture:** We will patch the Polars computational graphs in the feature modules to properly handle edge cases: zero or negative bounds inside square roots, log(0), short-window EMAs, and incorrect filter ordering for news ratios.

**Tech Stack:** Python, Polars, Pytest

---

### Task 1: Fix Garman-Klass `sqrt()` of negative value

**Files:**
- Modify: `src/trading_databricks/ml/features/volatility.py`
- Modify: `tests/ml/test_features_volatility.py`

- [ ] **Step 1: Write the failing test**

Open `tests/ml/test_features_volatility.py` and append this test case:

```python
import polars as pl
import pytest
from datetime import datetime, UTC
from trading_databricks.ml.features.volatility import compute_volatility

def test_garman_klass_negative_variance_handling():
    """Verify that Garman-Klass does not yield NaN when close-to-open variance is disproportionately large."""
    # Create 20 bars with small high/low ranges but huge close/open jumps
    bars = pl.DataFrame({
        "bar_index": list(range(20)),
        "open": [100.0] * 20,
        "high": [110.0] * 20,
        "low": [100.0] * 20,
        "close": [110.0] * 20,
    }).lazy()

    events = pl.DataFrame({
        "event_id": ["e1"],
        "symbol": ["BTC"],
        "event_ts": [datetime(2024, 1, 1, tzinfo=UTC)],
        "bar_index": [19],
        "avg_bars_per_day": [24.0]
    }).lazy()

    result = compute_volatility(events, bars).collect()
    
    # Before the fix, garman_klass_20 would be NaN
    gk = result["garman_klass_20"][0]
    assert gk is not None
    assert not pl.Series([gk]).is_nan()[0]
    assert gk >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_features_volatility.py::test_garman_klass_negative_variance_handling -v`
Expected: FAIL (AssertionError: assert False where False = is_nan)

- [ ] **Step 3: Write minimal implementation**

In `src/trading_databricks/ml/features/volatility.py`, update the Garman-Klass feature expression (around line 53) to clip the lower bound before the square root:

```python
        # Garman-Klass: sqrt(rolling_mean(0.5*ln(H/L)^2 - (2*ln2-1)*ln(C/O)^2))
        (
            pl.col("_log_hl_half")
            - pl.col("_log_co_sq").mul(2 * ln2 - 1)
        )
        .rolling_mean(20)
        .clip(lower_bound=0.0)
        .sqrt()
        .alias("garman_klass_20"),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_features_volatility.py::test_garman_klass_negative_variance_handling -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_features_volatility.py src/trading_databricks/ml/features/volatility.py
git commit -m "fix(ml): prevent NaN in Garman-Klass by clipping negative variance estimates"
```

---

### Task 2: Fix `log(0)` in technical features

**Files:**
- Modify: `src/trading_databricks/ml/features/technical.py`
- Modify: `tests/ml/test_features_technical.py`

- [ ] **Step 1: Write the failing test**

Open `tests/ml/test_features_technical.py` and append:

```python
import polars as pl
from datetime import datetime, UTC
from trading_databricks.ml.features.technical import compute_technical

def test_technical_log_zero_handling():
    """Verify that log(0) does not produce NaN/inf for log returns."""
    # Create bars where close drops to 0
    bars = pl.DataFrame({
        "bar_index": list(range(25)),
        "open": [10.0] * 25,
        "high": [10.0] * 25,
        "low": [10.0] * 25,
        "close": [10.0] * 24 + [0.0],  # Last bar is 0
    }).lazy()

    events = pl.DataFrame({
        "event_id": ["e1"],
        "symbol": ["BTC"],
        "event_ts": [datetime(2024, 1, 1, tzinfo=UTC)],
        "bar_index": [24],
    }).lazy()

    result = compute_technical(events, bars).collect()
    
    # log_return_5 should be null, not NaN or -inf
    assert result["log_return_5"][0] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_features_technical.py::test_technical_log_zero_handling -v`
Expected: FAIL (AssertionError: log_return_5 is NaN or -inf)

- [ ] **Step 3: Write minimal implementation**

In `src/trading_databricks/ml/features/technical.py`, update `_log_ret` and the `log_return` columns (lines ~29 and ~59) to use a safe log wrap:

Replace:
```python
        # Log returns (used for log_return_5, log_return_20 below)
        pl.col("close").log().diff().alias("_log_ret"),
```
With:
```python
        # Log returns (used for log_return_5, log_return_20 below)
        pl.when(pl.col("close") > 0).then(pl.col("close").log()).otherwise(None).diff().alias("_log_ret"),
```

And replace `log_return_5` / `log_return_20`:
```python
        # Log returns over N bars
        pl.col("close").log().diff(5).alias("log_return_5"),
        pl.col("close").log().diff(20).alias("log_return_20"),
```
With:
```python
        # Log returns over N bars
        pl.when(pl.col("close") > 0).then(pl.col("close").log()).otherwise(None).diff(5).alias("log_return_5"),
        pl.when(pl.col("close") > 0).then(pl.col("close").log()).otherwise(None).diff(20).alias("log_return_20"),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_features_technical.py::test_technical_log_zero_handling -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_features_technical.py src/trading_databricks/ml/features/technical.py
git commit -m "fix(ml): prevent inf/NaN in technical features when close is zero"
```

---

### Task 3: Fix MACD `min_periods`

**Files:**
- Modify: `src/trading_databricks/ml/features/technical.py`
- Modify: `tests/ml/test_features_technical.py`

- [ ] **Step 1: Write the failing test**

Open `tests/ml/test_features_technical.py` and append:

```python
def test_technical_macd_min_periods():
    """Verify MACD is null for early bars to prevent unstable EMA signals."""
    bars = pl.DataFrame({
        "bar_index": list(range(10)),  # Only 10 bars (less than 26)
        "open": [10.0] * 10,
        "high": [11.0] * 10,
        "low": [9.0] * 10,
        "close": [10.5] * 10,
    }).lazy()

    events = pl.DataFrame({
        "event_id": ["e1"],
        "symbol": ["BTC"],
        "event_ts": [datetime(2024, 1, 1, tzinfo=UTC)],
        "bar_index": [9],
    }).lazy()

    result = compute_technical(events, bars).collect()
    
    # macd_signal should be null because we haven't reached min_periods=26
    assert result["macd_signal"][0] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_features_technical.py::test_technical_macd_min_periods -v`
Expected: FAIL (AssertionError: macd_signal is a float, not None)

- [ ] **Step 3: Write minimal implementation**

In `src/trading_databricks/ml/features/technical.py`, update the MACD computation (around line 43):

```python
        # MACD signal (EMA12 - EMA26, then EMA9 of that)
        (
            pl.col("close").ewm_mean(span=12, min_periods=12)
            - pl.col("close").ewm_mean(span=26, min_periods=26)
        )
        .ewm_mean(span=9, min_periods=9)
        .alias("macd_signal"),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_features_technical.py::test_technical_macd_min_periods -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_features_technical.py src/trading_databricks/ml/features/technical.py
git commit -m "fix(ml): enforce min_periods on MACD EMAs for stable feature generation"
```

---

### Task 4: Fix `news_accepted_ratio` logic

**Files:**
- Modify: `src/trading_databricks/ml/features/news.py`
- Modify: `tests/ml/test_features_news.py`

- [ ] **Step 1: Write the failing test**

Open `tests/ml/test_features_news.py` and append:

```python
import polars as pl
from datetime import datetime, UTC
from trading_databricks.ml.features.news import compute_news

def test_news_accepted_ratio():
    """Verify accepted ratio is computed over all impactful news, not just accepted news."""
    events = pl.DataFrame({
        "event_id": ["e1"],
        "symbol": ["BTC"],
        "event_ts": [datetime(2024, 1, 2, tzinfo=UTC)],
    }).lazy()

    news_features = pl.DataFrame({
        "published_at": [datetime(2024, 1, 1, 12, tzinfo=UTC)] * 4,
        "is_impactful": [True, True, True, True],
        "validation_status": ["accepted", "accepted", "rejected", "rejected"],
        "l1_category": ["Sentiment"] * 4,
        "confidence": [0.9, 0.8, 0.5, 0.4],
        "expected_direction": [1, -1, 0, 0]
    }).lazy()

    result = compute_news(events, news_features).collect()
    
    # Total impactful news = 4. Accepted = 2. Ratio = 0.5.
    assert result["news_accepted_ratio_Sentiment"][0] == 0.5
    # Only the 2 accepted news count for the features
    assert result["news_count_Sentiment"][0] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_features_news.py::test_news_accepted_ratio -v`
Expected: FAIL (AssertionError: ratio is 1.0 instead of 0.5)

- [ ] **Step 3: Write minimal implementation**

In `src/trading_databricks/ml/features/news.py`:

Replace lines ~75-77:
```python
    news_df = news_features.filter(
        (pl.col("is_impactful")) & (pl.col("validation_status") == "accepted")
    ).collect()
```
With:
```python
    news_df = news_features.filter(pl.col("is_impactful")).collect()
```

Then, inside the `for l1 in _L1_CATEGORIES:` loop (lines ~98-115), update the aggregation to calculate the ratio on `all_cat` and the other metrics on `cat`:

```python
        for l1 in _L1_CATEGORIES:
            all_cat = window.filter(pl.col("l1_category") == l1)
            total_cat_cnt = len(all_cat)
            
            row[f"news_accepted_ratio_{l1}"] = (
                float((all_cat["validation_status"] == "accepted").mean())
                if total_cat_cnt > 0 else 0.0
            )

            cat = all_cat.filter(pl.col("validation_status") == "accepted")
            cnt = len(cat)
            row[f"news_count_{l1}"] = cnt
            row[f"news_max_conf_{l1}"] = (
                float(cat["confidence"].max()) if cnt > 0 else 0.0
            )
            if cnt > 0:
                # mode of expected_direction
                direction_mode = (
                    cat.group_by("expected_direction")
                    .agg(pl.len().alias("cnt"))
                    .sort("cnt", descending=True)["expected_direction"][0]
                )
                row[f"news_direction_{l1}"] = int(direction_mode)
            else:
                row[f"news_direction_{l1}"] = 0
                
            total_count += cnt
            if cnt > 0:
                sentiment_score += float(
                    (cat["confidence"] * cat["expected_direction"].cast(pl.Float64)).sum()
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_features_news.py::test_news_accepted_ratio -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_features_news.py src/trading_databricks/ml/features/news.py
git commit -m "fix(ml): calculate news_accepted_ratio before dropping rejected news"
```

---

### Task 5: Final verification

- [ ] Run: `uv run pytest tests/ml/test_features_*.py -v`
- [ ] Run: `uv run pytest -v`
- [ ] Run: `uv run ruff check src/trading_databricks/ml/features/`