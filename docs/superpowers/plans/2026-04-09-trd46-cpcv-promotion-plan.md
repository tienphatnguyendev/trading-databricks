# TRD-46: CPCV + Sample Weights + Promotion Gates — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an sklearn-compatible `CPCVSplitter`, a thin `compute_sample_weights()` wrapper, and `evaluate_promotion_gates()` with PSR/DSR/strategy-failure formulas and 10 quantitative gates.

**Architecture:** Pure Python/numpy — no Polars, no Spark. `CPCVSplitter` injects label-span arrays at construction time (`t_start`, `t_end`) to enable proper purge/embargo without violating sklearn's `.split()` interface. `sample_weights.py` is a thin wrapper that extracts `avg_uniqueness` from a labeled Polars DataFrame as a numpy array. `promotion.py` computes PSR, DSR, and strategy-failure probability using `scipy.stats`, then evaluates 10 gates returning a dict compatible with `PROMOTION_EVIDENCE_SCHEMA`.

**Tech Stack:** `numpy`, `scipy.stats`, `itertools`, `polars` (for sample weights extraction only).

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `src/trading_databricks/ml/training/__init__.py` | CREATE | Re-export public API |
| `src/trading_databricks/ml/training/cpcv.py` | CREATE | `CPCVSplitter` (sklearn-compatible CV splitter) |
| `src/trading_databricks/ml/training/sample_weights.py` | CREATE | `compute_sample_weights()` thin wrapper |
| `src/trading_databricks/ml/training/promotion.py` | CREATE | `compute_psr`, `compute_dsr`, `evaluate_promotion_gates` |
| `tests/ml/test_cpcv.py` | CREATE | Tests for CPCVSplitter |
| `tests/ml/test_sample_weights.py` | CREATE | Tests for compute_sample_weights |
| `tests/ml/test_promotion.py` | CREATE | Tests for promotion gate evaluation |

---

## Task 1: `CPCVSplitter` — group formation + split count

**Files:**
- Create: `src/trading_databricks/ml/training/__init__.py`
- Create: `src/trading_databricks/ml/training/cpcv.py`
- Create: `tests/ml/test_cpcv.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/test_cpcv.py
from __future__ import annotations

import numpy as np
import pytest

from trading_databricks.ml.config import CPCVConfig
from trading_databricks.ml.training.cpcv import CPCVSplitter


def _make_splitter(n: int = 120, label_span: int = 10) -> CPCVSplitter:
    """n samples, 6 groups of n//6, each label spans `label_span` bars."""
    t_start = np.arange(n, dtype=int)
    t_end = t_start + label_span
    return CPCVSplitter(CPCVConfig(), t_start=t_start, t_end=t_end)


def test_n_splits_equals_c_n_k() -> None:
    """C(6,2) = 15 splits for N=6, k=2."""
    splitter = _make_splitter(120)
    X = np.random.randn(120, 4)
    splits = list(splitter.split(X))
    assert len(splits) == 15


def test_each_test_set_has_correct_size() -> None:
    """Each test set = k_test groups = 2 × (120//6) = 40 samples."""
    splitter = _make_splitter(120)
    X = np.random.randn(120, 4)
    for train_idx, test_idx in splitter.split(X):
        assert len(test_idx) == 40, f"Test set size {len(test_idx)} ≠ 40"


def test_test_sets_are_disjoint_across_splits() -> None:
    """Test sets may overlap across splits — but within a split, train ∩ test = ∅."""
    splitter = _make_splitter(120)
    X = np.random.randn(120, 4)
    for train_idx, test_idx in splitter.split(X):
        assert len(set(train_idx) & set(test_idx)) == 0


def test_raises_on_too_few_samples() -> None:
    """Fewer than n_groups samples → ValueError."""
    t_start = np.arange(3, dtype=int)
    t_end = t_start + 2
    splitter = CPCVSplitter(CPCVConfig(), t_start=t_start, t_end=t_end)
    X = np.random.randn(3, 4)
    with pytest.raises(ValueError, match="minimum"):
        list(splitter.split(X))


def test_get_n_splits_returns_15() -> None:
    """get_n_splits() returns C(6,2)=15."""
    splitter = _make_splitter(120)
    assert splitter.get_n_splits() == 15
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/ml/test_cpcv.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `cpcv.py` — groups + splits (no purge yet)**

```python
# src/trading_databricks/ml/training/cpcv.py
from __future__ import annotations

import math
from itertools import combinations
from typing import Generator, Iterator

import numpy as np

from trading_databricks.ml.config import CPCVConfig


class CPCVSplitter:
    """Combinatorial Purged Cross-Validation splitter (AFML Ch. 12).

    Sklearn-compatible: implements split(X, y=None, groups=None) and get_n_splits().
    Label spans are injected at construction time to enable purge/embargo.

    Args:
        config: CPCVConfig with n_groups, k_test, embargo_pct.
        t_start: 1-D int array of bar indices where each event's label STARTS.
                 Must be aligned with rows of X passed to .split().
        t_end: 1-D int array of bar indices where each event's label ENDS (vertical_bar).
               Must be aligned with rows of X passed to .split().

    IMPORTANT: Sort data by event_ts BEFORE extracting t_start, t_end, and X.
    """

    def __init__(
        self,
        config: CPCVConfig = CPCVConfig(),
        *,
        t_start: np.ndarray,
        t_end: np.ndarray,
    ) -> None:
        self.config = config
        self.t_start = np.asarray(t_start, dtype=int)
        self.t_end = np.asarray(t_end, dtype=int)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return number of splits: C(n_groups, k_test)."""
        n, k = self.config.n_groups, self.config.k_test
        return math.comb(n, k)

    def split(
        self, X: np.ndarray, y=None, groups=None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, test_indices) for each CPCV fold.

        Raises:
            ValueError: If n_samples < n_groups.
        """
        n_samples = len(X)
        n_groups = self.config.n_groups

        if n_samples < n_groups:
            raise ValueError(
                f"CPCV requires minimum {n_groups} samples, got {n_samples}. "
                "Cannot form groups."
            )

        # Assign each sample to a chronological group
        group_size = n_samples // n_groups
        group_labels = np.zeros(n_samples, dtype=int)
        for g in range(n_groups):
            start = g * group_size
            end = (g + 1) * group_size if g < n_groups - 1 else n_samples
            group_labels[start:end] = g

        # Generate all C(n_groups, k_test) test group combinations
        for test_groups in combinations(range(n_groups), self.config.k_test):
            test_mask = np.isin(group_labels, test_groups)
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]

            # Apply purge and embargo
            train_idx = self._purge(train_idx, test_idx)
            train_idx = self._embargo(train_idx, test_idx, n_samples)

            yield train_idx, test_idx

    def _purge(self, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
        """Remove training samples whose label spans overlap with any test span."""
        if len(test_idx) == 0 or len(train_idx) == 0:
            return train_idx

        test_min_start = int(self.t_start[test_idx].min())
        test_max_end = int(self.t_end[test_idx].max())

        # Pre-filter: only check training samples near the test range
        candidates_mask = (
            (self.t_end[train_idx] >= test_min_start)
            & (self.t_start[train_idx] <= test_max_end)
        )
        candidates = train_idx[candidates_mask]

        purged: set[int] = set()
        for i in candidates:
            for j in test_idx:
                if (
                    int(self.t_start[i]) <= int(self.t_end[j])
                    and int(self.t_end[i]) >= int(self.t_start[j])
                ):
                    purged.add(int(i))
                    break

        return np.array([i for i in train_idx if i not in purged], dtype=int)

    def _embargo(
        self, train_idx: np.ndarray, test_idx: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """Remove embargo_pct × n_samples samples immediately after test group end."""
        if len(test_idx) == 0:
            return train_idx

        embargo_size = max(1, int(self.config.embargo_pct * n_samples))
        test_max_idx = int(test_idx.max())
        embargo_set = set(
            range(test_max_idx + 1, min(test_max_idx + 1 + embargo_size, n_samples))
        )
        return np.array([i for i in train_idx if i not in embargo_set], dtype=int)

    def get_backtest_paths(self) -> list[list[tuple[int, ...]]]:
        """Return 5 non-overlapping paths, each covering all n_groups.

        Each path is a list of test_group combos that together cover groups 0..N-1.
        φ[N=6, k=2] = 5 paths.
        """
        n = self.config.n_groups
        k = self.config.k_test
        all_combos = list(combinations(range(n), k))

        paths: list[list[tuple[int, ...]]] = []
        used_combos: set[tuple[int, ...]] = set()

        def find_path(
            remaining_groups: set[int], current_path: list[tuple[int, ...]]
        ) -> list[tuple[int, ...]] | None:
            if not remaining_groups:
                return current_path
            for combo in all_combos:
                if combo in used_combos:
                    continue
                if all(g in remaining_groups for g in combo):
                    used_combos.add(combo)
                    result = find_path(
                        remaining_groups - set(combo), current_path + [combo]
                    )
                    if result is not None:
                        return result
                    used_combos.remove(combo)
            return None

        while len(paths) < (math.comb(n, k) // (n // k)):
            path = find_path(set(range(n)), [])
            if path is None:
                break
            paths.append(path)

        return paths
```

- [ ] **Step 4: Create `__init__.py`**

```python
# src/trading_databricks/ml/training/__init__.py
from trading_databricks.ml.training.cpcv import CPCVSplitter
from trading_databricks.ml.training.promotion import evaluate_promotion_gates
from trading_databricks.ml.training.sample_weights import compute_sample_weights

__all__ = ["CPCVSplitter", "compute_sample_weights", "evaluate_promotion_gates"]
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/ml/test_cpcv.py -v
```

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/trading_databricks/ml/training/ tests/ml/test_cpcv.py
git commit -m "TRD-46 feat: CPCVSplitter with group formation, purge, embargo, backtest paths"
```

---

## Task 2: Purge correctness + backtest paths

**Files:**
- Modify: `tests/ml/test_cpcv.py`

- [ ] **Step 1: Add purge and path tests**

Add to `tests/ml/test_cpcv.py`:

```python
def test_purge_removes_overlapping_training_samples() -> None:
    """Training sample whose label overlaps test set is purged."""
    # 120 samples, group boundary between sample 19 and 20.
    # Sample 19: t_start=19, t_end=29 → spans into group 1 (samples 20-39).
    # When groups (1,...) are in test, sample 19 must be purged from train.
    n = 120
    t_start = np.arange(n, dtype=int)
    t_end = t_start.copy()
    t_end[19] = 29  # sample 19 spans into group 1

    splitter = CPCVSplitter(CPCVConfig(), t_start=t_start, t_end=t_end)
    X = np.random.randn(n, 4)

    # Find the split where group 1 (samples 20-39) is in test set
    for train_idx, test_idx in splitter.split(X):
        if 20 in test_idx and 39 in test_idx:
            # Sample 19 must NOT be in train_idx
            assert 19 not in train_idx, (
                "Sample 19 should be purged: its label (19..29) overlaps test group 1 (20..39)"
            )
            break


def test_backtest_paths_cover_all_groups() -> None:
    """Each backtest path covers all 6 groups exactly once."""
    splitter = _make_splitter(120)
    paths = splitter.get_backtest_paths()

    assert len(paths) >= 1  # At least 1 path found
    for path in paths:
        covered = set()
        for combo in path:
            for g in combo:
                assert g not in covered, f"Group {g} covered twice in path {path}"
                covered.add(g)
        assert covered == set(range(6)), f"Path {path} doesn't cover all 6 groups"
```

- [ ] **Step 2: Run**

```bash
uv run pytest tests/ml/test_cpcv.py -v
```

Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/ml/test_cpcv.py
git commit -m "TRD-46 test: purge correctness and backtest path coverage"
```

---

## Task 3: `compute_sample_weights`

**Files:**
- Create: `src/trading_databricks/ml/training/sample_weights.py`
- Create: `tests/ml/test_sample_weights.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/test_sample_weights.py
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from trading_databricks.ml.training.sample_weights import compute_sample_weights


def test_returns_numpy_array_of_avg_uniqueness() -> None:
    """compute_sample_weights extracts avg_uniqueness as numpy float64 array."""
    labels = pl.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "avg_uniqueness": [1.0, 0.8, 0.6],
            "label": [1, -1, 0],
        }
    )
    weights = compute_sample_weights(labels)

    assert isinstance(weights, np.ndarray)
    assert weights.dtype == np.float64
    np.testing.assert_array_almost_equal(weights, [1.0, 0.8, 0.6])


def test_weights_length_matches_labels() -> None:
    """Output array length equals number of rows in labels."""
    labels = pl.DataFrame({"event_id": ["e1"], "avg_uniqueness": [0.75]})
    weights = compute_sample_weights(labels)
    assert len(weights) == 1


def test_raises_when_avg_uniqueness_column_missing() -> None:
    """Labels DataFrame without avg_uniqueness → ColumnNotFoundError."""
    labels = pl.DataFrame({"event_id": ["e1"], "label": [1]})
    with pytest.raises(Exception):  # pl.exceptions.ColumnNotFoundError
        compute_sample_weights(labels)
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/ml/test_sample_weights.py -v
```

- [ ] **Step 3: Implement `sample_weights.py`**

```python
# src/trading_databricks/ml/training/sample_weights.py
from __future__ import annotations

import numpy as np
import polars as pl


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
    """
    return labels.get_column("avg_uniqueness").to_numpy().astype(np.float64)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/ml/test_sample_weights.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/ml/training/sample_weights.py tests/ml/test_sample_weights.py
git commit -m "TRD-46 feat: compute_sample_weights thin wrapper extracting avg_uniqueness"
```

---

## Task 4: PSR, DSR, and strategy-failure formulas

**Files:**
- Create: `src/trading_databricks/ml/training/promotion.py`
- Create: `tests/ml/test_promotion.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/test_promotion.py
from __future__ import annotations

import numpy as np
import pytest
import scipy.stats

from trading_databricks.ml.training.promotion import (
    compute_dsr,
    compute_psr,
    compute_strategy_failure_prob,
    evaluate_promotion_gates,
)


def test_psr_known_distribution() -> None:
    """PSR with known SR and normal returns should match analytic value."""
    # Normal returns: mean=0.02, std=0.1 → SR=0.2
    rng = np.random.default_rng(0)
    returns = rng.normal(loc=0.02, scale=0.1, size=252)
    sr = float(np.mean(returns) / np.std(returns, ddof=1))
    psr = compute_psr(sr, returns, benchmark_sharpe=0.0)

    assert 0.0 <= psr <= 1.0
    # With SR=0.2 and T=252, PSR should be significantly above 0.5
    assert psr > 0.5, f"PSR={psr} expected > 0.5 for positive SR"


def test_psr_zero_sharpe_gives_point_five() -> None:
    """SR=0 vs benchmark=0 → PSR ≈ 0.5 (symmetric)."""
    rng = np.random.default_rng(1)
    returns = rng.normal(loc=0.0, scale=0.1, size=252)
    # Force SR=0 exactly
    returns = returns - returns.mean()
    psr = compute_psr(0.0, returns, benchmark_sharpe=0.0)
    assert abs(psr - 0.5) < 0.05, f"PSR={psr} expected ≈ 0.5"


def test_dsr_higher_than_psr_with_multiple_trials() -> None:
    """DSR is more conservative than PSR (accounts for multiple testing)."""
    rng = np.random.default_rng(2)
    returns = rng.normal(loc=0.02, scale=0.1, size=252)
    sr = float(np.mean(returns) / np.std(returns, ddof=1))
    psr = compute_psr(sr, returns, benchmark_sharpe=0.0)
    dsr = compute_dsr(sr, returns, n_trials=50)
    # DSR < PSR because DSR uses a higher benchmark (expected max SR)
    assert dsr < psr, f"DSR={dsr} should be < PSR={psr}"
    assert 0.0 <= dsr <= 1.0


def test_dsr_with_one_trial_equals_psr() -> None:
    """n_trials=1 → e_max_sr=0 → DSR = PSR(SR, benchmark=0)."""
    rng = np.random.default_rng(3)
    returns = rng.normal(loc=0.01, scale=0.1, size=100)
    sr = float(np.mean(returns) / np.std(returns, ddof=1))
    psr = compute_psr(sr, returns, benchmark_sharpe=0.0)
    dsr = compute_dsr(sr, returns, n_trials=1)
    assert abs(psr - dsr) < 1e-9, f"PSR={psr} and DSR={dsr} should be equal for n_trials=1"


def test_strategy_failure_prob_is_complement_of_psr() -> None:
    """P(failure) = 1 - PSR(SR, benchmark=0)."""
    rng = np.random.default_rng(4)
    returns = rng.normal(loc=0.01, scale=0.1, size=200)
    sr = float(np.mean(returns) / np.std(returns, ddof=1))
    psr = compute_psr(sr, returns, benchmark_sharpe=0.0)
    sfp = compute_strategy_failure_prob(sr, returns)
    assert abs(sfp - (1.0 - psr)) < 1e-9
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/ml/test_promotion.py::test_psr_known_distribution tests/ml/test_promotion.py::test_psr_zero_sharpe_gives_point_five -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement PSR/DSR functions in `promotion.py`**

```python
# src/trading_databricks/ml/training/promotion.py
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import polars as pl
import scipy.stats

logger = logging.getLogger(__name__)


def compute_psr(
    sharpe: float,
    returns: np.ndarray,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio (AFML Ch. 14).

    P(SR* > benchmark_sharpe | sample SR = sharpe).

    Uses raw kurtosis (fisher=False): normal distribution has γ₄=3.
    """
    t = len(returns)
    if t < 2:
        return 1.0 if sharpe > benchmark_sharpe else 0.0

    gamma_3 = float(scipy.stats.skew(returns))
    gamma_4 = float(scipy.stats.kurtosis(returns, fisher=False))  # raw kurtosis

    variance_sr = (
        1.0 - gamma_3 * sharpe + (gamma_4 - 1.0) / 4.0 * sharpe**2
    ) / (t - 1)

    if variance_sr <= 0.0:
        sigma_sr = 1e-10
    else:
        sigma_sr = float(np.sqrt(variance_sr))

    if sigma_sr < 1e-10:
        return 1.0 if sharpe > benchmark_sharpe else 0.0

    z = (sharpe - benchmark_sharpe) / sigma_sr
    return float(scipy.stats.norm.cdf(z))


def compute_dsr(
    sharpe: float,
    returns: np.ndarray,
    n_trials: int,
) -> float:
    """Deflated Sharpe Ratio (AFML Ch. 14).

    Adjusts PSR for multiple testing by using E[max(SR)] as benchmark.
    """
    euler_mascheroni = 0.5772156649

    if n_trials <= 1:
        if n_trials < 1:
            logger.warning("n_trials=%d < 1; DSR = PSR with benchmark=0.", n_trials)
        e_max_sr = 0.0
    else:
        e_max_sr = float(
            (1.0 - euler_mascheroni) * scipy.stats.norm.ppf(1.0 - 1.0 / n_trials)
            + euler_mascheroni * scipy.stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        )

    return compute_psr(sharpe, returns, benchmark_sharpe=e_max_sr)


def compute_strategy_failure_prob(
    sharpe: float,
    returns: np.ndarray,
) -> float:
    """P(strategy is unprofitable) = 1 - PSR(SR* = 0) (AFML Ch. 15)."""
    return 1.0 - compute_psr(sharpe, returns, benchmark_sharpe=0.0)
```

- [ ] **Step 4: Run PSR/DSR tests**

```bash
uv run pytest tests/ml/test_promotion.py::test_psr_known_distribution tests/ml/test_promotion.py::test_psr_zero_sharpe_gives_point_five tests/ml/test_promotion.py::test_dsr_higher_than_psr_with_multiple_trials tests/ml/test_promotion.py::test_dsr_with_one_trial_equals_psr tests/ml/test_promotion.py::test_strategy_failure_prob_is_complement_of_psr -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/ml/training/promotion.py tests/ml/test_promotion.py
git commit -m "TRD-46 feat: compute_psr, compute_dsr, compute_strategy_failure_prob"
```

---

## Task 5: `evaluate_promotion_gates`

**Files:**
- Modify: `src/trading_databricks/ml/training/promotion.py`
- Modify: `tests/ml/test_promotion.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/ml/test_promotion.py`:

```python
def _make_good_path(n: int = 252) -> dict:
    """Construct a path result that passes all gates."""
    rng = np.random.default_rng(99)
    returns = rng.normal(loc=0.003, scale=0.01, size=n)  # SR ≈ 0.3 * sqrt(252)
    sharpe = float(np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(252)
    cum = np.cumprod(1 + returns) - 1
    dd = float(np.min(cum - np.maximum.accumulate(cum + 1) + 1))  # simplified
    return {
        "returns": returns,
        "sharpe": sharpe,
        "max_drawdown": 0.05,   # 5% < 25% gate
        "time_under_water": 30,  # 30 days < 90 gate
    }


def _make_good_meta_metrics() -> dict:
    return {
        "precision": 0.65,   # > 0.60
        "recall": 0.55,      # > 0.50
        "f1": 0.60,          # > 0.55
        "logloss": 0.60,     # < 0.65
    }


def test_all_gates_pass_for_good_strategy() -> None:
    """Well-behaved strategy → all_gates_pass=True."""
    paths = [_make_good_path() for _ in range(5)]
    meta = _make_good_meta_metrics()
    result = evaluate_promotion_gates(paths, meta, trial_count=1)

    assert result["all_gates_pass"] is True
    assert result["failure_reasons"] is None or result["failure_reasons"] == ""


def test_max_drawdown_gate_fails() -> None:
    """Exceeding max drawdown threshold → all_gates_pass=False."""
    paths = [_make_good_path() for _ in range(5)]
    paths[0]["max_drawdown"] = 0.30  # exceeds 0.25 gate
    meta = _make_good_meta_metrics()
    result = evaluate_promotion_gates(paths, meta, trial_count=1)

    assert result["all_gates_pass"] is False
    assert "max_drawdown" in result["failure_reasons"]


def test_meta_precision_gate_fails() -> None:
    """Low meta precision → gate fails, included in failure_reasons."""
    paths = [_make_good_path() for _ in range(5)]
    meta = _make_good_meta_metrics()
    meta["precision"] = 0.50  # below 0.60 gate
    result = evaluate_promotion_gates(paths, meta, trial_count=1)

    assert result["all_gates_pass"] is False
    assert "meta_precision" in result["failure_reasons"]


def test_result_has_promotion_evidence_schema_keys() -> None:
    """Result dict has all required keys from PROMOTION_EVIDENCE_SCHEMA."""
    paths = [_make_good_path() for _ in range(5)]
    result = evaluate_promotion_gates(paths, _make_good_meta_metrics(), trial_count=5)

    required_keys = {
        "psr", "dsr", "max_drawdown", "time_under_water",
        "cost_robust", "strategy_failure_prob",
        "meta_precision", "meta_recall", "meta_f1", "meta_logloss",
        "all_gates_pass", "failure_reasons", "evaluated_at",
    }
    assert required_keys.issubset(set(result.keys()))
    assert isinstance(result["evaluated_at"], datetime)
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/ml/test_promotion.py::test_all_gates_pass_for_good_strategy -v
```

Expected: FAIL (`evaluate_promotion_gates` not implemented).

- [ ] **Step 3: Implement `evaluate_promotion_gates` in `promotion.py`**

Add to `promotion.py`:

```python
def _check_cost_robustness(
    path_returns: np.ndarray,
    estimated_cost_per_trade: float = 0.001,
) -> bool:
    """Strategy profitable at 2× estimated costs."""
    cost_adjusted = path_returns - 2.0 * estimated_cost_per_trade
    return bool(float(np.mean(cost_adjusted)) > 0.0)


def evaluate_promotion_gates(
    cpcv_path_results: list[dict],
    meta_label_metrics: dict,
    trial_count: int,
) -> dict:
    """Evaluate all 10 promotion gates (AFML methodology).

    Args:
        cpcv_path_results: List of 5 dicts, each with:
            - returns: np.ndarray of per-bar returns
            - sharpe: annualised Sharpe ratio (float)
            - max_drawdown: max drawdown fraction (float, positive)
            - time_under_water: days under water (int/float)
        meta_label_metrics: Dict with precision, recall, f1, logloss (floats).
        trial_count: Number of models evaluated so far (for DSR).

    Returns:
        Dict compatible with PROMOTION_EVIDENCE_SCHEMA fields (excluding
        run_id, model_type, model_version, dataset_version — those are
        set by the training pipeline).
    """
    if trial_count < 1:
        logger.warning("trial_count=%d; DSR = PSR (first trial).", trial_count)

    all_returns = np.concatenate([p["returns"] for p in cpcv_path_results])
    mean_sharpe = float(np.mean([p["sharpe"] for p in cpcv_path_results]))
    max_dd = float(max(p["max_drawdown"] for p in cpcv_path_results))
    max_tuw = float(max(p["time_under_water"] for p in cpcv_path_results))

    psr_val = compute_psr(mean_sharpe, all_returns)
    dsr_val = compute_dsr(mean_sharpe, all_returns, n_trials=max(trial_count, 1))
    sfp_val = compute_strategy_failure_prob(mean_sharpe, all_returns)
    cost_ok = _check_cost_robustness(all_returns)

    gate_results = {
        "psr": psr_val > 0.95,
        "dsr": dsr_val > 0.90,
        "max_drawdown": max_dd < 0.25,
        "time_under_water": max_tuw < 90,
        "cost_robust": cost_ok,
        "strategy_failure_prob": sfp_val < 0.10,
        "meta_precision": meta_label_metrics["precision"] > 0.60,
        "meta_recall": meta_label_metrics["recall"] > 0.50,
        "meta_f1": meta_label_metrics["f1"] > 0.55,
        "meta_logloss": meta_label_metrics["logloss"] < 0.65,
    }

    all_pass = all(gate_results.values())
    failure_reasons = ", ".join(k for k, v in gate_results.items() if not v) or None

    return {
        "psr": psr_val,
        "dsr": dsr_val,
        "max_drawdown": max_dd,
        "time_under_water": int(max_tuw),
        "cost_robust": cost_ok,
        "strategy_failure_prob": sfp_val,
        "meta_precision": float(meta_label_metrics["precision"]),
        "meta_recall": float(meta_label_metrics["recall"]),
        "meta_f1": float(meta_label_metrics["f1"]),
        "meta_logloss": float(meta_label_metrics["logloss"]),
        "all_gates_pass": all_pass,
        "failure_reasons": failure_reasons,
        "evaluated_at": datetime.utcnow(),
    }
```

- [ ] **Step 4: Run all promotion tests**

```bash
uv run pytest tests/ml/test_promotion.py -v
```

Expected: All PASS.

- [ ] **Step 5: Run full suite + lint**

```bash
uv run pytest tests/ml/test_cpcv.py tests/ml/test_sample_weights.py tests/ml/test_promotion.py -v
uv run ruff check src/trading_databricks/ml/training/
```

- [ ] **Step 6: Commit**

```bash
git add src/trading_databricks/ml/training/promotion.py tests/ml/test_promotion.py
git commit -m "TRD-46 feat: evaluate_promotion_gates with all 10 gates"
```

---

## Verification

```bash
uv run pytest tests/ml/test_cpcv.py tests/ml/test_sample_weights.py tests/ml/test_promotion.py -v
```

All tests pass. Final: 7 files created.
