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
    for _train_idx, test_idx in splitter.split(X):
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
                "Sample 19 should be purged: label (19..29) overlaps test group 1"
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
