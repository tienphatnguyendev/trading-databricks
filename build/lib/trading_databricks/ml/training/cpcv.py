from __future__ import annotations

import math
from collections.abc import Iterator
from itertools import combinations

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
        config: CPCVConfig | None = None,
        *,
        t_start: np.ndarray,
        t_end: np.ndarray,
    ) -> None:
        self.config = config if config is not None else CPCVConfig()
        self.t_start = np.asarray(t_start, dtype=int)
        self.t_end = np.asarray(t_end, dtype=int)

    def get_n_splits(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> int:  # noqa: ARG002
        """Return number of splits: C(n_groups, k_test)."""
        n, k = self.config.n_groups, self.config.k_test
        return math.comb(n, k)

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:  # noqa: ARG002
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
        """Remove embargo_pct × n_samples samples after each test segment."""
        if len(test_idx) == 0:
            return train_idx

        embargo_size = max(1, int(self.config.embargo_pct * n_samples))

        # Find contiguous segments of test indices and embargo after each
        sorted_test = np.sort(test_idx)
        gaps = np.where(np.diff(sorted_test) > 1)[0]
        segment_ends = np.append(sorted_test[gaps], sorted_test[-1])

        embargo_set: set[int] = set()
        for seg_end in segment_ends:
            end = int(seg_end)
            embargo_set.update(range(end + 1, min(end + 1 + embargo_size, n_samples)))

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
