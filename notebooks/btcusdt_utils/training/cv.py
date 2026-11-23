"""Purged cross-validation for time-series ML (AFML Ch. 7.4).

Implements purged K-fold and walk-forward cross-validation that prevent
look-ahead bias in time-series models where labels have overlapping event
spans (e.g., triple-barrier labels).

Key Features
------------
* **Purging**: Removes training samples whose event spans overlap with test events
* **Embargo**: Adds a buffer after each test block to prevent serial correlation leakage
* **Walk-Forward**: Expanding window splits for production-realistic evaluation

References
----------
* Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, Chapter 7.4

Examples
--------
>>> # Standard purged K-fold
>>> from btcusdt_utils.training import PurgedKFoldCV, PurgedCVConfig
>>> config = PurgedCVConfig(n_splits=5, pct_embargo=0.01)
>>> cv = PurgedKFoldCV(
...     n_splits=config.n_splits,
...     starts=event_starts,
...     ends=event_ends,
...     pct_embargo=config.pct_embargo,
... )
>>> splits = cv.split()
>>> for train_idx, test_idx in splits:
...     # Train on train_idx, evaluate on test_idx
...     pass

>>> # Walk-forward CV
>>> from btcusdt_utils.training import walk_forward_cv
>>> wf_splits = [
...     (0.0, 0.4, 0.55),  # train[0:40%], test[40%:55%]
...     (0.0, 0.55, 0.7),  # train[0:55%], test[55%:70%]
...     (0.0, 0.7, 0.85),  # train[0:70%], test[70%:85%]
...     (0.0, 0.85, 1.0),  # train[0:85%], test[85%:100%]
... ]
>>> results = walk_forward_cv(starts, ends, wf_splits, pct_embargo=0.01)
>>> for train_idx, test_idx, n_purged, n_embargoed in results:
...     print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
else:
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PurgedCVConfig:
    """Configuration for purged K-fold cross-validation.

    Parameters
    ----------
    n_splits : int
        Number of folds (default: 5).
    pct_embargo : float
        Fraction of total samples to use as embargo buffer after each
        test block. AFML default is 0.01 (1% of T). Higher values
        reduce serial correlation leakage but decrease training data.

    Examples
    --------
    >>> config = PurgedCVConfig(n_splits=5, pct_embargo=0.01)
    >>> config.n_splits
    5
    >>> config.pct_embargo
    0.01
    """

    n_splits: int = 5
    pct_embargo: float = 0.01


class PurgedKFoldCV:
    """Purged K-Fold Cross-Validation (AFML Ch. 7.4).

    Generates time-series K-fold splits where:
    1. Test sets are contiguous time blocks (chronological order preserved)
    2. Training samples with event spans overlapping test events are purged
    3. An embargo buffer is applied after the test boundary to prevent
       serial correlation from leaking into training data

    This prevents look-ahead bias when labels have overlapping event spans
    (e.g., triple-barrier labels where multiple bars share the same price
    movements).

    Parameters
    ----------
    n_splits : int
        Number of folds.
    starts : np.ndarray, shape (n_samples,)
        Start indices of each event (in bar index space). For triple-barrier
        labels, this is the bar where the label event begins.
    ends : np.ndarray, shape (n_samples,)
        End indices of each event (in bar index space). For triple-barrier
        labels, this is the bar where the barrier is first touched.
    pct_embargo : float
        Fraction of T (total samples) to use as embargo after test boundary.
        Default: 0.01 (AFML recommendation).

    Attributes
    ----------
    T : int
        Total number of samples.
    embargo_size : int
        Number of samples in the embargo buffer (computed as int(T * pct_embargo)).

    Examples
    --------
    >>> import numpy as np
    >>> starts = np.array([0, 2, 4, 6, 8])
    >>> ends = np.array([3, 5, 7, 9, 11])
    >>> cv = PurgedKFoldCV(n_splits=2, starts=starts, ends=ends, pct_embargo=0.1)
    >>> splits = cv.split()
    >>> len(splits)
    2
    >>> train_idx, test_idx = splits[0]
    >>> len(train_idx) > 0 and len(test_idx) > 0
    True

    Notes
    -----
    **Purging Logic**: A training sample i is purged if its event span
    [starts[i], ends[i]] overlaps with ANY test sample's event span
    [starts[j], ends[j]]. Overlap occurs when:
        starts[i] <= ends[j] AND ends[i] >= starts[j]

    **Embargo Logic**: After each test block, the next `embargo_size` samples
    are removed from training to prevent using data that is serially
    correlated with the test set.

    **Walk-Forward Alternative**: For production-realistic evaluation, see
    the walk_forward_cv() function which uses expanding windows instead of
    K-fold splits.

    References
    ----------
    * Lopez de Prado (2018), Advances in Financial Machine Learning, §7.4
    """

    def __init__(
        self,
        n_splits: int,
        starts: np.ndarray,
        ends: np.ndarray,
        pct_embargo: float = 0.01,
    ) -> None:
        if np is None:
            raise ImportError(
                "NumPy is required for PurgedKFoldCV. "
                "Install with: pip install numpy"
            )

        self.n_splits = n_splits
        self.starts = starts
        self.ends = ends
        self.pct_embargo = pct_embargo
        self.T = len(starts)
        self.embargo_size = int(self.T * pct_embargo)

        if len(starts) != len(ends):
            raise ValueError(
                f"starts and ends must have same length. "
                f"Got {len(starts)} vs {len(ends)}"
            )
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if not 0.0 <= pct_embargo < 1.0:
            raise ValueError(
                f"pct_embargo must be in [0, 1), got {pct_embargo}"
            )

    def split(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate purged train/test index arrays for each fold.

        Returns
        -------
        list of (train_indices, test_indices) tuples
            Each tuple contains NumPy arrays of sample indices.
            - train_indices: samples to train on (with purging + embargo applied)
            - test_indices: samples to evaluate on (contiguous time block)

        Notes
        -----
        The final fold's test set extends to the end of the dataset (T),
        while earlier folds have equal-sized test sets of size ~T/n_splits.

        Examples
        --------
        >>> cv = PurgedKFoldCV(n_splits=3, starts=starts, ends=ends)
        >>> splits = cv.split()
        >>> for k, (train_idx, test_idx) in enumerate(splits):
        ...     print(f"Fold {k}: {len(train_idx)} train, {len(test_idx)} test")
        """
        fold_size = self.T // self.n_splits
        splits = []

        for k in range(self.n_splits):
            # Define test block boundaries
            test_start = k * fold_size
            test_end = (
                (k + 1) * fold_size if k < self.n_splits - 1 else self.T
            )
            test_indices = np.arange(test_start, test_end)

            # Time range spanned by test events (in bar-index space)
            test_event_start = self.starts[test_indices].min()
            test_event_end = self.ends[test_indices].max()

            # Collect all candidate training indices (before purging/embargo)
            all_train = np.concatenate([
                np.arange(0, test_start),           # before test block
                np.arange(test_end, self.T),        # after test block
            ])

            # Purge: remove train samples whose event span overlaps test
            # A sample i overlaps if: starts[i] <= test_event_end AND ends[i] >= test_event_start
            overlap_mask = (
                (self.starts[all_train] <= test_event_end)
                & (self.ends[all_train] >= test_event_start)
            )

            # Embargo: remove samples within embargo_size after test block end
            # These have serial correlation with test set
            embargo_end = min(test_end + self.embargo_size, self.T)
            embargo_mask = (
                (all_train >= test_end) & (all_train < embargo_end)
            )

            # Combined exclusion: purge OR embargo
            exclude_mask = overlap_mask | embargo_mask
            train_indices = all_train[~exclude_mask]

            splits.append((train_indices, test_indices))

        return splits

    def get_diagnostics(self) -> list[dict]:
        """Return per-fold diagnostics (purge/embargo counts, test event spans).

        Returns
        -------
        list of dict
            One dictionary per fold with keys:
            - fold : int - fold index (0-indexed)
            - n_test : int - number of test samples
            - n_train_before_purge : int - training candidates before purging
            - n_purged : int - number of samples removed by purging
            - n_embargoed : int - number of samples removed by embargo (not already purged)
            - n_train_after : int - final training set size
            - purge_pct : float - percentage of candidates purged
            - test_event_span : int - bar span covered by test events (ends.max() - starts.min())

        Examples
        --------
        >>> cv = PurgedKFoldCV(n_splits=5, starts=starts, ends=ends, pct_embargo=0.01)
        >>> diags = cv.get_diagnostics()
        >>> for d in diags:
        ...     print(f"Fold {d['fold']}: purged {d['purge_pct']:.1f}%, "
        ...           f"train size {d['n_train_after']}")
        """
        fold_size = self.T // self.n_splits
        diagnostics = []

        for k in range(self.n_splits):
            test_start = k * fold_size
            test_end = (
                (k + 1) * fold_size if k < self.n_splits - 1 else self.T
            )
            test_indices = np.arange(test_start, test_end)

            test_event_start = self.starts[test_indices].min()
            test_event_end = self.ends[test_indices].max()

            all_train = np.concatenate([
                np.arange(0, test_start),
                np.arange(test_end, self.T),
            ])

            overlap_mask = (
                (self.starts[all_train] <= test_event_end)
                & (self.ends[all_train] >= test_event_start)
            )

            embargo_end = min(test_end + self.embargo_size, self.T)
            embargo_mask = (
                (all_train >= test_end) & (all_train < embargo_end)
            )
            # Embargo-only count (samples not already purged)
            embargo_only = embargo_mask & ~overlap_mask

            exclude_mask = overlap_mask | embargo_mask
            train_indices = all_train[~exclude_mask]

            diagnostics.append({
                "fold": k,
                "n_test": len(test_indices),
                "n_train_before_purge": len(all_train),
                "n_purged": int(overlap_mask.sum()),
                "n_embargoed": int(embargo_only.sum()),
                "n_train_after": len(train_indices),
                "purge_pct": float(overlap_mask.sum()) / len(all_train) * 100,
                "test_event_span": int(test_event_end - test_event_start),
            })

        return diagnostics


# ══════════════════════════════════════════════════════════════
# Walk-Forward Cross-Validation
# ══════════════════════════════════════════════════════════════

def walk_forward_cv(
    starts: np.ndarray,
    ends: np.ndarray,
    splits: list[tuple[float, float, float]],
    pct_embargo: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    """Walk-forward cross-validation with purging and embargo.

    Implements expanding-window walk-forward validation where the model is
    trained ONLY on past data (no look-ahead). Each split uses progressively
    more training data and tests on the subsequent time window.

    This is the production-realistic alternative to K-fold CV:
    - K-fold: trains on past AND future relative to test → look-ahead bias
    - Walk-forward: trains ONLY on past → deployment-realistic evaluation

    Parameters
    ----------
    starts : np.ndarray, shape (n_samples,)
        Start indices of each event (bar index space). For triple-barrier
        labels, this is the bar where the label event begins.
    ends : np.ndarray, shape (n_samples,)
        End indices of each event (bar index space). For triple-barrier
        labels, this is the bar where the barrier is first touched.
    splits : list of (train_start_frac, train_end_frac, test_end_frac) tuples
        Each tuple defines one walk-forward split as fractions of T:
        - train_start_frac: where training window starts (typically 0.0)
        - train_end_frac: where training window ends (test starts here)
        - test_end_frac: where test window ends (next split starts here)
        Example: [(0.0, 0.4, 0.55), (0.0, 0.55, 0.7), (0.0, 0.7, 0.85), (0.0, 0.85, 1.0)]
        means:
        - Split 0: train[0:40%], test[40%:55%]
        - Split 1: train[0:55%], test[55%:70%] (expanding window)
        - Split 2: train[0:70%], test[70%:85%]
        - Split 3: train[0:85%], test[85%:100%]
    pct_embargo : float
        Fraction of T to use as embargo at the end of training window.
        Default: 0.01 (AFML recommendation). Samples in the embargo zone
        are excluded from training to prevent serial correlation leakage.

    Returns
    -------
    list of (train_indices, test_indices, n_purged, n_embargoed) tuples
        Each tuple contains:
        - train_indices : np.ndarray - training sample indices (with purging + embargo)
        - test_indices : np.ndarray - test sample indices
        - n_purged : int - number of samples removed by purging
        - n_embargoed : int - number of samples removed by embargo

    Examples
    --------
    >>> import numpy as np
    >>> starts = np.arange(0, 1000)
    >>> ends = np.arange(10, 1010)
    >>> wf_splits = [
    ...     (0.0, 0.4, 0.55),
    ...     (0.0, 0.55, 0.7),
    ...     (0.0, 0.7, 0.85),
    ...     (0.0, 0.85, 1.0),
    ... ]
    >>> results = walk_forward_cv(starts, ends, wf_splits, pct_embargo=0.01)
    >>> for k, (train_idx, test_idx, n_purged, n_emb) in enumerate(results):
    ...     print(f"Split {k}: {len(train_idx)} train, {len(test_idx)} test, "
    ...           f"purged {n_purged}, embargoed {n_emb}")

    Notes
    -----
    **Purging**: Training samples whose event spans overlap with test events
    are removed. This is critical when events span multiple bars (e.g.,
    triple-barrier labels where a single price movement affects multiple samples).

    **Embargo**: Additional samples at the end of the training window (before
    the test set) are excluded to prevent serial correlation. The embargo zone
    is `int(T * pct_embargo)` samples.

    **Production Deployment**: The final split (train on 85%, test on 15%) best
    represents how the model will perform in production, where it only has
    access to historical data.

    References
    ----------
    * Lopez de Prado (2018), Advances in Financial Machine Learning, Ch. 7
    * Phase 6B implementation: BTCUSDT EDA Parameter Optimization notebook

    See Also
    --------
    PurgedKFoldCV : K-fold cross-validation with purging (trains on past + future)
    """
    if np is None:
        raise ImportError(
            "NumPy is required for walk_forward_cv. "
            "Install with: pip install numpy"
        )

    if len(starts) != len(ends):
        raise ValueError(
            f"starts and ends must have same length. "
            f"Got {len(starts)} vs {len(ends)}"
        )

    T = len(starts)
    embargo_size = int(T * pct_embargo)
    wf_split_indices = []

    for _, train_end_frac, test_end_frac in splits:
        # Convert fractions to indices
        train_end = int(T * train_end_frac)
        test_end = int(T * test_end_frac) if test_end_frac < 1.0 else T
        test_start = train_end

        # Define test and raw training indices
        test_idx = np.arange(test_start, test_end)
        train_idx_raw = np.arange(0, train_end)

        # Find time range spanned by test events
        test_event_start = starts[test_idx].min()
        test_event_end = ends[test_idx].max()

        # Purge: remove training samples whose events overlap test events
        # A training sample i overlaps if:
        #   starts[i] <= test_event_end AND ends[i] >= test_event_start
        overlap = (
            (starts[train_idx_raw] <= test_event_end)
            & (ends[train_idx_raw] >= test_event_start)
        )
        n_purged = int(overlap.sum())

        # Embargo: remove embargo_size samples at end of training window
        # These samples are too close to test set (serial correlation)
        embargo_start = max(0, train_end - embargo_size)
        embargo_mask = train_idx_raw >= embargo_start
        embargo_only = embargo_mask & ~overlap  # Don't double-count purged
        n_embargoed = int(embargo_only.sum())

        # Combined exclusion: remove samples that are purged OR embargoed
        exclude = overlap | embargo_mask
        train_idx = train_idx_raw[~exclude]

        wf_split_indices.append((train_idx, test_idx, n_purged, n_embargoed))

    return wf_split_indices
