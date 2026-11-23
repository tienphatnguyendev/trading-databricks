"""Mutual information scoring with permutation noise floor.

Wraps ``sklearn.feature_selection.mutual_info_classif`` with a
permutation-based null distribution to separate genuine signal
from estimation noise.

Reference: Phase 4B §8–9 of BTCUSDT EDA notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from btcusdt_utils.scoring._config import ScoringConfig


@dataclass(frozen=True)
class NoiseFloor:
    """Permutation noise-floor thresholds."""

    p95: float
    p99: float
    max_per_shuffle: np.ndarray  # shape (n_perms,)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"p95={self.p95:.6f}, p99={self.p99:.6f}, "
            f"n_perms={len(self.max_per_shuffle)})"
        )


def compute_mi_scores(
    X: np.ndarray,
    y: np.ndarray,
    *,
    config: ScoringConfig | None = None,
) -> np.ndarray:
    """Compute MI scores for each feature against the label.

    Parameters
    ----------
    X      : Feature matrix (n_samples, n_features).
    y      : Binary label array.
    config : Scoring configuration (keyword-only).

    Returns
    -------
    1-D array of MI scores, one per feature.
    """
    if config is None:
        config = ScoringConfig()
    return mutual_info_classif(
        X, y,
        discrete_features=False,
        n_neighbors=config.n_neighbors,
        random_state=config.seed,
    )


def compute_noise_floor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    config: ScoringConfig | None = None,
) -> NoiseFloor:
    """Estimate MI noise floor via label permutation.

    Shuffles *y* ``config.n_perms`` times, computes MI for each
    shuffle, and records the **max** MI across all features per
    shuffle.  The 95th and 99th percentiles of this max-distribution
    give conservative noise-floor thresholds.

    Parameters
    ----------
    X      : Feature matrix (n_samples, n_features).
    y      : Binary label array.
    config : Scoring configuration (keyword-only).

    Returns
    -------
    NoiseFloor with ``p95``, ``p99``, and per-shuffle max array.
    """
    if config is None:
        config = ScoringConfig()

    rng = np.random.default_rng(config.seed)
    max_mis = np.zeros(config.n_perms)

    for p in range(config.n_perms):
        y_shuf = rng.permutation(y)
        mi_shuf = mutual_info_classif(
            X, y_shuf,
            discrete_features=False,
            n_neighbors=config.n_neighbors,
            random_state=p,
        )
        max_mis[p] = mi_shuf.max()

    return NoiseFloor(
        p95=float(np.percentile(max_mis, 95)),
        p99=float(np.percentile(max_mis, 99)),
        max_per_shuffle=max_mis,
    )
