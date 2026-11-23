"""Frozen dataclass config for feature scoring modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringConfig:
    """Parameters for MI scoring and univariate AUC evaluation.

    Attributes
    ----------
    n_neighbors   : k for mutual information k-NN estimator.
    n_perms       : Number of label-permutation shuffles for noise floor.
    test_ratio    : Fraction of data reserved for chronological test split.
    lgb_max_depth : Max tree depth for single-feature LightGBM.
    lgb_num_leaves: Number of leaves for single-feature LightGBM.
    lgb_lr        : Learning rate for single-feature LightGBM.
    lgb_rounds    : Max boosting rounds.
    lgb_early_stop: Early stopping patience.
    seed          : Global random seed.
    """

    n_neighbors: int = 5
    n_perms: int = 100
    test_ratio: float = 0.2
    lgb_max_depth: int = 3
    lgb_num_leaves: int = 8
    lgb_lr: float = 0.1
    lgb_rounds: int = 100
    lgb_early_stop: int = 10
    seed: int = 42
