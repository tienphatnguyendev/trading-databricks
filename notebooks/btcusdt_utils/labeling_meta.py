"""Meta-Labeling Framework (AFML Ch. 3.6)

Meta-labeling decouples the trading decision into two stages:
  1. Primary model: SIDE decision (which direction to trade)
  2. Secondary model: SIZE decision (how much to bet, given the side)

The secondary model predicts P(primary_correct | features), allowing
selective betting and position sizing based on confidence.

This module provides:
  - MetaLabelConfig: Configuration dataclass
  - generate_meta_labels(): Convert primary predictions → meta-labels (1=correct, 0=wrong)
  - create_meta_features(): Create lagged persistence features (no target leakage)

When direction prediction is weak but a simple heuristic (e.g., persistence)
captures direction well, meta-labeling can improve Sharpe ratio by filtering
low-confidence trades.

Extracted from: BTCUSDT EDA Parameter Optimization, Phase 7A-fix (Cell 32)
Author: Databricks AI Assistant
Date: 2026-05-01
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


__all__ = [
    "MetaLabelConfig",
    "generate_meta_labels",
    "create_meta_features",
]


@dataclass(frozen=True)
class MetaLabelConfig:
    """Configuration for meta-labeling.
    
    Attributes:
        primary_model: Primary model type ("persistence" = predict label[i] = label[i-1])
        switch_window: Lookback window for label switch rate calculation (default: 20)
        add_log_features: Whether to add log-transformed streak features (default: True)
    """
    primary_model: Literal["persistence"] = "persistence"
    switch_window: int = 20
    add_log_features: bool = True

    def __post_init__(self) -> None:
        if self.switch_window < 1:
            raise ValueError(f"switch_window must be >= 1, got {self.switch_window}")


def generate_meta_labels(
    y_actual: NDArray[np.int32],
    y_primary: NDArray[np.int32] | None = None,
) -> NDArray[np.int32]:
    """Generate meta-labels: 1 if primary prediction was correct, 0 if wrong.
    
    Meta-label[i] = 1 if primary_pred[i] == actual[i], else 0.
    
    If y_primary is None, defaults to persistence (predict label[i] = label[i-1]).
    First bar is assumed correct (no prior label to compare).
    
    Args:
        y_actual: Actual labels (triple-barrier outcomes), shape (T,)
        y_primary: Primary model predictions, shape (T,). If None, uses persistence.
    
    Returns:
        Meta-labels (0 or 1), shape (T,)
    
    Example:
        >>> y_actual = np.array([1, 1, 0, 0, 1], dtype=np.int32)
        >>> meta = generate_meta_labels(y_actual)  # persistence
        >>> meta
        array([1, 1, 0, 1, 0], dtype=int32)
    """
    if y_primary is None:
        # Default primary model: persistence (shifted labels)
        y_primary = np.roll(y_actual, 1)
        y_primary[0] = y_actual[0]  # Assume first bar is correct
    
    if len(y_actual) != len(y_primary):
        raise ValueError(
            f"y_actual and y_primary must have same length: "
            f"{len(y_actual)} vs {len(y_primary)}"
        )
    
    # Meta-label: 1 if primary was correct, 0 if wrong
    meta_labels = (y_primary == y_actual).astype(np.int32)
    return meta_labels


def _compute_persist_streak(labels: NDArray[np.int32]) -> NDArray[np.float64]:
    """Count consecutive same-label bars (streak length).
    
    streak[i] = number of consecutive bars with same label ending at bar i.
    
    Args:
        labels: Label array, shape (T,)
    
    Returns:
        Streak lengths, shape (T,)
    """
    streak = np.ones(len(labels), dtype=np.float64)
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            streak[i] = streak[i - 1] + 1
        else:
            streak[i] = 1
    return streak


def _compute_label_switch_rate(
    labels: NDArray[np.int32],
    window: int = 20,
) -> NDArray[np.float64]:
    """Rolling rate of label changes in the last `window` bars.
    
    switch_rate[i] = fraction of bars in window [i-window+1, i] where label changed.
    For i < window, uses expanding window average.
    
    Args:
        labels: Label array, shape (T,)
        window: Lookback window size (default: 20)
    
    Returns:
        Switch rates, shape (T,)
    """
    switches = np.zeros(len(labels), dtype=np.float64)
    switches[1:] = (labels[1:] != labels[:-1]).astype(np.float64)
    
    # Cumulative sum for rolling window
    cum_switches = np.cumsum(switches)
    rate = np.zeros(len(labels), dtype=np.float64)
    
    # Rolling window
    for i in range(window, len(labels)):
        rate[i] = (cum_switches[i] - cum_switches[i - window]) / window
    
    # Expanding window for early bars
    for i in range(1, min(window, len(labels))):
        rate[i] = cum_switches[i] / i
    
    return rate


def create_meta_features(
    X_base: NDArray[np.float64],
    y_labels: NDArray[np.int32],
    config: MetaLabelConfig | None = None,
) -> tuple[NDArray[np.float64], list[str]]:
    """Create meta-labeling features with proper lag (no target leakage).
    
    Combines base features with lagged persistence features:
      - persist_streak_lag1: streak length as of bar i-1 (not bar i)
      - label_switch_rate_lag1: switch rate up to bar i-1
      - log_streak_lag1: log1p(persist_streak_lag1) (optional)
    
    All persistence features are lagged by 1 bar to prevent target leakage.
    At prediction time t, we only know y[t-1] (previous bar's label),
    NOT y[t] (current bar hasn't resolved yet).
    
    Args:
        X_base: Base feature matrix, shape (T, F)
        y_labels: Actual labels (triple-barrier outcomes), shape (T,)
        config: Meta-label configuration (if None, uses defaults)
    
    Returns:
        (X_meta, feature_names) where:
          - X_meta: Combined feature matrix, shape (T, F + 2 or F + 3)
          - feature_names: List of feature names (base + lagged)
    
    Example:
        >>> X_base = np.random.randn(1000, 10)
        >>> y_labels = np.random.randint(0, 2, 1000, dtype=np.int32)
        >>> X_meta, names = create_meta_features(X_base, y_labels)
        >>> X_meta.shape
        (1000, 13)  # 10 base + 3 lagged
    """
    if config is None:
        config = MetaLabelConfig()
    
    if len(X_base) != len(y_labels):
        raise ValueError(
            f"X_base and y_labels must have same length: "
            f"{len(X_base)} vs {len(y_labels)}"
        )
    
    # Compute persistence features (unlagged)
    persist_streak = _compute_persist_streak(y_labels)
    label_switch_rate = _compute_label_switch_rate(y_labels, window=config.switch_window)
    
    # Lag by 1 bar to prevent target leakage
    persist_streak_lag = np.roll(persist_streak, 1)
    persist_streak_lag[0] = 1.0  # No history for first bar
    
    label_switch_rate_lag = np.roll(label_switch_rate, 1)
    label_switch_rate_lag[0] = 0.0
    
    # Build combined feature matrix
    extra_features = [persist_streak_lag, label_switch_rate_lag]
    feature_names_extra = ["persist_streak_lag1", "label_switch_rate_lag1"]
    
    if config.add_log_features:
        log_streak_lag = np.log1p(persist_streak_lag)
        extra_features.append(log_streak_lag)
        feature_names_extra.append("log_streak_lag1")
    
    X_extra = np.column_stack(extra_features)
    X_meta = np.column_stack([X_base, X_extra])
    
    # Feature names: assume base features named "feat_0", "feat_1", ...
    # Caller should replace with actual names if available
    base_names = [f"feat_{i}" for i in range(X_base.shape[1])]
    feature_names = base_names + feature_names_extra
    
    return X_meta, feature_names
