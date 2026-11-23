"""Bet Sizing Utilities (AFML Ch. 3.6 - Position Sizing)

Position sizing algorithms that convert meta-labeling probabilities into
bet sizes. Meta-labeling decouples SIDE (direction) from SIZE (confidence),
allowing selective betting based on predicted primary model accuracy.

This module provides:
  - kelly_bet_size(): Kelly criterion position sizing
  - linear_bet_size(): Linear probability-to-size scaling
  - bet_size_from_meta_prob(): High-level wrapper for meta-labeling pipelines

When combined with meta-labeling, these functions convert P(primary_correct)
into actionable position sizes, improving Sharpe ratio by filtering low-
confidence trades and scaling bets proportionally to edge.

Extracted from: BTCUSDT EDA Parameter Optimization, Phase 7C (Cell 35)
Author: Databricks AI Assistant
Date: 2026-05-01
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray


__all__ = [
    "kelly_bet_size",
    "linear_bet_size",
    "bet_size_from_meta_prob",
]


def kelly_bet_size(
    prob: float | NDArray[np.float64],
    odds: float = 1.0,
    max_leverage: float = 1.0,
) -> float | NDArray[np.float64]:
    """Calculate Kelly criterion bet size.
    
    Kelly formula: f* = (p*b - q) / b
    where:
      - p = probability of win
      - q = 1 - p (probability of loss)
      - b = odds received on the bet (net odds)
      - f* = fraction of capital to bet
    
    For even odds (b=1), this simplifies to: f* = 2p - 1
    
    Args:
        prob: Win probability or array of probabilities, shape (N,)
        odds: Net odds on the bet (default 1.0 for even money)
        max_leverage: Maximum fraction of capital to bet (default 1.0)
    
    Returns:
        Bet size as fraction of capital, shape (N,) or scalar
        Returns 0 for prob <= 0.5 (no edge)
    
    Example:
        >>> kelly_bet_size(0.55, odds=1.0)  # 55% win prob, even odds
        0.1
        >>> kelly_bet_size(0.60, odds=1.0)
        0.2
        >>> kelly_bet_size(np.array([0.50, 0.55, 0.60, 0.70]))
        array([0. , 0.1, 0.2, 0.4])
    
    Notes:
        - Returns 0 when prob <= 0.5 (no positive edge)
        - Full Kelly can be aggressive; many practitioners use fractional Kelly
          (e.g., 0.5 * kelly_bet_size()) for more stable equity curves
    """
    # Ensure prob is in [0, 1]
    prob = np.clip(prob, 0.0, 1.0)
    
    # Kelly formula: f* = (p*b - q) / b = (p*b - (1-p)) / b
    kelly_frac = (prob * odds - (1 - prob)) / odds
    
    # No betting if no edge (prob <= 0.5 for even odds)
    kelly_frac = np.clip(kelly_frac, 0.0, max_leverage)
    
    return kelly_frac


def linear_bet_size(
    prob: float | NDArray[np.float64],
    min_prob: float = 0.5,
    max_prob: float = 1.0,
    min_size: float = 0.0,
    max_size: float = 1.0,
) -> float | NDArray[np.float64]:
    """Linear scaling from probability to bet size.
    
    Maps probability range [min_prob, max_prob] to bet size range [min_size, max_size].
    Probabilities below min_prob return min_size (often 0 = no bet).
    Probabilities above max_prob return max_size (capped position).
    
    Formula:
        size = min_size + (prob - min_prob) / (max_prob - min_prob) * (max_size - min_size)
    
    Args:
        prob: Probability or array of probabilities, shape (N,)
        min_prob: Probability threshold for min_size (default 0.5)
        max_prob: Probability threshold for max_size (default 1.0)
        min_size: Bet size at min_prob (default 0.0 = no bet)
        max_size: Bet size at max_prob (default 1.0 = full capital)
    
    Returns:
        Bet size, shape (N,) or scalar, in [min_size, max_size]
    
    Example:
        >>> linear_bet_size(0.50)  # At threshold, no bet
        0.0
        >>> linear_bet_size(0.75)  # Midpoint between 0.5 and 1.0
        0.5
        >>> linear_bet_size(0.90)  # 80% of the way to max_prob
        0.8
        >>> linear_bet_size(np.array([0.50, 0.60, 0.70, 0.80, 0.90]))
        array([0. , 0.2, 0.4, 0.6, 0.8])
    
    Notes:
        - Simple and intuitive: higher confidence = larger bet
        - Less aggressive than Kelly criterion
        - Good for conservative strategies or when true odds are unknown
    """
    # Clip prob to [min_prob, max_prob]
    prob = np.clip(prob, min_prob, max_prob)
    
    # Linear interpolation
    prob_range = max_prob - min_prob
    size_range = max_size - min_size
    
    if prob_range == 0:
        # Degenerate case: all probs map to min_size
        return np.full_like(prob, min_size) if isinstance(prob, np.ndarray) else min_size
    
    size = min_size + (prob - min_prob) / prob_range * size_range
    
    return size


def bet_size_from_meta_prob(
    meta_prob: NDArray[np.float64],
    method: Literal["kelly", "linear", "threshold"] = "linear",
    threshold: float | None = None,
    **kwargs,
) -> NDArray[np.float64]:
    """Convert meta-labeling probabilities to bet sizes.
    
    High-level wrapper for position sizing in meta-labeling pipelines.
    Meta-prob represents P(primary_model_correct | features).
    
    Args:
        meta_prob: Meta-label probabilities, shape (N,)
        method: Sizing method - "kelly", "linear", or "threshold"
        threshold: For "threshold" method, bet full size if prob >= threshold, else 0
        **kwargs: Additional arguments passed to kelly_bet_size() or linear_bet_size()
    
    Returns:
        Bet sizes, shape (N,), typically in [0, 1]
    
    Methods:
        - "kelly": Kelly criterion (kwargs: odds, max_leverage)
        - "linear": Linear scaling (kwargs: min_prob, max_prob, min_size, max_size)
        - "threshold": Binary filtering (kwargs: threshold - if None, defaults to 0.5)
    
    Example:
        >>> meta_prob = np.array([0.50, 0.60, 0.70, 0.80, 0.90])
        >>> 
        >>> # Linear scaling (default)
        >>> bet_size_from_meta_prob(meta_prob, method="linear")
        array([0. , 0.2, 0.4, 0.6, 0.8])
        >>> 
        >>> # Kelly criterion
        >>> bet_size_from_meta_prob(meta_prob, method="kelly", odds=1.0)
        array([0. , 0.2, 0.4, 0.6, 0.8])
        >>> 
        >>> # Threshold filtering
        >>> bet_size_from_meta_prob(meta_prob, method="threshold", threshold=0.65)
        array([0., 0., 1., 1., 1.])
    
    Notes:
        - For meta-labeling: meta_prob is output from secondary model
        - Primary model (e.g., persistence) determines SIDE (long/short)
        - This function determines SIZE (how much to bet)
    """
    if method == "kelly":
        return kelly_bet_size(meta_prob, **kwargs)
    
    elif method == "linear":
        return linear_bet_size(meta_prob, **kwargs)
    
    elif method == "threshold":
        thresh = threshold if threshold is not None else 0.5
        max_size = kwargs.get("max_size", 1.0)
        return np.where(meta_prob >= thresh, max_size, 0.0)
    
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Expected one of: 'kelly', 'linear', 'threshold'"
        )
