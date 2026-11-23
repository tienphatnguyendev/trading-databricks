"""Sample weighting for overlapping labels.

Implements uniqueness weights and return attribution from Lopez de Prado AFML Chapter 4.

Key concepts:
* Uniqueness: Down-weight samples with concurrent label overlap
* Return attribution: Weight samples by their return importance
* Combined weighting: Uniqueness × return attribution

References:
* Advances in Financial Machine Learning (Lopez de Prado, 2018), Chapter 4
"""

import numpy as np
import pandas as pd
from typing import Optional


def get_ind_matrix(
    bar_index: np.ndarray,
    event_starts: np.ndarray,
    event_ends: np.ndarray,
) -> pd.DataFrame:
    """Build indicator matrix showing which bars belong to which events.
    
    For each event i and bar t, ind_matrix[t, i] = 1 if bar t is within
    the time span of event i (i.e., event_starts[i] <= t <= event_ends[i]).
    
    This is used to compute label overlap and average uniqueness.
    
    Parameters
    ----------
    bar_index : np.ndarray
        Array of bar indices for each sample (shape: n_samples)
    event_starts : np.ndarray
        Starting bar index for each event (shape: n_samples)
    event_ends : np.ndarray
        Ending bar index for each event (shape: n_samples)
    
    Returns
    -------
    ind_matrix : pd.DataFrame
        Binary indicator matrix (shape: n_bars × n_samples)
        ind_matrix.loc[bar, event] = 1 if bar is in event's time span
    
    Complexity
    ----------
    O(n_samples × n_bars) — sparse for short events
    
    Example
    -------
    >>> bar_index = np.array([0, 5, 10])
    >>> event_starts = np.array([0, 5, 10])
    >>> event_ends = np.array([3, 8, 12])
    >>> ind = get_ind_matrix(bar_index, event_starts, event_ends)
    >>> ind.loc[6, 1]  # Bar 6 is in event 1 (spans bars 5-8)
    1.0
    >>> ind.loc[6, 0]  # Bar 6 is NOT in event 0 (spans bars 0-3)
    0.0
    """
    # Get unique bars (may be less than max bar if not all bars have events)
    unique_bars = np.unique(np.concatenate([event_starts, event_ends, bar_index]))
    
    # Initialize indicator matrix: rows = bars, cols = events
    ind_matrix = pd.DataFrame(
        0.0,
        index=unique_bars,
        columns=range(len(bar_index)),
    )
    
    # For each event, mark all bars in its time span
    for i, (start, end) in enumerate(zip(event_starts, event_ends)):
        # Find bars in range [start, end]
        bars_in_event = unique_bars[(unique_bars >= start) & (unique_bars <= end)]
        ind_matrix.loc[bars_in_event, i] = 1.0
    
    return ind_matrix


def compute_avg_uniqueness(
    bar_index: np.ndarray,
    event_starts: np.ndarray,
    event_ends: np.ndarray,
) -> np.ndarray:
    """Compute average uniqueness for each sample.
    
    Uniqueness measures how much each sample's label is influenced by
    concurrent events. For event i spanning bars [t1, t2], uniqueness
    at bar t is 1 / (number of concurrent events at bar t).
    
    Average uniqueness is the mean uniqueness across all bars in the event.
    
    Parameters
    ----------
    bar_index : np.ndarray
        Array of bar indices for each sample (shape: n_samples)
    event_starts : np.ndarray
        Starting bar index for each event (shape: n_samples)
    event_ends : np.ndarray
        Ending bar index for each event (shape: n_samples)
    
    Returns
    -------
    avg_uniqueness : np.ndarray
        Average uniqueness for each sample (shape: n_samples)
        Values in (0, 1], where 1 = no overlap, <1 = some overlap
    
    Complexity
    ----------
    O(n_samples × n_bars) for indicator matrix
    O(n_bars × n_samples) for concurrency
    O(n_samples) for averaging
    Total: O(n_samples × n_bars)
    
    Example
    -------
    >>> bar_index = np.array([0, 1, 2])
    >>> event_starts = np.array([0, 0, 2])  # Events 0,1 overlap at bar 0-1
    >>> event_ends = np.array([1, 2, 3])
    >>> uniqueness = compute_avg_uniqueness(bar_index, event_starts, event_ends)
    >>> uniqueness[0]  # Event 0: bars 0-1 shared with event 1
    0.5  # (1/2 + 1/2) / 2
    >>> uniqueness[2]  # Event 2: bars 2-3, bar 2 shared, bar 3 unique
    0.75  # (1/2 + 1/1) / 2
    """
    # Build indicator matrix
    ind_matrix = get_ind_matrix(bar_index, event_starts, event_ends)
    
    # Compute concurrency: number of events at each bar
    # concurrency[t] = number of events whose span includes bar t
    concurrency = ind_matrix.sum(axis=1)  # shape: (n_bars,)
    
    # Compute uniqueness matrix: 1 / concurrency for each (bar, event) pair
    # If ind_matrix[t, i] = 1 and concurrency[t] = 3, then uniqueness[t, i] = 1/3
    uniqueness_matrix = ind_matrix.divide(concurrency, axis=0)
    
    # Compute average uniqueness for each event
    # avg_uniqueness[i] = mean of uniqueness values across bars in event i's span
    avg_uniqueness = uniqueness_matrix[uniqueness_matrix > 0].mean(axis=0).values
    
    return avg_uniqueness


def compute_return_attribution_weights(
    returns: np.ndarray,
    time_decay: float = 1.0,
) -> np.ndarray:
    """Compute return attribution weights.
    
    Weight samples by their absolute return, with optional time decay.
    Samples with larger |return| get higher weight.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns for each sample (shape: n_samples)
    time_decay : float, optional
        Exponential time decay factor (default: 1.0 = no decay)
        If < 1.0, older samples get exponentially lower weight
    
    Returns
    -------
    weights : np.ndarray
        Return attribution weights (shape: n_samples)
        Normalized to sum to n_samples
    
    Complexity
    ----------
    O(n_samples)
    
    Example
    -------
    >>> returns = np.array([0.01, -0.02, 0.005])
    >>> weights = compute_return_attribution_weights(returns)
    >>> weights  # Proportional to |return|
    array([1.33, 2.67, 0.67])  # Normalized to sum to 3
    """
    # Absolute return magnitude
    abs_returns = np.abs(returns)
    
    # Apply time decay if specified
    if time_decay < 1.0:
        n = len(returns)
        decay_factors = time_decay ** np.arange(n - 1, -1, -1)
        abs_returns = abs_returns * decay_factors
    
    # Normalize to sum to n_samples (so mean weight = 1.0)
    weights = abs_returns / abs_returns.mean()
    
    return weights


def compute_sample_weights(
    bar_index: np.ndarray,
    event_starts: np.ndarray,
    event_ends: np.ndarray,
    returns: Optional[np.ndarray] = None,
    time_decay: float = 1.0,
) -> np.ndarray:
    """Compute combined sample weights.
    
    Combines uniqueness weights (to down-weight overlapping labels) with
    optional return attribution weights (to up-weight important samples).
    
    Final weight: uniqueness × return_attribution
    
    Parameters
    ----------
    bar_index : np.ndarray
        Array of bar indices for each sample (shape: n_samples)
    event_starts : np.ndarray
        Starting bar index for each event (shape: n_samples)
    event_ends : np.ndarray
        Ending bar index for each event (shape: n_samples)
    returns : np.ndarray, optional
        Array of returns for each sample (default: None = no return weighting)
    time_decay : float, optional
        Exponential time decay factor for return weights (default: 1.0)
    
    Returns
    -------
    weights : np.ndarray
        Combined sample weights (shape: n_samples)
        Normalized to sum to n_samples (mean weight = 1.0)
    
    Complexity
    ----------
    O(n_samples × n_bars) for uniqueness
    O(n_samples) for return attribution
    Total: O(n_samples × n_bars)
    
    Example
    -------
    >>> bar_index = np.array([0, 1, 2])
    >>> event_starts = np.array([0, 0, 2])
    >>> event_ends = np.array([1, 2, 3])
    >>> returns = np.array([0.01, -0.02, 0.005])
    >>> weights = compute_sample_weights(
    ...     bar_index, event_starts, event_ends, returns
    ... )
    >>> weights.sum()  # Should sum to 3 (n_samples)
    3.0
    """
    # Compute uniqueness weights
    uniqueness = compute_avg_uniqueness(bar_index, event_starts, event_ends)
    
    # Compute return attribution weights if returns provided
    if returns is not None:
        return_weights = compute_return_attribution_weights(returns, time_decay)
        # Combine: uniqueness × return_attribution
        weights = uniqueness * return_weights
    else:
        weights = uniqueness
    
    # Normalize to sum to n_samples
    weights = weights * len(weights) / weights.sum()
    
    return weights
