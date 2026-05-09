"""Sequential bootstrap for overlapping labels.

Implements sequential bootstrap with embargo from Lopez de Prado AFML Chapter 4.

Key concepts:
* Standard bootstrap: Sample with replacement uniformly (can oversample overlapping events)
* Sequential bootstrap: Sample with replacement respecting uniqueness
* Embargo: After selecting a sample, temporarily block nearby samples

References:
* Advances in Financial Machine Learning (Lopez de Prado, 2018), Chapter 4
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

from btcusdt_utils.sampling.weights import get_ind_matrix


def sequential_bootstrap(
    ind_matrix: pd.DataFrame,
    sample_length: Optional[int] = None,
    warmup_samples: Optional[list] = None,
    compare_with_std: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Sequential bootstrap with uniqueness.
    
    Sample with replacement from a set of events, where each sample is drawn
    with probability proportional to its average uniqueness. This reduces
    oversampling of overlapping events compared to standard bootstrap.
    
    Algorithm:
    1. Compute initial average uniqueness for each event
    2. Draw sample i with probability proportional to uniqueness
    3. Update uniqueness after drawing (overlapping events get lower probability)
    4. Repeat until sample_length samples drawn
    
    Parameters
    ----------
    ind_matrix : pd.DataFrame
        Indicator matrix (n_bars × n_samples)
        ind_matrix.loc[bar, event] = 1 if bar is in event's time span
    sample_length : int, optional
        Number of samples to draw (default: n_samples = bootstrap sample size)
    warmup_samples : list, optional
        Pre-selected samples to start with (default: None)
    compare_with_std : bool, optional
        If True, also return standard bootstrap for comparison (default: False)
    verbose : bool, optional
        If True, print progress every 10% (default: False)
    
    Returns
    -------
    seq_bootstrap_indices : np.ndarray
        Indices of sampled events (shape: sample_length)
    std_bootstrap_indices : np.ndarray, optional
        Standard bootstrap indices for comparison (only if compare_with_std=True)
    
    Complexity
    ----------
    O(sample_length × n_bars × n_samples) worst case
    In practice, O(sample_length × avg_event_length × avg_concurrency)
    
    Example
    -------
    >>> ind_matrix = pd.DataFrame({
    ...     0: [1, 1, 0],  # Event 0 spans bars 0-1
    ...     1: [1, 1, 1],  # Event 1 spans bars 0-2 (high overlap)
    ...     2: [0, 0, 1],  # Event 2 spans bar 2
    ... })
    >>> indices = sequential_bootstrap(ind_matrix, sample_length=10)
    >>> np.bincount(indices)  # Event 1 should be sampled less due to overlap
    array([4, 3, 3])  # More balanced than standard bootstrap
    """
    if sample_length is None:
        sample_length = ind_matrix.shape[1]  # n_samples
    
    if warmup_samples is None:
        warmup_samples = []
    
    # Initialize
    n_samples = ind_matrix.shape[1]
    seq_bootstrap_indices = []
    
    # Step 1: Compute initial average uniqueness
    # concurrency[t] = number of events at bar t
    concurrency = ind_matrix.sum(axis=1).values  # shape: (n_bars,)
    
    # uniqueness_matrix[t, i] = 1 / concurrency[t] if ind_matrix[t, i] = 1, else 0
    uniqueness_matrix = ind_matrix.copy()
    uniqueness_matrix = uniqueness_matrix.divide(
        pd.Series(concurrency, index=ind_matrix.index),
        axis=0,
    )
    
    # avg_uniqueness[i] = mean uniqueness across bars in event i
    avg_uniqueness = uniqueness_matrix[uniqueness_matrix > 0].mean(axis=0).values
    
    # Add warmup samples
    for sample_idx in warmup_samples:
        seq_bootstrap_indices.append(sample_idx)
        # Update average uniqueness after drawing this sample
        avg_uniqueness = _update_avg_uniqueness(
            ind_matrix, uniqueness_matrix, avg_uniqueness, sample_idx
        )
    
    # Step 2-4: Draw samples sequentially
    for i in range(len(warmup_samples), sample_length):
        # Draw sample with probability proportional to avg_uniqueness
        # Normalize to get probabilities
        probs = avg_uniqueness / avg_uniqueness.sum()
        
        # Sample
        sample_idx = np.random.choice(n_samples, p=probs)
        seq_bootstrap_indices.append(sample_idx)
        
        # Update average uniqueness
        avg_uniqueness = _update_avg_uniqueness(
            ind_matrix, uniqueness_matrix, avg_uniqueness, sample_idx
        )
        
        if verbose and (i + 1) % max(1, sample_length // 10) == 0:
            pct = 100 * (i + 1) / sample_length
            print(f"  Progress: {i+1}/{sample_length} ({pct:.0f}%)")
    
    seq_bootstrap_indices = np.array(seq_bootstrap_indices)
    
    # Standard bootstrap for comparison
    if compare_with_std:
        std_bootstrap_indices = np.random.choice(
            n_samples, size=sample_length, replace=True
        )
        return seq_bootstrap_indices, std_bootstrap_indices
    
    return seq_bootstrap_indices, None


def _update_avg_uniqueness(
    ind_matrix: pd.DataFrame,
    uniqueness_matrix: pd.DataFrame,
    avg_uniqueness: np.ndarray,
    sample_idx: int,
) -> np.ndarray:
    """Update average uniqueness after drawing a sample.
    
    When sample i is drawn, all bars in its time span get higher concurrency,
    which reduces uniqueness for all overlapping events.
    
    Parameters
    ----------
    ind_matrix : pd.DataFrame
        Indicator matrix (n_bars × n_samples)
    uniqueness_matrix : pd.DataFrame
        Current uniqueness matrix (n_bars × n_samples)
    avg_uniqueness : np.ndarray
        Current average uniqueness (shape: n_samples)
    sample_idx : int
        Index of the sample just drawn
    
    Returns
    -------
    updated_avg_uniqueness : np.ndarray
        Updated average uniqueness (shape: n_samples)
    """
    # Find bars affected by this sample
    affected_bars = ind_matrix.index[ind_matrix.iloc[:, sample_idx] > 0]
    
    # For each affected bar, reduce uniqueness for all overlapping events
    for bar in affected_bars:
        # Find all events at this bar
        events_at_bar = ind_matrix.columns[ind_matrix.loc[bar] > 0]
        
        # Increase concurrency at this bar
        # uniqueness = 1 / (concurrency + 1) for all events at this bar
        old_uniqueness = uniqueness_matrix.loc[bar, events_at_bar]
        new_concurrency = 1.0 / old_uniqueness + 1.0
        uniqueness_matrix.loc[bar, events_at_bar] = 1.0 / new_concurrency
    
    # Recompute average uniqueness for all events
    avg_uniqueness = uniqueness_matrix[uniqueness_matrix > 0].mean(axis=0).values
    
    return avg_uniqueness


def sequential_bootstrap_with_embargo(
    bar_index: np.ndarray,
    event_starts: np.ndarray,
    event_ends: np.ndarray,
    sample_length: Optional[int] = None,
    embargo_pct: float = 0.01,
    compare_with_std: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Sequential bootstrap with embargo.
    
    After drawing a sample, temporarily block nearby samples (embargo period)
    to reduce temporal correlation in the bootstrap sample.
    
    Parameters
    ----------
    bar_index : np.ndarray
        Array of bar indices for each sample (shape: n_samples)
    event_starts : np.ndarray
        Starting bar index for each event (shape: n_samples)
    event_ends : np.ndarray
        Ending bar index for each event (shape: n_samples)
    sample_length : int, optional
        Number of samples to draw (default: n_samples)
    embargo_pct : float, optional
        Embargo period as fraction of dataset size (default: 0.01 = 1%)
    compare_with_std : bool, optional
        If True, also return standard bootstrap (default: False)
    verbose : bool, optional
        If True, print progress (default: False)
    
    Returns
    -------
    seq_bootstrap_indices : np.ndarray
        Indices of sampled events (shape: sample_length)
    std_bootstrap_indices : np.ndarray, optional
        Standard bootstrap indices for comparison (only if compare_with_std=True)
    
    Example
    -------
    >>> bar_index = np.array([0, 5, 10, 15])
    >>> event_starts = np.array([0, 5, 10, 15])
    >>> event_ends = np.array([4, 9, 14, 19])
    >>> indices = sequential_bootstrap_with_embargo(
    ...     bar_index, event_starts, event_ends,
    ...     sample_length=10, embargo_pct=0.1
    ... )
    >>> len(indices)
    10
    """
    # Build indicator matrix
    ind_matrix = get_ind_matrix(bar_index, event_starts, event_ends)
    
    # Compute embargo size in bars
    n_bars = len(np.unique(np.concatenate([event_starts, event_ends, bar_index])))
    embargo_bars = int(embargo_pct * n_bars)
    
    if verbose:
        print(f"Sequential bootstrap with embargo: {embargo_bars} bars")
    
    # Initialize
    if sample_length is None:
        sample_length = len(bar_index)
    
    seq_bootstrap_indices = []
    blocked_samples = set()  # Samples currently under embargo
    
    # Compute initial average uniqueness
    concurrency = ind_matrix.sum(axis=1).values
    uniqueness_matrix = ind_matrix.copy()
    uniqueness_matrix = uniqueness_matrix.divide(
        pd.Series(concurrency, index=ind_matrix.index),
        axis=0,
    )
    avg_uniqueness = uniqueness_matrix[uniqueness_matrix > 0].mean(axis=0).values
    
    # Draw samples with embargo
    for i in range(sample_length):
        # Filter out blocked samples
        available_mask = np.ones(len(bar_index), dtype=bool)
        if blocked_samples:
            available_mask[list(blocked_samples)] = False
        
        # Draw sample with probability proportional to avg_uniqueness (only available)
        available_uniqueness = avg_uniqueness.copy()
        available_uniqueness[~available_mask] = 0.0
        
        if available_uniqueness.sum() == 0:
            # All samples blocked - clear embargo and retry
            blocked_samples.clear()
            available_uniqueness = avg_uniqueness.copy()
        
        probs = available_uniqueness / available_uniqueness.sum()
        sample_idx = np.random.choice(len(bar_index), p=probs)
        seq_bootstrap_indices.append(sample_idx)
        
        # Update average uniqueness
        avg_uniqueness = _update_avg_uniqueness(
            ind_matrix, uniqueness_matrix, avg_uniqueness, sample_idx
        )
        
        # Apply embargo: block samples within embargo_bars of this sample
        sample_end = event_ends[sample_idx]
        for j in range(len(bar_index)):
            if event_starts[j] <= sample_end + embargo_bars:
                blocked_samples.add(j)
        
        # Clear embargo for samples far enough from latest sample
        blocked_samples = {
            j for j in blocked_samples
            if event_starts[j] <= sample_end + embargo_bars
        }
        
        if verbose and (i + 1) % max(1, sample_length // 10) == 0:
            pct = 100 * (i + 1) / sample_length
            print(f"  Progress: {i+1}/{sample_length} ({pct:.0f}%)")
    
    seq_bootstrap_indices = np.array(seq_bootstrap_indices)
    
    # Standard bootstrap for comparison
    if compare_with_std:
        std_bootstrap_indices = np.random.choice(
            len(bar_index), size=sample_length, replace=True
        )
        return seq_bootstrap_indices, std_bootstrap_indices
    
    return seq_bootstrap_indices, None
