"""Sample weighting and sequential bootstrap (AFML Ch. 4).

Computes per-sample uniqueness weights from overlapping triple-barrier
labels.  Labels that share price data with many neighbors receive lower
weight, reducing effective sample size inflation.

All algorithms are pure NumPy — inherently sequential with no Polars
equivalent.

Reference: López de Prado (2018), AFML, Ch. 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WeightsConfig:
    """Parameters for sample weighting and sequential bootstrap."""

    seed: int = 42


# ── Concurrency ────────────────────────────────────────────────

def compute_concurrency(
    event_starts: np.ndarray,
    event_ends: np.ndarray,
) -> np.ndarray:
    """Count of active labels at each bar index.

    Parameters
    ----------
    event_starts : 1-D int array of bar indices where each label begins.
    event_ends   : 1-D int array of bar indices where each label resolves.

    Returns
    -------
    1-D int32 array of length ``max(event_ends) + 1``.
    ``C[t]`` = number of labels active at bar *t*.
    """
    max_bar = int(event_ends.max()) + 1
    C = np.zeros(max_bar, dtype=np.int32)
    for s, e in zip(event_starts, event_ends):
        C[s : e + 1] += 1
    return C


# ── Per-sample uniqueness ──────────────────────────────────────

def compute_sample_weights(
    event_starts: np.ndarray,
    event_ends: np.ndarray,
) -> np.ndarray:
    """Per-sample uniqueness weights (AFML Eq. 4.1).

    For each label *i* spanning ``[start_i, end_i]``::

        u_i = mean(1 / C[t])  for t in [start_i, end_i]

    where ``C[t]`` is the concurrency at bar *t*.

    Parameters
    ----------
    event_starts : 1-D int array of event start bar indices.
    event_ends   : 1-D int array of event end bar indices.

    Returns
    -------
    1-D float64 array of per-sample uniqueness ∈ (0, 1].
    1.0 = fully independent; < 0.5 = highly overlapping.
    """
    C = compute_concurrency(event_starts, event_ends)
    inv_C = np.where(C > 0, 1.0 / C, 0.0)

    n_samples = len(event_starts)
    weights = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        s, e = int(event_starts[i]), int(event_ends[i])
        weights[i] = inv_C[s : e + 1].mean()

    return weights


# ── Sequential bootstrap ──────────────────────────────────────

def sequential_bootstrap(
    event_starts: np.ndarray,
    event_ends: np.ndarray,
    n_draws: int | None = None,
    *,
    config: WeightsConfig | None = None,
) -> np.ndarray:
    """Sequential bootstrap with uniqueness-aware sampling (AFML Ch. 4.5.2).

    At each draw, recompute the average uniqueness of every candidate
    given the already-drawn samples, then sample proportionally.
    Labels with less overlap to the current draw set have higher
    probability.

    Complexity: O(n_draws × n_events × avg_span).

    Parameters
    ----------
    event_starts : 1-D int array of event start bar indices.
    event_ends   : 1-D int array of event end bar indices.
    n_draws      : Number of bootstrap draws (default: ``len(event_starts)``).
    config       : WeightsConfig (keyword-only).

    Returns
    -------
    1-D int64 array of drawn sample indices.
    """
    if config is None:
        config = WeightsConfig()

    n_events = len(event_starts)
    if n_draws is None:
        n_draws = n_events

    max_bar = int(event_ends.max()) + 1
    rng = np.random.default_rng(config.seed)
    drawn = np.empty(n_draws, dtype=np.int64)

    # Running concurrency from already-drawn samples
    C_drawn = np.zeros(max_bar, dtype=np.int32)

    for step in range(n_draws):
        # Compute average uniqueness for each candidate
        avg_u = np.empty(n_events, dtype=np.float64)
        for i in range(n_events):
            s, e = int(event_starts[i]), int(event_ends[i])
            # +1 accounts for this candidate being added
            avg_u[i] = np.mean(1.0 / (C_drawn[s : e + 1] + 1))

        probs = avg_u / avg_u.sum()
        idx = rng.choice(n_events, p=probs)
        drawn[step] = idx

        # Update running concurrency
        s, e = int(event_starts[idx]), int(event_ends[idx])
        C_drawn[s : e + 1] += 1

    return drawn
