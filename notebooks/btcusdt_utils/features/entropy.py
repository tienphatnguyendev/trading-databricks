"""Entropy features on binary return sequences.

Computes rolling Shannon entropy and Lempel-Ziv complexity
to measure predictability and structural randomness of returns.

All functions operate on binary-encoded returns (1 if positive,
0 otherwise) and use pure NumPy — these algorithms are inherently
sequential and have no Polars equivalent.

Reference: AFML Ch. 18–20; Phase 4B §2 of BTCUSDT EDA notebook.
"""

from __future__ import annotations

import numpy as np

from btcusdt_utils.features._config import EntropyConfig


# ── Private helpers ─────────────────────────────────────────────

def _rolling_shannon_entropy(binary: np.ndarray, w: int) -> np.ndarray:
    """Rolling Shannon entropy (bits) on a binary sequence.

    Returns NaN for the first ``w - 1`` elements (warm-up period).
    A window of all-same bits yields 0.0 (perfectly predictable).
    """
    n = len(binary)
    out = np.full(n, np.nan)
    for i in range(w - 1, n):
        p = binary[i - w + 1 : i + 1].sum() / w
        if 0 < p < 1:
            out[i] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        else:
            out[i] = 0.0
    return out


def _lempel_ziv_complexity(binary_seq: np.ndarray) -> float:
    """Normalized Lempel-Ziv complexity of a binary sequence.

    Returns c(n) / (n / log2(n)), where c(n) is the number of
    distinct phrases in the LZ76 decomposition.
    """
    n = len(binary_seq)
    if n == 0:
        return 0.0
    phrases: set[str] = set()
    current = ""
    for bit in binary_seq:
        current += str(bit)
        if current not in phrases:
            phrases.add(current)
            current = ""
    c = len(phrases) + (1 if current else 0)
    denom = n / max(np.log2(n), 1.0)
    return c / denom if denom > 0 else 0.0


def _rolling_lz_complexity(binary: np.ndarray, w: int) -> np.ndarray:
    """Rolling Lempel-Ziv complexity over a sliding window."""
    n = len(binary)
    out = np.full(n, np.nan)
    for i in range(w - 1, n):
        out[i] = _lempel_ziv_complexity(binary[i - w + 1 : i + 1])
    return out


# ── Public API ────────────────────────────────────────────────

def compute_entropy_features(
    log_ret: np.ndarray,
    *,
    config: EntropyConfig | None = None,
) -> dict[str, np.ndarray]:
    """Compute all entropy features from log returns.

    Parameters
    ----------
    log_ret : 1-D array of log returns.
    config  : Entropy configuration (keyword-only).  Uses
              ``EntropyConfig()`` defaults when ``None``.

    Returns
    -------
    dict mapping feature name → 1-D NumPy array.
    """
    if config is None:
        config = EntropyConfig()

    binary_ret = (log_ret > 0).astype(np.int8)
    arrays: dict[str, np.ndarray] = {}

    for w in config.shannon_windows:
        arrays[f"shannon_{w}"] = _rolling_shannon_entropy(binary_ret, w)

    for w in config.lz_windows:
        arrays[f"lz_complexity_{w}"] = _rolling_lz_complexity(binary_ret, w)

    return arrays
