"""Triple-barrier labeling (AFML Ch. 3)."""

from __future__ import annotations

import numpy as np


def apply_triple_barrier(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    vol: np.ndarray,
    pt_mult: float,
    sl_mult: float,
    max_hold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply triple-barrier labeling via vectorized wavefront scan.

    For each event at bar i (i = 0 … n−max_hold−1), scan forward
    k = 1 … max_hold checking if high[i+k] breaches upper or
    low[i+k] breaches lower.  All events at step k are processed
    in one vectorized pass — O(max_hold × n) total.

    Barrier levels::

        upper[i] = close[i] × (1 + pt_mult × vol[i])
        lower[i] = close[i] × (1 − sl_mult × vol[i])

    Parameters
    ----------
    close    : 1-D array of bar close prices
    high     : 1-D array of bar high prices
    low      : 1-D array of bar low prices
    vol      : 1-D array of EWMA volatility estimates
    pt_mult  : profit-taking multiplier (barrier = close × (1 + pt_mult × vol))
    sl_mult  : stop-loss multiplier (barrier = close × (1 − sl_mult × vol))
    max_hold : maximum holding period (bars) — vertical barrier

    Returns
    -------
    labels       : int8  — 1 (TP win), 0 (SL loss), or vertical outcome
    touch_bars   : int32 — bars until barrier touch
    touch_types  : int8  — 0 (TP), 1 (SL), 2 (vertical)
    event_starts : int64 — bar index where event begins
    event_ends   : int64 — bar index where event resolves
    """
    n = len(close)
    n_events = n - max_hold

    if n_events <= 0:
        empty_i8 = np.array([], dtype=np.int8)
        empty_i32 = np.array([], dtype=np.int32)
        empty_i64 = np.array([], dtype=np.int64)
        return empty_i8, empty_i32, empty_i8, empty_i64, empty_i64

    # ── Barrier levels at event start ─────────────────────────
    close_ev = close[:n_events]
    vol_ev = vol[:n_events]
    upper = close_ev * (1.0 + pt_mult * vol_ev)
    lower = close_ev * (1.0 - sl_mult * vol_ev)

    # ── Output arrays (default: vertical barrier) ─────────────
    labels      = np.full(n_events, -1, dtype=np.int8)
    touch_bars  = np.full(n_events, max_hold, dtype=np.int32)
    touch_types = np.full(n_events, 2, dtype=np.int8)
    event_starts = np.arange(n_events, dtype=np.int64)
    event_ends   = event_starts + max_hold  # default: vertical

    unresolved = np.ones(n_events, dtype=bool)

    # ── Wavefront scan ────────────────────────────────────────
    for k in range(1, max_hold + 1):
        if not unresolved.any():
            break

        h_k = high[k : k + n_events]
        l_k = low[k : k + n_events]

        tp_hit = unresolved & (h_k >= upper)
        sl_hit = unresolved & (l_k <= lower)

        # TP only
        tp_only = tp_hit & ~sl_hit
        if tp_only.any():
            labels[tp_only] = 1
            touch_bars[tp_only] = k
            touch_types[tp_only] = 0
            event_ends[tp_only] = event_starts[tp_only] + k

        # SL only
        sl_only = sl_hit & ~tp_hit
        if sl_only.any():
            labels[sl_only] = 0
            touch_bars[sl_only] = k
            touch_types[sl_only] = 1
            event_ends[sl_only] = event_starts[sl_only] + k

        # Same-bar collision: deeper exceedance wins
        both = tp_hit & sl_hit
        if both.any():
            tp_excess = h_k[both] - upper[both]
            sl_excess = lower[both] - l_k[both]
            tp_wins = tp_excess >= sl_excess

            both_idx = np.where(both)[0]
            tp_win = both_idx[tp_wins]
            sl_win = both_idx[~tp_wins]

            for idx_arr, lbl, tt in [(tp_win, 1, 0), (sl_win, 0, 1)]:
                if idx_arr.size:
                    labels[idx_arr] = lbl
                    touch_bars[idx_arr] = k
                    touch_types[idx_arr] = tt
                    event_ends[idx_arr] = event_starts[idx_arr] + k

        unresolved &= ~(tp_hit | sl_hit)

    # ── Vertical barrier: close-to-close direction ────────────
    vert = labels == -1
    if vert.any():
        vi = np.where(vert)[0]
        labels[vert] = (close[vi + max_hold] >= close[vi]).astype(np.int8)

    return labels, touch_bars, touch_types, event_starts, event_ends
