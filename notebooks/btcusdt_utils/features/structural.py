"""Structural break features (CUSUM, rolling ADF, martingale).

Detects regime changes and mean-reversion breakdowns via
σ-normalized CUSUM accumulators, rolling Augmented Dickey-Fuller
test statistics (SADF-lite), and conditional transition ratios.

All functions use pure NumPy — these algorithms are inherently
sequential and have no Polars equivalent.

Reference: AFML Ch. 2.5, 17; Phase 4B §3 of BTCUSDT EDA notebook.
"""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.stattools import adfuller

from btcusdt_utils.features._config import StructuralConfig


# ── Private helpers ─────────────────────────────────────────────

def _rolling_vol(log_ret: np.ndarray, w: int) -> np.ndarray:
    """Rolling standard deviation with NaN warm-up."""
    n = len(log_ret)
    out = np.full(n, np.nan)
    for i in range(w - 1, n):
        out[i] = np.nanstd(log_ret[i - w + 1 : i + 1])
    finite = out[np.isfinite(out)]
    fill = float(np.nanmedian(finite)) if len(finite) > 0 else 1e-10
    return np.where(np.isfinite(out), out, fill)


def _cusum_series(
    returns: np.ndarray,
    vol: np.ndarray,
    reset_every: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """σ-normalized CUSUM+ and CUSUM− series.

    Monitors cumulative deviations from zero, optionally resetting
    the accumulators every *reset_every* bars to limit lookback.
    """
    n = len(returns)
    s_pos = np.zeros(n)
    s_neg = np.zeros(n)
    for i in range(1, n):
        if reset_every and (i % reset_every == 0):
            s_pos[i] = max(0.0, returns[i])
            s_neg[i] = max(0.0, -returns[i])
        else:
            s_pos[i] = max(0.0, s_pos[i - 1] + returns[i])
            s_neg[i] = max(0.0, s_neg[i - 1] - returns[i])
    vol_safe = np.where(vol > 1e-10, vol, 1e-10)
    return s_pos / vol_safe, s_neg / vol_safe


def _rolling_adf_stat(series: np.ndarray, w: int) -> np.ndarray:
    """Rolling ADF test statistic (SADF-lite) with fixed window."""
    n = len(series)
    out = np.full(n, np.nan)
    min_valid = max(5, int(w * 0.8))
    for i in range(w - 1, n):
        window = series[i - w + 1 : i + 1]
        valid = window[np.isfinite(window)]
        if len(valid) < min_valid or valid.max() == valid.min():
            continue
        try:
            result = adfuller(valid, maxlag=1, regression="c", autolag=None)
            out[i] = result[0]
        except Exception:
            pass
    return out


def _rolling_martingale_ratio(
    binary: np.ndarray,
    w: int,
) -> np.ndarray:
    """P(r_{t+1}>0 | r_t>0) / P(r_{t+1}>0) in a rolling window.

    Values > 1 indicate momentum (sub-martingale); < 1 indicates
    mean-reversion (super-martingale).
    """
    n = len(binary)
    out = np.full(n, np.nan)
    for i in range(w, n):
        window = binary[i - w : i]
        p_up = window.sum() / w
        if p_up <= 0 or p_up >= 1:
            continue
        up_count = int(np.sum(window[:-1] == 1))
        if up_count > 0:
            up_up = int(np.sum((window[:-1] == 1) & (window[1:] == 1)))
            out[i] = (up_up / up_count) / p_up
    return out


# ── Public API ────────────────────────────────────────────────

def compute_structural_features(
    log_ret: np.ndarray,
    *,
    config: StructuralConfig | None = None,
) -> dict[str, np.ndarray]:
    """Compute all structural break features from log returns.

    Parameters
    ----------
    log_ret : 1-D array of log returns.
    config  : Structural configuration (keyword-only).  Uses
              ``StructuralConfig()`` defaults when ``None``.

    Returns
    -------
    dict mapping feature name → 1-D NumPy array.
    """
    if config is None:
        config = StructuralConfig()

    arrays: dict[str, np.ndarray] = {}
    ret_clean = np.nan_to_num(log_ret, nan=0.0)
    binary_ret = (log_ret > 0).astype(np.int8)
    vol = _rolling_vol(log_ret, config.vol_window)

    # CUSUM features
    for reset in config.cusum_resets:
        suffix = "raw" if reset is None else str(reset)
        cp, cm = _cusum_series(ret_clean, vol, reset)
        arrays[f"cusum_plus_{suffix}"] = cp
        arrays[f"cusum_minus_{suffix}"] = cm

    # Rolling ADF statistics
    for w in config.adf_windows:
        arrays[f"adf_stat_{w}"] = _rolling_adf_stat(log_ret, w)

    # Martingale ratio
    for w in config.martingale_windows:
        arrays[f"martingale_ratio_{w}"] = _rolling_martingale_ratio(
            binary_ret, w,
        )

    return arrays
