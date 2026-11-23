"""Stationarity tests and fractional differentiation (AFML Ch. 5)."""

from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss as _kpss


# ── Stationarity tests ────────────────────────────────────────

def adf_test(series: np.ndarray, name: str = "") -> dict:
    """Run ADF test and return structured results."""
    series = series[np.isfinite(series)]
    result = adfuller(series, maxlag=None, autolag="AIC")
    return {
        "name": name,
        "adf_stat": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "stationary": result[1] < 0.01,
    }


def kpss_test(series: np.ndarray, name: str = "") -> dict:
    """Run KPSS test (null = stationary) and return structured results."""
    series = series[np.isfinite(series)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_value, n_lags, crit = _kpss(series, regression="c", nlags="auto")
    return {
        "name": name,
        "kpss_stat": stat,
        "p_value": p_value,
        "n_lags": n_lags,
        "stationary": p_value > 0.05,  # KPSS: fail to reject = stationary
    }


# ── FFD fractional differentiation (AFML Ch. 5.4.2) ──────────
#
# Native NumPy implementation of Fixed-Width Window (FFD)
# fractional differentiation.  `fracdiff` pip package has no
# compatible wheel for this Python version.
#
# Reference: López de Prado (2018), AFML, §5.4.2, Snippet 5.3.
# FFD uses a FIXED window of w weights (O(n) memory), unlike
# the expanding-window method which grows to O(n²).

def _ffd_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """Compute FFD weights for fractional order *d*.

    Weights are the coefficients of the binomial series (1-B)^d
    truncated when |w_k| < *threshold*.

    Returns
    -------
    1-D array of weights, w[0]=1, w[1]=-d, w[2]=d(d-1)/2, …
    """
    weights = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
    return np.array(weights, dtype=np.float64)


def ffd(series: np.ndarray, d: float, threshold: float = 1e-5) -> np.ndarray:
    """Apply Fixed-Width Window fractional differentiation.

    Parameters
    ----------
    series    : 1-D array of prices or feature values
    d         : fractional differentiation order, 0 < d ≤ 1
    threshold : weight truncation cutoff

    Returns
    -------
    Fractionally differentiated series (same length as input,
    first ``w-1`` values are NaN where w = number of weights).
    """
    w = _ffd_weights(d, threshold)
    width = len(w)
    n = len(series)
    result = np.full(n, np.nan, dtype=np.float64)

    for t in range(width - 1, n):
        result[t] = np.dot(w, series[t - width + 1 : t + 1][::-1])

    return result


def find_min_ffd(
    series: np.ndarray,
    *,
    d_range: np.ndarray | None = None,
    threshold: float = 1e-5,
    adf_pval: float = 0.01,
    min_corr: float = 0.90,
) -> dict:
    """Find minimum *d* that achieves stationarity (AFML §5.5).

    Sweeps *d* from 0 to 1 in steps of 0.05.  For each *d*:
      1. Apply FFD
      2. Run ADF test (want p < *adf_pval*)
      3. Check correlation with original (want > *min_corr*)

    Returns
    -------
    dict with optimal d, ADF p-value, correlation, and full sweep.
    """
    if d_range is None:
        d_range = np.arange(0.0, 1.05, 0.05)

    results: list[dict] = []
    series_clean = series[np.isfinite(series)]

    for d in d_range:
        if d == 0.0:
            ffd_series = series_clean.copy()
        else:
            ffd_series = ffd(series_clean, d, threshold)

        valid = ffd_series[np.isfinite(ffd_series)]
        if len(valid) < 50:
            continue

        adf_result = adfuller(valid, maxlag=None, autolag="AIC")
        corr = float(np.corrcoef(
            series_clean[-len(valid):], valid
        )[0, 1])

        results.append({
            "d": round(d, 2),
            "adf_stat": adf_result[0],
            "adf_pval": adf_result[1],
            "corr": corr,
            "n_valid": len(valid),
            "stationary": adf_result[1] < adf_pval,
            "memory_ok": corr >= min_corr,
        })

    # Find minimum d satisfying both constraints
    optimal_d = None
    for r in results:
        if r["stationary"] and r["memory_ok"]:
            optimal_d = r["d"]
            break

    return {
        "optimal_d": optimal_d,
        "sweep": results,
        "adf_threshold": adf_pval,
        "corr_threshold": min_corr,
    }
