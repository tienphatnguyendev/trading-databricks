from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
import scipy.stats

logger = logging.getLogger(__name__)


def compute_psr(
    sharpe: float,
    returns: np.ndarray,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio (AFML Ch. 14).

    P(SR* > benchmark_sharpe | sample SR = sharpe).

    Uses raw kurtosis (fisher=False): normal distribution has γ₄=3.
    """
    t = len(returns)
    if t < 2:
        return 1.0 if sharpe > benchmark_sharpe else 0.0

    gamma_3 = float(scipy.stats.skew(returns))
    gamma_4 = float(scipy.stats.kurtosis(returns, fisher=False))  # raw kurtosis

    # Raw kurtosis (fisher=False): γ₄=3 for normal distribution.
    # AFML Ch. 14 formula uses (γ₄ - 1), NOT (γ₄ - 3).
    variance_sr = (
        1.0 - gamma_3 * sharpe + (gamma_4 - 1.0) / 4.0 * sharpe**2
    ) / (t - 1)

    sigma_sr = 1e-10 if variance_sr <= 0.0 else float(np.sqrt(variance_sr))

    if sigma_sr < 1e-10:
        return 1.0 if sharpe > benchmark_sharpe else 0.0

    z = (sharpe - benchmark_sharpe) / sigma_sr
    return float(scipy.stats.norm.cdf(z))


def compute_dsr(
    sharpe: float,
    returns: np.ndarray,
    n_trials: int,
) -> float:
    """Deflated Sharpe Ratio (AFML Ch. 14).

    Adjusts PSR for multiple testing by using E[max(SR)] as benchmark.
    """
    euler_mascheroni = 0.5772156649

    if n_trials <= 1:
        if n_trials < 1:
            logger.warning("n_trials=%d < 1; DSR = PSR with benchmark=0.", n_trials)
        e_max_sr = 0.0
    else:
        e_max_sr = float(
            (1.0 - euler_mascheroni) * scipy.stats.norm.ppf(1.0 - 1.0 / n_trials)
            + euler_mascheroni * scipy.stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        )

    return compute_psr(sharpe, returns, benchmark_sharpe=e_max_sr)


def compute_strategy_failure_prob(
    sharpe: float,
    returns: np.ndarray,
) -> float:
    """P(strategy is unprofitable) = 1 - PSR(SR* = 0) (AFML Ch. 15)."""
    return 1.0 - compute_psr(sharpe, returns, benchmark_sharpe=0.0)


def _check_cost_robustness(
    path_returns: np.ndarray,
    estimated_cost_per_trade: float = 0.001,
) -> bool:
    """Strategy profitable at 2× estimated costs."""
    cost_adjusted = path_returns - 2.0 * estimated_cost_per_trade
    return bool(float(np.mean(cost_adjusted)) > 0.0)


def evaluate_promotion_gates(
    cpcv_path_results: list[dict],
    meta_label_metrics: dict,
    trial_count: int,
) -> dict:
    """Evaluate all 10 promotion gates (AFML methodology).

    Args:
        cpcv_path_results: List of 5 dicts, each with:
            - returns: np.ndarray of per-bar returns
            - sharpe: annualised Sharpe ratio (float)
            - max_drawdown: max drawdown fraction (float, positive)
            - time_under_water: days under water (int/float)
        meta_label_metrics: Dict with precision, recall, f1, logloss (floats).
        trial_count: Number of models evaluated so far (for DSR).

    Returns:
        Dict compatible with PROMOTION_EVIDENCE_SCHEMA fields (excluding
        run_id, model_type, model_version, dataset_version — those are
        set by the training pipeline).
    """
    if not cpcv_path_results:
        raise ValueError("cpcv_path_results must contain at least one path result")

    if trial_count < 1:
        logger.warning("trial_count=%d; DSR = PSR (first trial).", trial_count)

    all_returns = np.concatenate([p["returns"] for p in cpcv_path_results])
    mean_sharpe = float(np.mean([p["sharpe"] for p in cpcv_path_results]))
    max_dd = float(max(p["max_drawdown"] for p in cpcv_path_results))
    max_tuw = float(max(p["time_under_water"] for p in cpcv_path_results))

    psr_val = compute_psr(mean_sharpe, all_returns)
    dsr_val = compute_dsr(mean_sharpe, all_returns, n_trials=max(trial_count, 1))
    sfp_val = compute_strategy_failure_prob(mean_sharpe, all_returns)
    cost_ok = _check_cost_robustness(all_returns)

    gate_results = {
        "psr": psr_val > 0.95,
        "dsr": dsr_val > 0.90,
        "max_drawdown": max_dd < 0.25,
        "time_under_water": max_tuw < 90,
        "cost_robust": cost_ok,
        "strategy_failure_prob": sfp_val < 0.10,
        "meta_precision": meta_label_metrics["precision"] > 0.60,
        "meta_recall": meta_label_metrics["recall"] > 0.50,
        "meta_f1": meta_label_metrics["f1"] > 0.55,
        "meta_logloss": meta_label_metrics["logloss"] < 0.65,
    }

    all_pass = all(gate_results.values())
    failure_reasons = ", ".join(k for k, v in gate_results.items() if not v) or None

    return {
        "psr": psr_val,
        "dsr": dsr_val,
        "max_drawdown": max_dd,
        "time_under_water": int(max_tuw),
        "cost_robust": cost_ok,
        "strategy_failure_prob": sfp_val,
        "meta_precision": float(meta_label_metrics["precision"]),
        "meta_recall": float(meta_label_metrics["recall"]),
        "meta_f1": float(meta_label_metrics["f1"]),
        "meta_logloss": float(meta_label_metrics["logloss"]),
        "all_gates_pass": all_pass,
        "failure_reasons": failure_reasons,
        "evaluated_at": datetime.now(UTC),
    }
