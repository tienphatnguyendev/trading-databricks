from __future__ import annotations

import numpy as np

from trading_databricks.ml.training.promotion import (
    compute_dsr,
    compute_psr,
    compute_strategy_failure_prob,
    evaluate_promotion_gates,
)


def test_psr_known_distribution() -> None:
    """PSR with known SR and normal returns should match analytic value."""
    # Normal returns: mean=0.02, std=0.1 → SR=0.2
    rng = np.random.default_rng(0)
    returns = rng.normal(loc=0.02, scale=0.1, size=252)
    sr = float(np.mean(returns) / np.std(returns, ddof=1))
    psr = compute_psr(sr, returns, benchmark_sharpe=0.0)

    assert 0.0 <= psr <= 1.0
    # With SR=0.2 and T=252, PSR should be significantly above 0.5
    assert psr > 0.5, f"PSR={psr} expected > 0.5 for positive SR"


def test_psr_zero_sharpe_gives_point_five() -> None:
    """SR=0 vs benchmark=0 → PSR ≈ 0.5 (symmetric)."""
    rng = np.random.default_rng(1)
    returns = rng.normal(loc=0.0, scale=0.1, size=252)
    # Force SR=0 exactly
    returns = returns - returns.mean()
    psr = compute_psr(0.0, returns, benchmark_sharpe=0.0)
    assert abs(psr - 0.5) < 0.05, f"PSR={psr} expected ≈ 0.5"


def test_dsr_higher_than_psr_with_multiple_trials() -> None:
    """DSR is more conservative than PSR (accounts for multiple testing)."""
    rng = np.random.default_rng(2)
    returns = rng.normal(loc=0.02, scale=0.1, size=252)
    sr = float(np.mean(returns) / np.std(returns, ddof=1))
    psr = compute_psr(sr, returns, benchmark_sharpe=0.0)
    dsr = compute_dsr(sr, returns, n_trials=50)
    # DSR < PSR because DSR uses a higher benchmark (expected max SR)
    assert dsr < psr, f"DSR={dsr} should be < PSR={psr}"
    assert 0.0 <= dsr <= 1.0


def test_dsr_with_one_trial_equals_psr() -> None:
    """n_trials=1 → e_max_sr=0 → DSR = PSR(SR, benchmark=0)."""
    rng = np.random.default_rng(3)
    returns = rng.normal(loc=0.01, scale=0.1, size=100)
    sr = float(np.mean(returns) / np.std(returns, ddof=1))
    psr = compute_psr(sr, returns, benchmark_sharpe=0.0)
    dsr = compute_dsr(sr, returns, n_trials=1)
    assert abs(psr - dsr) < 1e-9, f"PSR={psr} != DSR={dsr} for n_trials=1"


def test_strategy_failure_prob_is_complement_of_psr() -> None:
    """P(failure) = 1 - PSR(SR, benchmark=0)."""
    rng = np.random.default_rng(4)
    returns = rng.normal(loc=0.01, scale=0.1, size=200)
    sr = float(np.mean(returns) / np.std(returns, ddof=1))
    psr = compute_psr(sr, returns, benchmark_sharpe=0.0)
    sfp = compute_strategy_failure_prob(sr, returns)
    assert abs(sfp - (1.0 - psr)) < 1e-9


def _make_good_path(n: int = 252) -> dict:
    """Construct a path result that passes all gates."""
    rng = np.random.default_rng(99)
    returns = rng.normal(loc=0.003, scale=0.01, size=n)  # SR ≈ 0.3 * sqrt(252)
    sharpe = float(np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(252)
    return {
        "returns": returns,
        "sharpe": sharpe,
        "max_drawdown": 0.05,   # 5% < 25% gate
        "time_under_water": 30,  # 30 days < 90 gate
    }


def _make_good_meta_metrics() -> dict:
    return {
        "precision": 0.65,   # > 0.60
        "recall": 0.55,      # > 0.50
        "f1": 0.60,          # > 0.55
        "logloss": 0.60,     # < 0.65
    }


def test_all_gates_pass_for_good_strategy() -> None:
    """Well-behaved strategy → all_gates_pass=True."""
    paths = [_make_good_path() for _ in range(5)]
    meta = _make_good_meta_metrics()
    result = evaluate_promotion_gates(paths, meta, trial_count=1)

    assert result["all_gates_pass"] is True
    assert result["failure_reasons"] is None or result["failure_reasons"] == ""


def test_max_drawdown_gate_fails() -> None:
    """Exceeding max drawdown threshold → all_gates_pass=False."""
    paths = [_make_good_path() for _ in range(5)]
    paths[0]["max_drawdown"] = 0.30  # exceeds 0.25 gate
    meta = _make_good_meta_metrics()
    result = evaluate_promotion_gates(paths, meta, trial_count=1)

    assert result["all_gates_pass"] is False
    assert "max_drawdown" in result["failure_reasons"]


def test_meta_precision_gate_fails() -> None:
    """Low meta precision → gate fails, included in failure_reasons."""
    paths = [_make_good_path() for _ in range(5)]
    meta = _make_good_meta_metrics()
    meta["precision"] = 0.50  # below 0.60 gate
    result = evaluate_promotion_gates(paths, meta, trial_count=1)

    assert result["all_gates_pass"] is False
    assert "meta_precision" in result["failure_reasons"]


def test_result_has_promotion_evidence_schema_keys() -> None:
    """Result dict has all required keys from PROMOTION_EVIDENCE_SCHEMA."""
    from datetime import datetime
    paths = [_make_good_path() for _ in range(5)]
    result = evaluate_promotion_gates(paths, _make_good_meta_metrics(), trial_count=5)

    required_keys = {
        "psr", "dsr", "max_drawdown", "time_under_water",
        "cost_robust", "strategy_failure_prob",
        "meta_precision", "meta_recall", "meta_f1", "meta_logloss",
        "all_gates_pass", "failure_reasons", "evaluated_at",
    }
    assert required_keys.issubset(set(result.keys()))
    assert isinstance(result["evaluated_at"], datetime)


def test_dsr_with_zero_trials_equals_psr() -> None:
    """n_trials=0 (first model ever) -> DSR = PSR with benchmark=0."""
    rng = np.random.default_rng(5)
    returns = rng.normal(loc=0.01, scale=0.1, size=100)
    sr = float(np.mean(returns) / np.std(returns, ddof=1))
    psr = compute_psr(sr, returns, benchmark_sharpe=0.0)
    dsr = compute_dsr(sr, returns, n_trials=0)
    assert abs(psr - dsr) < 1e-9, f"PSR={psr} and DSR={dsr} should be equal for n_trials=0"


def test_zero_returns_fails_psr_gate() -> None:
    """All-zero returns -> SR=0, PSR~0.5, fails PSR>0.95 gate."""
    zero_path = {
        "returns": np.zeros(252),
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "time_under_water": 0,
    }
    paths = [zero_path for _ in range(5)]
    meta = _make_good_meta_metrics()
    result = evaluate_promotion_gates(paths, meta, trial_count=1)
    assert result["all_gates_pass"] is False
    assert "psr" in result["failure_reasons"]


def test_psr_with_single_observation() -> None:
    """T=1 -> PSR = 1.0 if sharpe > benchmark else 0.0."""
    returns = np.array([0.05])
    assert compute_psr(0.1, returns, benchmark_sharpe=0.0) == 1.0
    assert compute_psr(-0.1, returns, benchmark_sharpe=0.0) == 0.0
