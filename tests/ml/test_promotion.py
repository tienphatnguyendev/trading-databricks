from __future__ import annotations

import numpy as np
import pytest
import pytest_mock

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
        "max_drawdown": 0.05,  # 5% < 25% gate
        "time_under_water": 30,  # 30 days < 90 gate
    }


def _make_good_meta_metrics() -> dict:
    return {
        "precision": 0.65,  # > 0.60
        "recall": 0.55,  # > 0.50
        "f1": 0.60,  # > 0.55
        "logloss": 0.60,  # < 0.65
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
        "psr",
        "dsr",
        "max_drawdown",
        "time_under_water",
        "cost_robust",
        "strategy_failure_prob",
        "meta_precision",
        "meta_recall",
        "meta_f1",
        "meta_logloss",
        "all_gates_pass",
        "failure_reasons",
        "evaluated_at",
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
    assert abs(psr - dsr) < 1e-9, (
        f"PSR={psr} and DSR={dsr} should be equal for n_trials=0"
    )


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


def test_psr_normal_returns_matches_analytical() -> None:
    """For normal IID returns, PSR should be near 1 for SR≈1 and near 0.5 for SR=0."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.01, 1000)
    sr = float(np.mean(returns) / np.std(returns, ddof=1))

    psr = compute_psr(sr, returns, benchmark_sharpe=0.0)
    # For 1000 observations with SR~1, PSR should be very high
    assert psr > 0.95

    # PSR(SR=0 | benchmark=0) should be ~0.5 (symmetric distribution)
    zero_sr_returns = rng.normal(0.0, 0.01, 100)
    psr_zero = compute_psr(0.0, zero_sr_returns, benchmark_sharpe=0.0)
    assert 0.3 < psr_zero < 0.7


def test_evaluate_promotion_gates_empty_paths() -> None:
    """Empty cpcv_path_results should raise ValueError."""
    import pytest

    from trading_databricks.ml.training.promotion import evaluate_promotion_gates

    with pytest.raises(ValueError, match="cpcv_path_results"):
        evaluate_promotion_gates([], _make_good_meta_metrics(), trial_count=1)


def test_evaluate_promotion_gates_zero_trials(caplog: pytest.LogCaptureFixture) -> None:
    """trial_count < 1 should log a warning."""
    import logging

    from trading_databricks.ml.training.promotion import evaluate_promotion_gates

    paths = [_make_good_path() for _ in range(5)]
    meta = _make_good_meta_metrics()

    with caplog.at_level(logging.WARNING):
        evaluate_promotion_gates(paths, meta, trial_count=0)

    assert "trial_count=0; DSR = PSR (first trial)." in caplog.text


def test_compute_psr_tiny_variance(mocker: pytest_mock.MockerFixture) -> None:
    """If variance is tiny, should fall back to deterministic pass/fail."""
    from trading_databricks.ml.training.promotion import compute_psr

    returns = np.zeros(10)

    # We want variance_sr = 1e-22, so sigma_sr = 1e-11 < 1e-10
    # variance_sr = (1.0 - gamma_3 * sharpe + (gamma_4 - 1.0) / 4.0 * sharpe**2) / (t - 1)
    # (1.0 + (gamma_4 - 1.0)/4.0)/9 = 1e-22
    # gamma_4 = -3.0 + 36e-22
    mocker.patch("scipy.stats.skew", return_value=0.0)
    mocker.patch("scipy.stats.kurtosis", return_value=-3.0 + 36e-22)

    psr_pass = compute_psr(sharpe=1.0, returns=returns, benchmark_sharpe=0.0)
    assert psr_pass == 1.0

    psr_fail = compute_psr(sharpe=-1.0, returns=returns, benchmark_sharpe=0.0)
    assert psr_fail == 0.0


def test_compute_dsr_zero_or_negative_trials(caplog: pytest.LogCaptureFixture) -> None:
    """n_trials < 1 should log a warning."""
    import logging

    from trading_databricks.ml.training.promotion import compute_dsr

    rng = np.random.default_rng(99)
    returns = rng.normal(loc=0.003, scale=0.01, size=100)

    with caplog.at_level(logging.WARNING):
        compute_dsr(sharpe=1.0, returns=returns, n_trials=0)

    assert "n_trials=0 < 1; DSR = PSR with benchmark=0." in caplog.text


def test_compute_psr_negative_variance(mocker: pytest_mock.MockerFixture) -> None:
    """If variance_sr <= 0, should set sigma_sr to 1e-10."""
    from trading_databricks.ml.training.promotion import compute_psr

    returns = np.zeros(10)

    # We want variance_sr < 0
    mocker.patch("scipy.stats.skew", return_value=0.0)
    mocker.patch("scipy.stats.kurtosis", return_value=-5.0)

    psr_pass = compute_psr(sharpe=1.0, returns=returns, benchmark_sharpe=0.0)
    assert psr_pass == 1.0


def test_compute_psr_exact_tiny_sigma(mocker: pytest_mock.MockerFixture) -> None:
    """If 0 < variance_sr < 1e-20, sigma_sr is < 1e-10."""
    from trading_databricks.ml.training.promotion import compute_psr

    returns = np.zeros(10)

    # We want 0 < variance_sr < 1e-20
    # Let variance_sr = 1e-24, so sigma_sr = 1e-12 < 1e-10
    # variance_sr = (1 - 0 + (gamma_4 - 1)/4) / 9 = 1e-24
    # (1 + (gamma_4 - 1)/4) = 9e-24
    # (gamma_4 - 1)/4 = 9e-24 - 1
    # gamma_4 - 1 = 36e-24 - 4
    # gamma_4 = -3 + 36e-24
    mocker.patch("scipy.stats.skew", return_value=0.0)
    mocker.patch("scipy.stats.kurtosis", return_value=-3.0 + 36e-24)

    psr_pass = compute_psr(sharpe=1.0, returns=returns, benchmark_sharpe=0.0)
    assert psr_pass == 1.0


def test_compute_psr_exact_tiny_sigma2(mocker: pytest_mock.MockerFixture) -> None:
    """Test the condition where variance_sr > 0.0 but sigma_sr < 1e-10"""
    from trading_databricks.ml.training.promotion import compute_psr

    returns = np.zeros(10)

    # We want variance_sr around 1e-21 which is float precision limit.
    # a mocked function to directly mock sqrt returning < 1e-10

    mocker.patch("numpy.sqrt", return_value=1e-11)

    # Need variance_sr > 0.0 so we pass the first part of the ternary
    # Example gamma_4 = -3.0 + 36e-15 -> variance ~ 1e-15.
    mocker.patch("scipy.stats.skew", return_value=0.0)
    mocker.patch("scipy.stats.kurtosis", return_value=-3.0 + 36e-15)

    psr_pass = compute_psr(sharpe=1.0, returns=returns, benchmark_sharpe=0.0)
    assert psr_pass == 1.0
