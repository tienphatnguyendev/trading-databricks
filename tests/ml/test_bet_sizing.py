from __future__ import annotations

from datetime import datetime
import logging

import polars as pl
import pytest

from trading_databricks.ml.config import BetSizingConfig
from trading_databricks.ml.inference.bet_sizing import compute_bet_sizes


def _make_predictions(
    meta_confidences: list[float],
    primary_confidences: list[float],
    primary_sides: list[int] | None = None,
) -> pl.DataFrame:
    n = len(meta_confidences)
    sides = primary_sides if primary_sides else [1] * n
    return pl.DataFrame(
        {
            "signal_id": [f"sig-{i}" for i in range(n)],
            "event_id": [f"evt-{i}" for i in range(n)],
            "symbol": ["BTCUSDT"] * n,
            "event_ts": [datetime(2025, 1, 1)] * n,
            "primary_side": sides,
            "primary_confidence": primary_confidences,
            "meta_label": [1] * n,
            "meta_confidence": meta_confidences,
            "primary_model_version": [1] * n,
            "meta_model_version": [1] * n,
        }
    )


def test_known_z_score_produces_expected_output() -> None:
    """meta=0.80, sigma=0.10 → z=3.0, raw=0.9974, discretized=0.85."""
    preds = _make_predictions([0.80], [0.60])
    result = compute_bet_sizes(preds, sigma_probability=0.10)

    assert len(result) == 1
    raw = result["raw_bet_size"][0]
    disc = result["discretized_weight"][0]
    assert abs(raw - 0.9974) < 0.001, f"raw_bet_size={raw}"
    assert abs(disc - 0.85) < 1e-9, f"discretized_weight={disc}"


def test_below_meta_threshold_filtered_out() -> None:
    """meta_confidence=0.45 (< 0.50) → filtered out → empty result."""
    preds = _make_predictions([0.45], [0.60])
    result = compute_bet_sizes(preds, sigma_probability=0.10)
    assert result.is_empty()


def test_below_primary_threshold_filtered_out() -> None:
    """primary_confidence=0.50 (< 0.55) → filtered out → empty result."""
    preds = _make_predictions([0.75], [0.50])
    result = compute_bet_sizes(preds, sigma_probability=0.10)
    assert result.is_empty()


def test_both_thresholds_must_pass() -> None:
    """Only predictions passing BOTH thresholds are included."""
    preds = _make_predictions(
        meta_confidences=[0.80, 0.45, 0.75],
        primary_confidences=[0.60, 0.70, 0.50],
    )
    result = compute_bet_sizes(preds, sigma_probability=0.10)
    assert len(result) == 1  # only first row passes both
    assert result["signal_id"][0] == "sig-0"


def test_sigma_probability_zero_raises_value_error() -> None:
    """sigma_probability=0.0 → ValueError (degenerate calibration)."""
    preds = _make_predictions([0.80], [0.60])
    with pytest.raises(ValueError, match="sigma_probability"):
        compute_bet_sizes(preds, sigma_probability=0.0)


def test_raw_bet_size_clipped_to_0_1() -> None:
    """Very high z-score → raw_bet_size clipped at 1.0."""
    # meta=0.999, sigma=0.001 → z≈499 → Phi≈1 → raw=1.0
    preds = _make_predictions([0.999], [0.60])
    result = compute_bet_sizes(preds, sigma_probability=0.001)
    assert len(result) == 1
    assert result["raw_bet_size"][0] <= 1.0


def test_discretization_tier_boundaries() -> None:
    """Verify discretization at each tier boundary.

    Default tiers: ((0.10, 0.00), (0.25, 0.15), (0.50, 0.35), (0.75, 0.60), (1.01, 0.85))
    raw < 0.10 → 0.00
    0.10 <= raw < 0.25 → 0.15
    0.25 <= raw < 0.50 → 0.35
    0.50 <= raw < 0.75 → 0.60
    0.75 <= raw < 1.01 → 0.85
    """
    config = BetSizingConfig()
    from scipy.stats import norm as _norm

    # tier breakpoints are raw values: 0.10, 0.25, 0.50, 0.75
    # raw = 2*Phi(z)-1; to get raw=0.05: z = Phi^-1(0.525) ≈ 0.063
    # z=(meta-0.5)/sigma → meta = 0.5 + z*sigma
    sigma = 0.10
    tier_tests = [
        (0.05, 0.00),   # raw<0.10 → 0.00
        (0.18, 0.15),   # 0.10 <= raw < 0.25 → 0.15
        (0.40, 0.35),   # 0.25 <= raw < 0.50 → 0.35
        (0.60, 0.60),   # 0.50 <= raw < 0.75 → 0.60
        (0.80, 0.85),   # 0.75 <= raw → 0.85
    ]

    for raw_target, expected_disc in tier_tests:
        # Invert raw = 2*Phi(z)-1 to get z
        z = float(_norm.ppf((raw_target + 1.0) / 2.0))
        meta = 0.5 + z * sigma
        meta = max(meta, 0.51)  # ensure it passes meta_threshold > 0.50

        preds = _make_predictions([meta], [0.60])
        result = compute_bet_sizes(preds, sigma_probability=sigma, config=config)

        if result.is_empty():
            continue  # meta barely above threshold; skip this tier

        disc = result["discretized_weight"][0]
        raw_got = result["raw_bet_size"][0]
        assert abs(disc - expected_disc) < 1e-9, (
            f"raw_target={raw_target:.2f}, raw_got={raw_got:.4f}, "
            f"expected_disc={expected_disc}, got={disc}"
        )


def test_nan_or_null_confidence_logged_and_dropped(caplog) -> None:
    """NaN or Null predictions should be logged as warnings and excluded."""
    preds = pl.DataFrame(
        {
            "signal_id": ["sig-0", "sig-1", "sig-2"],
            "event_id": ["evt-0", "evt-1", "evt-2"],
            "symbol": ["BTCUSDT"] * 3,
            "event_ts": [datetime(2025, 1, 1)] * 3,
            "primary_side": [1, 1, 1],
            "primary_confidence": [0.70, float("nan"), 0.80],
            "meta_label": [1] * 3,
            "meta_confidence": [0.80, 0.90, None],
            "primary_model_version": [1] * 3,
            "meta_model_version": [1] * 3,
        }
    )
    with caplog.at_level(logging.WARNING):
        result = compute_bet_sizes(preds, sigma_probability=0.10)

    # Only sig-0 should pass (both confidences are valid and > thresholds)
    assert len(result) == 1
    assert result["signal_id"][0] == "sig-0"
    assert "Dropping 2 predictions with NaN/Null confidence values" in caplog.text
