from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from importlib import import_module

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from trading_databricks.ml import december_comparison

_VALIDATION_START = datetime(2025, 12, 18, tzinfo=UTC)
_TEST_START = datetime(2025, 12, 25, tzinfo=UTC)
_HOUR = timedelta(hours=1)


@pytest.fixture(scope="module")
def hourly_dataset() -> pl.DataFrame:
    first_origin = datetime(2025, 12, 2, 1, tzinfo=UTC)
    origins = [first_origin + i * _HOUR for i in range(719)]
    hourly_variances = [
        (4.0 + math.sin(i / 7) + 0.4 * math.cos(i / 3)) * 1e-6 for i in range(744)
    ]
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * len(origins),
            "forecast_at": origins,
            "label_end": [origin + _HOUR for origin in origins],
            "max_feature_trade_at": [origin - timedelta(seconds=1) for origin in origins],
            "rv_1h": hourly_variances[24:-1],
            "rv_6h_mean": [
                sum(hourly_variances[i - 5 : i + 1]) / 6 for i in range(24, 743)
            ],
            "rv_24h_mean": [
                sum(hourly_variances[i - 23 : i + 1]) / 24 for i in range(24, 743)
            ],
            "actual_variance": hourly_variances[25:],
        }
    ).with_columns(
        split=pl.when(pl.col("forecast_at") < _VALIDATION_START)
        .then(pl.lit("train"))
        .when(pl.col("forecast_at") < _TEST_START)
        .then(pl.lit("validation"))
        .otherwise(pl.lit("test"))
    )


def test_compare_models_freezes_the_validation_winners(
    hourly_dataset: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[datetime, int, str]] = []

    def forecast_constant(
        candidate: december_comparison._Candidate,
        training: pl.DataFrame,
        forecast_rows: pl.DataFrame,
    ) -> pl.Series:
        cutoff = forecast_rows["forecast_at"][0]
        assert training["label_end"].max() == cutoff
        calls.append((cutoff, training.height, candidate.model))
        prediction = 8e-6
        if candidate.model == "ewma_6h":
            prediction = 4e-6
        if (
            candidate.model == "xgboost_gamma"
            and candidate.max_depth == 1
            and candidate.n_estimators == 75
        ):
            prediction = 4e-6
        return pl.Series([prediction] * forecast_rows.height)

    monkeypatch.setattr(december_comparison, "_predict_variance", forecast_constant)

    result = december_comparison.compare_models(hourly_dataset.reverse())

    assert result.validation_scores.height == 15
    assert result.validation_scores["n_forecasts"].unique().to_list() == [168]
    assert [
        (row["role"], row["model"], row["max_depth"], row["n_estimators"])
        for row in result.selection.to_dicts()
    ] == [
        ("baseline", "ewma_6h", None, None),
        ("challenger", "xgboost_gamma", 1, 75),
    ]
    assert {
        (cutoff, count) for cutoff, count, _model in calls if cutoff < _TEST_START
    } == {(_VALIDATION_START + timedelta(days=day), 383 + 24 * day) for day in range(7)}
    assert len(calls) == 107
    assert [
        (count, model) for cutoff, count, model in calls if cutoff == _TEST_START
    ] == [
        (551, "ewma_6h"),
        (551, "xgboost_gamma"),
    ]

    changed_test_labels = hourly_dataset.with_columns(
        pl.when(pl.col("forecast_at") >= _TEST_START)
        .then(pl.lit(8e-6))
        .otherwise(pl.col("actual_variance"))
        .alias("actual_variance")
    )
    changed = december_comparison.compare_models(changed_test_labels)

    assert_frame_equal(result.validation_scores, changed.validation_scores)
    assert_frame_equal(result.selection, changed.selection)
    assert result.test_predictions.height == 168
    assert result.test_predictions["forecast_at"][0] == _TEST_START
    assert changed.test_predictions["baseline_variance"][0] == 4e-6
    assert changed.test_predictions["challenger_variance"][0] == 4e-6
    assert changed.test_predictions["baseline_qlike"][0] == pytest.approx(
        math.log(4e-6) + 2
    )
    assert changed.test_predictions["challenger_abs_vol_error_bps"][0] == pytest.approx(
        (math.sqrt(8e-6) - math.sqrt(4e-6)) * 10_000
    )


def test_compare_models_rejects_an_unmatured_label_for_the_horizon(
    hourly_dataset: pl.DataFrame,
) -> None:
    invalid = hourly_dataset.with_columns(
        pl.when(pl.col("forecast_at") == _VALIDATION_START - _HOUR)
        .then(pl.col("label_end") + _HOUR)
        .otherwise(pl.col("label_end"))
        .alias("label_end")
    )

    with pytest.raises(ValueError, match="one hour"):
        december_comparison.compare_models(invalid)


def test_compare_models_runs_real_gamma_models(hourly_dataset: pl.DataFrame) -> None:
    pytest.importorskip("xgboost")
    try:
        import_module("lightgbm")
    except OSError as exc:
        pytest.skip(f"LightGBM native runtime is unavailable: {exc}")

    result = december_comparison.compare_models(hourly_dataset)

    all_boosters = result.validation_scores.filter(pl.col("role") == "challenger")
    assert dict(all_boosters.group_by("model").len().iter_rows()) == {
        "lightgbm_gamma": 6,
        "xgboost_gamma": 6,
    }
    assert result.validation_scores["mean_qlike"].is_finite().all()
    assert result.test_predictions.filter(
        (pl.col("baseline_variance") <= 0) | (pl.col("challenger_variance") <= 0)
    ).is_empty()


def test_ewma_applies_only_completed_hours_and_has_six_hour_half_life(
    hourly_dataset: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_spike = _TEST_START + _HOUR
    spike_times = [_VALIDATION_START, test_spike]
    dataset = hourly_dataset.with_columns(
        rv_1h=pl.when(pl.col("forecast_at").is_in(spike_times))
        .then(1e-5)
        .otherwise(4e-6),
        actual_variance=pl.when(pl.col("label_end").is_in(spike_times))
        .then(1e-5)
        .otherwise(4e-6),
    )
    predict_original = december_comparison._predict_variance

    def forecast_with_real_temporal_baselines(
        candidate: december_comparison._Candidate,
        training: pl.DataFrame,
        forecast_rows: pl.DataFrame,
    ) -> pl.Series:
        if candidate.model in {"persistence", "ewma_6h"}:
            return predict_original(candidate, training, forecast_rows)
        return pl.Series([8e-6] * forecast_rows.height)

    monkeypatch.setattr(
        december_comparison, "_predict_variance", forecast_with_real_temporal_baselines
    )

    result = december_comparison.compare_models(dataset)

    assert result.selection["model"][0] == "ewma_6h"
    first, spike, after_six_hours = result.test_predictions.filter(
        pl.col("forecast_at").is_in([_TEST_START, test_spike, test_spike + 6 * _HOUR])
    )["baseline_variance"]
    step = (1 - 2 ** (-1 / 6)) * 6e-6
    assert first == pytest.approx(4e-6)
    assert spike == pytest.approx(4e-6 + step)
    assert after_six_hours == pytest.approx(4e-6 + step / 2)
