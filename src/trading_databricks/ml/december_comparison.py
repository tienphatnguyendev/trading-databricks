"""December BTCUSDT variance comparison with a fixed, final-week holdout."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import NamedTuple

import polars as pl
from sklearn.linear_model import GammaRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from trading_databricks.ml.december_persistence import score_persistence

_FEATURES = ("rv_1h", "rv_6h_mean", "rv_24h_mean")
_MONTH_START = datetime(2025, 12, 1, tzinfo=UTC)
_VALIDATION_START = datetime(2025, 12, 18, tzinfo=UTC)
_TEST_START = datetime(2025, 12, 25, tzinfo=UTC)
_MONTH_END = datetime(2026, 1, 1, tzinfo=UTC)
_HOUR = timedelta(hours=1)
_DAY = timedelta(days=1)
_BOOSTING_RATE = 0.05
_SEED = 42
_SCORE_COLUMNS = ("predicted_variance", "qlike", "absolute_volatility_error_bps")


class ComparisonResult(NamedTuple):
    """Candidate validation scores, two selected rows, and paired test forecasts."""

    validation_scores: pl.DataFrame
    selection: pl.DataFrame
    test_predictions: pl.DataFrame


class _Candidate(NamedTuple):
    role: str
    model: str
    max_depth: int | None = None
    n_estimators: int | None = None


_CANDIDATES = (
    _Candidate("baseline", "persistence"),
    _Candidate("baseline", "ewma_6h"),
    _Candidate("baseline", "har_gamma"),
    *(
        _Candidate("challenger", model, depth, rounds)
        for model in ("lightgbm_gamma", "xgboost_gamma")
        for depth in (1, 2)
        for rounds in (25, 75, 150)
    ),
)


def _prepare_data(dataset: pl.DataFrame) -> pl.DataFrame:
    origin = pl.col("forecast_at")
    data = (
        dataset.select(
            "forecast_at",
            "label_end",
            "max_feature_trade_at",
            *_FEATURES,
            "actual_variance",
        )
        .filter(origin.is_between(_MONTH_START, _MONTH_END, closed="left"))
        .sort("forecast_at")
    )
    if data.filter((pl.col("label_end") != origin + _HOUR).fill_null(True)).height:
        raise ValueError("Each label must end one hour after its forecast origin.")
    if data.filter(origin.diff() != _HOUR).height:
        raise ValueError("EWMA requires consecutive, unique hourly forecast origins.")
    available_features = (pl.col("max_feature_trade_at") < origin) & pl.all_horizontal(
        pl.col(*_FEATURES).is_finite() & (pl.col(*_FEATURES) >= 0)
    )
    if data.filter(~available_features.fill_null(False)).height:
        raise ValueError(
            "Forecasts require finite, nonnegative features available at origin."
        )

    # The current row's rv_1h is the completed hour ending at that origin.
    return data.with_columns(_ewma_6h=pl.col("rv_1h").ewm_mean(half_life=6, adjust=False))


def _predict_variance(
    candidate: _Candidate, training: pl.DataFrame, forecast_rows: pl.DataFrame
) -> pl.Series:
    if candidate.model == "persistence":
        return forecast_rows["rv_1h"]
    if candidate.model == "ewma_6h":
        return forecast_rows["_ewma_6h"]

    train_features = training.select(*_FEATURES).to_numpy()
    forecast_features = forecast_rows.select(*_FEATURES).to_numpy()
    outcomes = training["actual_variance"].to_numpy()
    if candidate.model == "har_gamma":
        estimator = make_pipeline(
            StandardScaler(), GammaRegressor(alpha=0.0, max_iter=1_000)
        )
        with threadpool_limits(limits=1):
            estimator.fit(train_features, outcomes)
            return pl.Series(estimator.predict(forecast_features))

    depth, rounds = candidate.max_depth, candidate.n_estimators
    assert depth is not None and rounds is not None
    if candidate.model == "lightgbm_gamma":
        from lightgbm import Dataset, train

        booster = train(
            {
                "objective": "gamma",
                "learning_rate": _BOOSTING_RATE,
                "max_depth": depth,
                "num_leaves": 2**depth,
                "num_threads": 1,
                "seed": _SEED,
                "deterministic": True,
                "force_col_wise": True,
                "verbosity": -1,
            },
            Dataset(train_features, label=outcomes),
            num_boost_round=rounds,
        )
        return pl.Series(booster.predict(forecast_features, num_threads=1))

    from xgboost import XGBRegressor

    estimator = XGBRegressor(
        objective="reg:gamma",
        tree_method="hist",
        learning_rate=_BOOSTING_RATE,
        max_depth=depth,
        n_estimators=rounds,
        n_jobs=1,
        random_state=_SEED,
    )
    estimator.fit(train_features, outcomes)
    return pl.Series(estimator.predict(forecast_features), dtype=pl.Float64)


def _score_model(
    candidate: _Candidate, training: pl.DataFrame, forecast_rows: pl.DataFrame
) -> pl.DataFrame:
    prediction = _predict_variance(candidate, training, forecast_rows)
    actual = forecast_rows["actual_variance"]
    if not (actual.is_finite() & (actual >= 0)).fill_null(False).all():
        raise ValueError("Evaluation requires finite, nonnegative realized variance.")
    if not (prediction.is_finite() & (prediction >= 0)).fill_null(False).all():
        raise ValueError(f"{candidate.model} produced an invalid variance forecast.")
    # Use the persistence benchmark's QLIKE, units, and forecast floor for every model.
    return score_persistence(
        pl.DataFrame({"rv_1h": prediction, "actual_variance": actual})
    ).select(*_SCORE_COLUMNS)


def compare_models(dataset: pl.DataFrame) -> ComparisonResult:
    """Rank candidates on seven daily refits, then evaluate only the two winners."""
    data = _prepare_data(dataset)
    origin = pl.col("forecast_at")
    label_end = pl.col("label_end")
    folds = [
        (
            data.filter(label_end <= cutoff),
            data.filter(origin.is_between(cutoff, cutoff + _DAY, closed="left")),
        )
        for cutoff in (_VALIDATION_START + day * _DAY for day in range(7))
    ]
    pretest = data.filter(label_end <= _TEST_START)
    test = data.filter(origin >= _TEST_START)
    if test.height != 168 or any(
        fit.is_empty() or val.height != 24 for fit, val in folds
    ):
        raise ValueError("Comparison requires complete validation and test weeks.")
    target = pretest["actual_variance"]
    if not (target.is_finite() & (target > 0)).fill_null(False).all():
        raise ValueError(
            "Gamma training requires finite, strictly positive target variance."
        )

    summaries = []
    for candidate in _CANDIDATES:
        scores = pl.concat(
            _score_model(candidate, training, validation)
            for training, validation in folds
        )
        metrics = scores.select(
            mean_qlike=pl.col("qlike").mean(),
            mae_bps=pl.col("absolute_volatility_error_bps").mean(),
            n_forecasts=pl.len(),
        ).row(0, named=True)
        summaries.append(candidate._asdict() | metrics)

    validation_scores = pl.DataFrame(summaries)
    ranked = validation_scores.with_row_index("_candidate").sort(
        "mean_qlike", "_candidate"
    )
    selected = [
        ranked.filter(pl.col("role") == role)["_candidate"][0]
        for role in ("baseline", "challenger")
    ]
    selection = validation_scores[selected]
    predictions = test.select("forecast_at", "label_end", "actual_variance")
    for index in selected:
        candidate = _CANDIDATES[index]
        role = candidate.role
        scores = _score_model(candidate, pretest, test)
        predictions = predictions.hstack(
            scores.rename(
                {
                    "predicted_variance": f"{role}_variance",
                    "qlike": f"{role}_qlike",
                    "absolute_volatility_error_bps": f"{role}_abs_vol_error_bps",
                }
            )
        )

    return ComparisonResult(
        validation_scores,
        selection,
        predictions.select(
            "forecast_at",
            "label_end",
            "actual_variance",
            "baseline_variance",
            "challenger_variance",
            "baseline_qlike",
            "challenger_qlike",
            "baseline_abs_vol_error_bps",
            "challenger_abs_vol_error_bps",
        ),
    )
