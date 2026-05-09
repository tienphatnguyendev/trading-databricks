"""End-to-end training pipeline with Optuna + MLflow."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import optuna
import mlflow
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from btcusdt_utils.training._config import (
    ModelConfig, TrainResult, create_model,
    DEFAULT_N_TRIALS, DEFAULT_SEED, suppress_training_warnings,
)
from btcusdt_utils.training.objectives import (
    create_lgb_objective, create_xgb_objective,
    create_rf_objective, create_lr_objective, create_dt_objective,
)

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


def default_model_configs() -> list[ModelConfig]:
    """Return default ModelConfig list for all supported algorithms."""
    configs = [
        ModelConfig(
            name="LightGBM",
            objective_factory=create_lgb_objective,
            uses_scaled_data=False,
        ),
        ModelConfig(
            name="RandomForest",
            objective_factory=create_rf_objective,
            base_params={"random_state": DEFAULT_SEED, "n_jobs": -1},
            uses_scaled_data=False,
        ),
        ModelConfig(
            name="LogisticRegression",
            objective_factory=create_lr_objective,
            base_params={"random_state": DEFAULT_SEED, "max_iter": 1000},
            uses_scaled_data=True,
        ),
        ModelConfig(
            name="DecisionTree",
            objective_factory=create_dt_objective,
            base_params={"random_state": DEFAULT_SEED},
            uses_scaled_data=False,
        ),
    ]
    if _HAS_XGB:
        configs.insert(
            1,
            ModelConfig(
                name="XGBoost",
                objective_factory=create_xgb_objective,
                uses_scaled_data=False,
            ),
        )
    return configs


def train_and_evaluate(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    X_train_scaled: np.ndarray,
    X_val_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    feature_cols: list[str],
    *,
    experiment_name: str,
    n_trials: int = DEFAULT_N_TRIALS,
    timeout: int | None = None,
    seed: int = DEFAULT_SEED,
    model_configs: list[ModelConfig] | None = None,
) -> tuple[
    dict[str, TrainResult],
    Any,
    str,
    str | None,
]:
    """Train and evaluate multiple models with Optuna + MLflow.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
        Raw (unscaled) feature matrices.
    y_train, y_val, y_test : np.ndarray
        Target labels.
    X_train_scaled, X_val_scaled, X_test_scaled : np.ndarray
        Scaled feature matrices (for LogisticRegression).
    feature_cols : list[str]
        Feature column names.
    experiment_name : str
        MLflow experiment name or path.
    n_trials : int, default=DEFAULT_N_TRIALS
        Optuna trials per model.
    timeout : int | None, default=None
        Optuna timeout in seconds.
    seed : int, default=DEFAULT_SEED
        Random seed.
    model_configs : list[ModelConfig] | None, default=None
        Model configurations. If None, uses default_model_configs().

    Returns
    -------
    model_results : dict[str, TrainResult]
        Results for each model.
    best_model : Any
        The best trained model.
    best_model_name : str
        Name of the best model.
    best_run_id : str | None
        MLflow run ID of the best model.
    """
    if model_configs is None:
        model_configs = default_model_configs()

    mlflow.set_experiment(experiment_name)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    model_results: dict[str, TrainResult] = {}
    best_val_auc = -np.inf
    best_model = None
    best_model_name = ""
    best_run_id = None

    for config in model_configs:
        print(f"\n{'─' * 60}")
        print(f"Training {config.name}...")
        print(f"{'─' * 60}")

        # Select data (scaled or raw)
        X_tr = X_train_scaled if config.uses_scaled_data else X_train
        X_v = X_val_scaled if config.uses_scaled_data else X_val
        X_te = X_test_scaled if config.uses_scaled_data else X_test

        # Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        objective = config.objective_factory(
            X_tr, y_train, X_v, y_val, seed=seed
        )

        with suppress_training_warnings():
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=False,
            )

        best_params = study.best_params
        best_val_metric = study.best_value

        print(f"  Best val ROC-AUC: {best_val_metric:.4f}")
        print(f"  Best params: {best_params}")

        # Retrain on train+val
        X_trainval = np.vstack([X_tr, X_v])
        y_trainval = np.concatenate([y_train, y_val])
        model = create_model(config, best_params)

        with suppress_training_warnings():
            model.fit(X_trainval, y_trainval)

        # Evaluate on test
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        test_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

        print(
            f"  Test: acc={test_metrics['accuracy']:.3f} "
            f"f1={test_metrics['f1']:.3f} auc={test_metrics['roc_auc']:.3f}"
        )

        # Log to MLflow
        run_id = None
        try:
            with mlflow.start_run(run_name=config.name) as run:
                run_id = run.info.run_id
                mlflow.log_params(best_params)
                mlflow.log_param("model_type", config.name)
                mlflow.log_param("n_features", len(feature_cols))
                mlflow.log_metric("val_roc_auc", best_val_metric)
                for k, v in test_metrics.items():
                    mlflow.log_metric(f"test_{k}", v)
                mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            warnings.warn(f"MLflow logging failed for {config.name}: {e}")

        # Store result
        result = TrainResult(
            name=config.name,
            model=model,
            best_params=best_params,
            val_metric=best_val_metric,
            test_metrics=test_metrics,
            run_id=run_id,
        )
        model_results[config.name] = result

        # Track best
        if best_val_metric > best_val_auc:
            best_val_auc = best_val_metric
            best_model = model
            best_model_name = config.name
            best_run_id = run_id

    return model_results, best_model, best_model_name, best_run_id
