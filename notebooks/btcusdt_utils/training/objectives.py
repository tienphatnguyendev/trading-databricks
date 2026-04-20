"""Optuna objective factories for hyperparameter tuning.

Each factory returns a callable ``(trial) -> float`` suitable for
``study.optimize()``.  XGBoost objective is only available when
the ``xgboost`` package is installed.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from btcusdt_utils.training._config import DEFAULT_SEED, DEFAULT_EARLY_STOPPING_ROUNDS

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


def create_lgb_objective(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_v: np.ndarray, y_v: np.ndarray,
    *, seed: int = DEFAULT_SEED,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
) -> Callable[[optuna.Trial], float]:
    """Optuna objective for LightGBM binary classification."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary", "metric": "auc",
            "verbosity": -1, "boosting_type": "gbdt", "seed": seed,
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr, eval_set=[(X_v, y_v)],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        return roc_auc_score(y_v, model.predict_proba(X_v)[:, 1])
    return objective


def create_xgb_objective(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_v: np.ndarray, y_v: np.ndarray,
    *, seed: int = DEFAULT_SEED,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
) -> Callable[[optuna.Trial], float]:
    """Optuna objective for XGBoost binary classification."""
    if not _HAS_XGB:
        raise ImportError("xgboost is required for create_xgb_objective")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "seed": seed, "verbosity": 0,
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "early_stopping_rounds": early_stopping_rounds,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
        return roc_auc_score(y_v, model.predict_proba(X_v)[:, 1])
    return objective


def create_rf_objective(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_v: np.ndarray, y_v: np.ndarray,
    *, seed: int = DEFAULT_SEED,
) -> Callable[[optuna.Trial], float]:
    """Optuna objective for RandomForest."""
    def objective(trial: optuna.Trial) -> float:
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 5, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            random_state=seed, n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        return roc_auc_score(y_v, model.predict_proba(X_v)[:, 1])
    return objective


def create_lr_objective(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_v: np.ndarray, y_v: np.ndarray,
    *, seed: int = DEFAULT_SEED,
) -> Callable[[optuna.Trial], float]:
    """Optuna objective for LogisticRegression."""
    def objective(trial: optuna.Trial) -> float:
        model = LogisticRegression(
            C=trial.suggest_float("C", 1e-4, 100, log=True),
            penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed,
        )
        model.fit(X_tr, y_tr)
        return roc_auc_score(y_v, model.predict_proba(X_v)[:, 1])
    return objective


def create_dt_objective(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_v: np.ndarray, y_v: np.ndarray,
    *, seed: int = DEFAULT_SEED,
) -> Callable[[optuna.Trial], float]:
    """Optuna objective for DecisionTree."""
    def objective(trial: optuna.Trial) -> float:
        model = DecisionTreeClassifier(
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 50),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            random_state=seed,
        )
        model.fit(X_tr, y_tr)
        return roc_auc_score(y_v, model.predict_proba(X_v)[:, 1])
    return objective
