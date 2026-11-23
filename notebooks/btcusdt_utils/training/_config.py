"""Training configuration, dataclasses, and internal helpers."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


DEFAULT_N_TRIALS: int = 20
DEFAULT_SEED: int = 42
DEFAULT_EARLY_STOPPING_ROUNDS: int = 50


@contextmanager
def suppress_training_warnings() -> Iterator[None]:
    """Scoped suppression of verbose LightGBM/XGBoost warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
        warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
        warnings.filterwarnings("ignore", category=FutureWarning, module="lightgbm")
        yield


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single model type in the training pipeline."""

    name: str
    objective_factory: Callable
    base_params: dict[str, Any] = field(default_factory=dict)
    uses_scaled_data: bool = False

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"scaled={self.uses_scaled_data})"
        )


@dataclass(frozen=True)
class TrainResult:
    """Result from training a single model configuration."""

    name: str
    model: Any
    best_params: dict[str, Any]
    val_metric: float
    test_metrics: dict[str, float]
    run_id: str | None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"val_auc={self.val_metric:.4f}, "
            f"test_auc={self.test_metrics.get('roc_auc', 0):.4f})"
        )


# ── Model class registry ──────────────────────────────────

MODEL_CLASSES: dict[str, type] = {
    "LightGBM": lgb.LGBMClassifier,
    "RandomForest": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "DecisionTree": DecisionTreeClassifier,
}
if _HAS_XGB:
    MODEL_CLASSES["XGBoost"] = xgb.XGBClassifier


def create_model(config: ModelConfig, best_params: dict[str, Any]) -> Any:
    """Instantiate a model with base + Optuna-tuned params."""
    cls = MODEL_CLASSES[config.name]
    return cls(**config.base_params, **best_params)
