"""Optuna-powered model training subpackage.

Submodules
----------
_config     : ModelConfig, TrainResult, defaults, helpers
objectives  : Optuna objective factories (LGB, XGB, RF, LR, DT)
pipeline    : train_and_evaluate, default_model_configs
"""

from btcusdt_utils.training._config import (
    ModelConfig, TrainResult,
    DEFAULT_N_TRIALS, DEFAULT_SEED, DEFAULT_EARLY_STOPPING_ROUNDS,
)
from btcusdt_utils.training.objectives import (
    create_lgb_objective, create_xgb_objective,
    create_rf_objective, create_lr_objective, create_dt_objective,
)
from btcusdt_utils.training.pipeline import (
    default_model_configs, train_and_evaluate,
)

__all__ = [
    "ModelConfig", "TrainResult",
    "DEFAULT_N_TRIALS", "DEFAULT_SEED", "DEFAULT_EARLY_STOPPING_ROUNDS",
    "create_lgb_objective", "create_xgb_objective",
    "create_rf_objective", "create_lr_objective", "create_dt_objective",
    "default_model_configs", "train_and_evaluate",
]
