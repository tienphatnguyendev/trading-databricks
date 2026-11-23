"""Optuna-powered model training subpackage.

Submodules
----------
_config     : ModelConfig, TrainResult, defaults, helpers
objectives  : Optuna objective factories (LGB, XGB, RF, LR, DT)
pipeline    : train_and_evaluate, default_model_configs
cv          : Purged K-fold and walk-forward cross-validation (AFML Ch. 7.4)
"""

# CV module has no optuna dependency — always available
from btcusdt_utils.training.cv import (
    PurgedCVConfig, PurgedKFoldCV, walk_forward_cv,
)

# Training modules require optuna — conditional import
try:
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
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

__all__ = [
    "PurgedCVConfig", "PurgedKFoldCV", "walk_forward_cv",
]

if _HAS_OPTUNA:
    __all__.extend([
        "ModelConfig", "TrainResult",
        "DEFAULT_N_TRIALS", "DEFAULT_SEED", "DEFAULT_EARLY_STOPPING_ROUNDS",
        "create_lgb_objective", "create_xgb_objective",
        "create_rf_objective", "create_lr_objective", "create_dt_objective",
        "default_model_configs", "train_and_evaluate",
    ])
