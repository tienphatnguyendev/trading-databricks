"""Feature scoring and sample weighting subpackage.

Submodules
----------
_config     : ScoringConfig frozen dataclass
mi          : Mutual information scoring + permutation noise floor
univariate  : Single-feature LightGBM AUC
_weights    : Sample uniqueness weights + sequential bootstrap (AFML Ch. 4)
"""

from btcusdt_utils.scoring._config import ScoringConfig
from btcusdt_utils.scoring.mi import (
    NoiseFloor,
    compute_mi_scores,
    compute_noise_floor,
)
from btcusdt_utils.scoring.univariate import compute_univariate_auc
from btcusdt_utils.scoring._weights import (
    WeightsConfig,
    compute_concurrency,
    compute_sample_weights,
    sequential_bootstrap,
)

__all__ = [
    "ScoringConfig",
    "NoiseFloor",
    "compute_mi_scores",
    "compute_noise_floor",
    "compute_univariate_auc",
    "WeightsConfig",
    "compute_concurrency",
    "compute_sample_weights",
    "sequential_bootstrap",
]
