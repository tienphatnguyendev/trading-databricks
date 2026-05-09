"""Sample weighting and sequential bootstrap for ML training.

Provides:
* Uniqueness weights for overlapping labels
* Return attribution weights
* Sequential bootstrap with embargo

References:
* Advances in Financial Machine Learning (Lopez de Prado, 2018)
  - Chapter 4: Sample Weights
  - Chapter 5: Fractionally Differentiated Features
  - Chapter 7: Cross-Validation
"""

from btcusdt_utils.sampling.weights import (
    compute_avg_uniqueness,
    compute_sample_weights,
    compute_return_attribution_weights,
)
from btcusdt_utils.sampling.bootstrap import (
    sequential_bootstrap,
    get_ind_matrix,
)

__all__ = [
    "compute_avg_uniqueness",
    "compute_sample_weights",
    "compute_return_attribution_weights",
    "sequential_bootstrap",
    "get_ind_matrix",
]
