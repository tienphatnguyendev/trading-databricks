"""Model evaluation subpackage.

Submodules
----------
metrics     : compute_metrics, BaselineResult, compute_baselines
comparison  : build_comparison_table
plots       : ROC, PR, confusion matrix, feature importance, calibration
"""

from btcusdt_utils.evaluation.metrics import (
    compute_metrics,
    BaselineResult,
    compute_baselines,
)
from btcusdt_utils.evaluation.comparison import build_comparison_table
from btcusdt_utils.evaluation.plots import (
    plot_roc_curves,
    plot_precision_recall,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_calibration,
    plot_cv_comparison,
    plot_meta_labeling_dashboard,
    plot_regime_analysis,
)

__all__ = [
    "compute_metrics", "BaselineResult", "compute_baselines",
    "build_comparison_table",
    "plot_roc_curves", "plot_precision_recall",
    "plot_confusion_matrix", "plot_feature_importance",
    "plot_calibration",
    "plot_cv_comparison",
    "plot_meta_labeling_dashboard",
    "plot_regime_analysis",
]
