"""btcusdt_utils — reusable utilities for BTCUSDT ML pipelines.

Flat modules
------------
constants     : Catalog and path constants
viz           : Plotly dark theme configuration
quality       : Bar and label quality metrics (AFML-inspired)
stationarity  : ADF/KPSS tests and FFD fractional differentiation
bars          : Tick-to-bar aggregation (time, volume, dollar, tick)
labeling      : Triple-barrier labeling (AFML Ch. 3)

Subpackages
-----------
features/     : Technical, microstructure, entropy, structural features
scoring/      : MI scoring, noise floor, univariate AUC, sample weights
evaluation/   : Classification metrics, baselines, Plotly plots
training/     : Optuna hyperparameter tuning + MLflow logging
"""

__version__ = "0.4.0"

# ── Flat modules ─────────────────────────────────────────
from btcusdt_utils.constants import (
    CATALOG, VOLUME, BRONZE, SILVER, GOLD,
    RAW_AGGTRADES, RAW_AGGTRADES_PATH,
    RAW_NEWS, RAW_NEWS_PATH,
)
from btcusdt_utils.viz import (
    _BG, _BG_CARD, _TEXT, _TEXT_MUT, _GRID,
    _RED, _GREEN, _BLUE, _ORANGE, _PURPLE, _CYAN,
    _FONT, DARK_LAYOUT, dark_fig,
)
from btcusdt_utils.quality import (
    BarQuality, LabelQuality,
    compute_bar_quality, compute_label_quality,
    label_entropy, label_avg_uniqueness,
)
from btcusdt_utils.stationarity import (
    adf_test, kpss_test, ffd, find_min_ffd,
)
from btcusdt_utils.bars import (
    build_time_bars, build_threshold_bars, cusum_filter,
)
from btcusdt_utils.labeling import apply_triple_barrier

# ── Features subpackage ──────────────────────────────────
from btcusdt_utils.features import (
    TechnicalConfig, MicrostructureConfig, EntropyConfig, StructuralConfig,
    compute_technical_indicators,
    compute_microstructure_features,
    compute_entropy_features,
    compute_structural_features,
)

# ── Scoring subpackage ───────────────────────────────────
try:
    from btcusdt_utils.scoring import (
        ScoringConfig, NoiseFloor,
        compute_mi_scores, compute_noise_floor,
        compute_univariate_auc,
        WeightsConfig,
        compute_concurrency, compute_sample_weights,
        sequential_bootstrap,
    )
except ImportError:
    pass  # available after `%pip install lightgbm scikit-learn`

# ── Evaluation subpackage ────────────────────────────────
try:
    from btcusdt_utils.evaluation import (
        BaselineResult,
        compute_metrics, compute_baselines,
        build_comparison_table,
        plot_roc_curves, plot_precision_recall,
        plot_confusion_matrix, plot_feature_importance,
        plot_calibration,
    )
except ImportError:
    pass

# ── Training subpackage ──────────────────────────────────
try:
    from btcusdt_utils.training import (
        ModelConfig, TrainResult,
        create_lgb_objective, create_xgb_objective,
        create_rf_objective, create_lr_objective, create_dt_objective,
        default_model_configs, train_and_evaluate,
        DEFAULT_N_TRIALS, DEFAULT_SEED, DEFAULT_EARLY_STOPPING_ROUNDS,
    )
except ImportError:
    pass

__all__ = [
    "__version__",
    # constants
    "CATALOG", "VOLUME", "BRONZE", "SILVER", "GOLD",
    "RAW_AGGTRADES", "RAW_AGGTRADES_PATH",
    "RAW_NEWS", "RAW_NEWS_PATH",
    # viz
    "_BG", "_BG_CARD", "_TEXT", "_TEXT_MUT", "_GRID",
    "_RED", "_GREEN", "_BLUE", "_ORANGE", "_PURPLE", "_CYAN",
    "_FONT", "DARK_LAYOUT", "dark_fig",
    # quality
    "BarQuality", "LabelQuality",
    "compute_bar_quality", "compute_label_quality",
    "label_entropy", "label_avg_uniqueness",
    # stationarity
    "adf_test", "kpss_test", "ffd", "find_min_ffd",
    # bars
    "build_time_bars", "build_threshold_bars", "cusum_filter",
    # labeling
    "apply_triple_barrier",
    # features
    "TechnicalConfig", "MicrostructureConfig",
    "EntropyConfig", "StructuralConfig",
    "compute_technical_indicators",
    "compute_microstructure_features",
    "compute_entropy_features",
    "compute_structural_features",
    # scoring
    "ScoringConfig", "NoiseFloor",
    "compute_mi_scores", "compute_noise_floor",
    "compute_univariate_auc",
    "WeightsConfig",
    "compute_concurrency", "compute_sample_weights",
    "sequential_bootstrap",
    # evaluation
    "BaselineResult",
    "compute_metrics", "compute_baselines",
    "build_comparison_table",
    "plot_roc_curves", "plot_precision_recall",
    "plot_confusion_matrix", "plot_feature_importance",
    "plot_calibration",
    # training
    "ModelConfig", "TrainResult",
    "create_lgb_objective", "create_xgb_objective",
    "create_rf_objective", "create_lr_objective", "create_dt_objective",
    "default_model_configs", "train_and_evaluate",
    "DEFAULT_N_TRIALS", "DEFAULT_SEED", "DEFAULT_EARLY_STOPPING_ROUNDS",
]
