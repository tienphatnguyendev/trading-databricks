"""btcusdt_utils — reusable utilities for BTCUSDT ML pipelines.

Flat modules
------------
constants        : Catalog and path constants
config           : Centralized configuration (extends constants)
viz              : Plotly dark theme configuration + Phase 2 dashboard builders
quality          : Bar and label quality metrics (AFML-inspired)
stationarity     : ADF/KPSS tests and FFD fractional differentiation
bars             : Tick-to-bar aggregation (time, volume, dollar, tick)
labeling         : Triple-barrier labeling (AFML Ch. 3)
labeling_sweep   : Triple-barrier parameter sweep with quality gating
labeling_analysis: Sweep DataFrame building, heatmap prep, gap analysis
tracking         : MLflow experiment tracking abstraction
reporting        : Console reporting (ASCII tables, section headers)

Subpackages
-----------
features/     : Technical, microstructure, entropy, structural features
scoring/      : MI scoring, noise floor, univariate AUC, sample weights
evaluation/   : Classification metrics, baselines, Plotly plots
training/     : Optuna hyperparameter tuning + MLflow logging + Purged/Walk-Forward CV
sampling/     : Uniqueness weights, return attribution, sequential bootstrap
"""

from importlib import import_module


__version__ = "0.11.0"

# Resolve root-level convenience exports only when callers request them.
# Importing bars must not require optional training or statistics libraries.
_MODULE_EXPORTS = {
    'constants': (
        'CATALOG', 'VOLUME', 'BRONZE',
        'SILVER', 'GOLD', 'RAW_AGGTRADES',
        'RAW_AGGTRADES_PATH', 'RAW_NEWS', 'RAW_NEWS_PATH',
    ),
    'config': (
        'CatalogConfig', 'TrainingConfig', 'VisualizationConfig',
        'MLflowConfig', 'BarSamplingConfig', 'TripleBarrierConfig',
        'NotebookConfig',
    ),
    'viz': (
        '_BG', '_BG_CARD', '_TEXT',
        '_TEXT_MUT', '_GRID', '_RED',
        '_GREEN', '_BLUE', '_ORANGE',
        '_PURPLE', '_CYAN', '_FONT',
        'DARK_LAYOUT', 'dark_fig', 'plot_ptsl_heatmap',
        'plot_touch_bar_histogram', 'plot_sweep_dashboard',
    ),
    'quality': (
        'BarQuality', 'LabelQuality', 'compute_bar_quality',
        'compute_label_quality', 'label_entropy', 'label_avg_uniqueness',
    ),
    'stationarity': (
        'adf_test', 'kpss_test', 'ffd',
        'find_min_ffd',
    ),
    'bars': (
        'build_time_bars', 'build_threshold_bars', 'cusum_filter',
    ),
    'labeling': (
        'apply_triple_barrier',
    ),
    'labeling_sweep': (
        'SweepConfig', 'SweepResult', 'SweepStats',
        'SweepReporter', 'run_triple_barrier_sweep', 'prune_label_arrays',
    ),
    'labeling_analysis': (
        'GapAnalysisTarget', 'GapAnalysisResult', 'build_sweep_dataframe',
        'build_heatmap_data', 'analyze_gaps', 'filter_top_n',
        'filter_by_vol_span', 'get_top_n_configs',
    ),
    'labeling_meta': (
        'MetaLabelConfig', 'generate_meta_labels', 'create_meta_features',
    ),
    'tracking': (
        'ExperimentConfig', 'MLflowTracker',
    ),
    'reporting': (
        'print_section_header', 'print_subsection', 'print_top_configs_table',
        'print_gap_analysis_table', 'print_label_distribution', 'print_feature_scores',
        'print_cv_results',
    ),
    'features': (
        'TechnicalConfig', 'MicrostructureConfig', 'EntropyConfig',
        'StructuralConfig', 'compute_technical_indicators', 'compute_microstructure_features',
        'compute_entropy_features', 'compute_structural_features',
    ),
    'sampling': (
        'compute_avg_uniqueness', 'compute_return_attribution_weights', 'get_ind_matrix',
    ),
    'trading': (
        'kelly_bet_size', 'linear_bet_size', 'bet_size_from_meta_prob',
    ),
    'scoring': (
        'ScoringConfig', 'NoiseFloor', 'compute_mi_scores',
        'compute_noise_floor', 'compute_univariate_auc', 'WeightsConfig',
        'compute_concurrency', 'compute_sample_weights', 'sequential_bootstrap',
    ),
    'evaluation': (
        'BaselineResult', 'compute_metrics', 'compute_baselines',
        'build_comparison_table', 'plot_roc_curves', 'plot_precision_recall',
        'plot_confusion_matrix', 'plot_feature_importance', 'plot_calibration',
        'plot_cv_comparison', 'plot_meta_labeling_dashboard', 'plot_regime_analysis',
    ),
    'training': (
        'ModelConfig', 'TrainResult', 'create_lgb_objective',
        'create_xgb_objective', 'create_rf_objective', 'create_lr_objective',
        'create_dt_objective', 'default_model_configs', 'train_and_evaluate',
        'DEFAULT_N_TRIALS', 'DEFAULT_SEED', 'DEFAULT_EARLY_STOPPING_ROUNDS',
        'PurgedCVConfig', 'PurgedKFoldCV', 'walk_forward_cv',
    ),
}
_EXPORTS = {
    name: (module, name)
    for module, names in _MODULE_EXPORTS.items()
    for name in names
} | {
    'compute_sample_weights_v2': ('sampling', 'compute_sample_weights'),
    'sequential_bootstrap_v2': ('sampling', 'sequential_bootstrap'),
}


def __getattr__(name: str):
    """Load a convenience export while preserving dependency import errors."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "__version__",
    # constants
    "CATALOG", "VOLUME", "BRONZE", "SILVER", "GOLD",
    "RAW_AGGTRADES", "RAW_AGGTRADES_PATH",
    "RAW_NEWS", "RAW_NEWS_PATH",
    # config
    "CatalogConfig",
    "TrainingConfig",
    "VisualizationConfig",
    "MLflowConfig",
    "BarSamplingConfig",
    "TripleBarrierConfig",
    "NotebookConfig",
    # viz
    "_BG", "_BG_CARD", "_TEXT", "_TEXT_MUT", "_GRID",
    "_RED", "_GREEN", "_BLUE", "_ORANGE", "_PURPLE", "_CYAN",
    "_FONT", "DARK_LAYOUT", "dark_fig",
    "plot_ptsl_heatmap",  # NEW
    "plot_touch_bar_histogram",  # NEW
    "plot_sweep_dashboard",  # NEW
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
    # labeling_sweep
    "SweepConfig", "SweepResult", "SweepStats", "SweepReporter",
    "run_triple_barrier_sweep", "prune_label_arrays",
    # labeling_analysis
    "GapAnalysisTarget", "GapAnalysisResult",
    "build_sweep_dataframe", "build_heatmap_data", "analyze_gaps",
    "filter_top_n", "filter_by_vol_span", "get_top_n_configs",
    # tracking (NEW)
    "ExperimentConfig",
    "MLflowTracker",
    # reporting (NEW)
    "print_section_header",
    "print_subsection",
    "print_top_configs_table",
    "print_gap_analysis_table",
    "print_label_distribution",
    "print_feature_scores",
    "print_cv_results",
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
    # sampling (NEW)
    "compute_avg_uniqueness",
    "compute_sample_weights_v2",
    "compute_return_attribution_weights",
    "sequential_bootstrap_v2",
    "get_ind_matrix",
    # evaluation
    "BaselineResult",
    "compute_metrics", "compute_baselines",
    "build_comparison_table",
    "plot_roc_curves", "plot_precision_recall",
    "plot_confusion_matrix", "plot_feature_importance",
    "plot_calibration",
    "plot_cv_comparison",
    "plot_meta_labeling_dashboard",
    "plot_regime_analysis",
    # training
    "ModelConfig", "TrainResult",
    "create_lgb_objective", "create_xgb_objective",
    "create_rf_objective", "create_lr_objective", "create_dt_objective",
    "default_model_configs", "train_and_evaluate",
    "DEFAULT_N_TRIALS", "DEFAULT_SEED", "DEFAULT_EARLY_STOPPING_ROUNDS",
    "PurgedCVConfig", "PurgedKFoldCV", "walk_forward_cv",
    # Meta-labeling
    "MetaLabelConfig",
    "generate_meta_labels",
    "create_meta_features",
    # Bet sizing
    "kelly_bet_size",
    "linear_bet_size",
    "bet_size_from_meta_prob",
]
