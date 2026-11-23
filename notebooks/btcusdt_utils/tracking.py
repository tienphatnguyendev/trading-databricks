"""MLflow experiment tracking abstraction layer.

Centralizes MLflow logging to eliminate boilerplate and coupling.
Supports bar quality, sweep results, final model selection logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow

from btcusdt_utils.quality import BarQuality, LabelQuality


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExperimentConfig:
    """MLflow experiment configuration.
    
    Attributes:
        base_path: Base experiment path (e.g., "/Users/{user}/BTCUSDT")
        phase: Phase identifier (e.g., "Phase1_BarSampling", "Phase2")
    """
    base_path: str
    phase: str
    
    @property
    def name(self) -> str:
        """Full experiment name."""
        return f"{self.base_path}_{self.phase}"


# ═══════════════════════════════════════════════════════════════════════════
# Tracker
# ═══════════════════════════════════════════════════════════════════════════

class MLflowTracker:
    """Centralized MLflow experiment tracking.
    
    Eliminates boilerplate by providing high-level logging methods
    for common AFML pipeline events.
    
    Example:
        >>> config = ExperimentConfig(
        ...     base_path="/Users/user@example.com/BTCUSDT",
        ...     phase="Phase1_BarSampling",
        ... )
        >>> tracker = MLflowTracker(config)
        >>> tracker.log_bar_quality(bar_quality_obj)
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize tracker and set MLflow experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        mlflow.set_experiment(config.name)
    
    def log_bar_quality(self, bar_quality: BarQuality, run_name: str | None = None) -> None:
        """Log bar quality metrics to MLflow.
        
        Args:
            bar_quality: BarQuality instance from Phase 1
            run_name: Optional run name (default: bar_quality.name)
        """
        run_name = run_name or bar_quality.name
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "bar_type": bar_quality.name,
                "n_bars": bar_quality.n_bars,
            })
            mlflow.log_metrics({
                "jb_stat": bar_quality.jb_stat,
                "jb_pval": bar_quality.jb_pval,
                "lb_stat": bar_quality.lb_stat,
                "lb_pval": bar_quality.lb_pval,
                "acf_1": bar_quality.acf_1,
                "acf_1_abs": abs(bar_quality.acf_1),
                "ret_mean": bar_quality.ret_mean,
                "ret_std": bar_quality.ret_std,
                "cv": bar_quality.cv,
            })
    
    def log_sweep_result(
        self,
        config_name: str,
        params: dict[str, Any],
        quality: LabelQuality,
    ) -> None:
        """Log a single triple-barrier sweep configuration.
        
        Args:
            config_name: Configuration identifier (e.g., "PT1.0_SL1.0_h30_s50")
            params: Parameter dict (pt_mult, sl_mult, max_hold, vol_span)
            quality: LabelQuality instance
        """
        with mlflow.start_run(run_name=config_name):
            mlflow.log_params(params)
            mlflow.log_metrics({
                "n_samples": quality.n_samples,
                "entropy": quality.entropy,
                "avg_uniqueness": quality.avg_uniqueness,
                "acf_1": quality.acf_1,
                "acf_5": quality.acf_5,
                "median_touch_bars": quality.median_touch_bars,
                "vertical_pct": quality.vertical_pct,
                "class_balance": quality.class_balance,
                "score": quality.score,
            })
    
    def log_final_selection(
        self,
        best_config_name: str,
        best_quality: LabelQuality,
        n_labels: int,
        touch_type_counts: dict[int, int],
        gap_results: list[dict] | None = None,
    ) -> None:
        """Log Phase 2C final triple-barrier selection.
        
        Args:
            best_config_name: Name of selected configuration
            best_quality: LabelQuality for best config
            n_labels: Total number of labels generated
            touch_type_counts: Dict mapping touch_type (0=TP, 1=SL, 2=vert) to count
            gap_results: Optional list of gap analysis dicts
        """
        with mlflow.start_run(run_name="final_selection"):
            mlflow.log_params({
                "best_config": best_config_name,
                "n_labels": n_labels,
            })
            mlflow.log_metrics({
                "entropy": best_quality.entropy,
                "avg_uniqueness": best_quality.avg_uniqueness,
                "acf_1": best_quality.acf_1,
                "acf_5": best_quality.acf_5,
                "median_touch_bars": best_quality.median_touch_bars,
                "vertical_pct": best_quality.vertical_pct,
                "class_balance": best_quality.class_balance,
                "score": best_quality.score,
                "tp_count": touch_type_counts.get(0, 0),
                "sl_count": touch_type_counts.get(1, 0),
                "vert_count": touch_type_counts.get(2, 0),
            })
            
            if gap_results:
                # Log gap analysis as tags
                for gap in gap_results:
                    tag_key = f"gap_{gap['metric']}"
                    tag_value = "✓" if gap.get("passed", False) else "✗"
                    mlflow.set_tag(tag_key, tag_value)
    
    def log_feature_scoring(
        self,
        feature_scores: dict[str, float],
        noise_floor: float,
        n_above_floor: int,
        top_n: int = 20,
    ) -> None:
        """Log Phase 4 feature scoring results.
        
        Args:
            feature_scores: Dict mapping feature name -> MI/AUC score
            noise_floor: Noise floor threshold
            n_above_floor: Number of features above noise floor
            top_n: Number of top features to log individually
        """
        with mlflow.start_run(run_name="feature_scoring"):
            mlflow.log_params({
                "n_features": len(feature_scores),
                "n_above_floor": n_above_floor,
                "noise_floor": noise_floor,
            })
            
            # Log top-N feature scores individually
            sorted_features = sorted(
                feature_scores.items(), key=lambda x: x[1], reverse=True
            )
            for i, (feat_name, score) in enumerate(sorted_features[:top_n], 1):
                mlflow.log_metric(f"feat_{i:02d}_{feat_name}", score)
    
    def log_model_performance(
        self,
        run_name: str,
        metrics: dict[str, float],
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Generic method for logging model training/evaluation.
        
        Args:
            run_name: MLflow run name
            metrics: Performance metrics (AUC, accuracy, etc.)
            params: Optional hyperparameters
            tags: Optional metadata tags
        """
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
    
    def log_cv_results(
        self,
        run_name: str,
        cv_type: str,
        fold_metrics: list[dict[str, float]],
        mean_metrics: dict[str, float],
        config: dict[str, Any] | None = None,
    ) -> None:
        """Log cross-validation results (Phase 6).
        
        Args:
            run_name: MLflow run name
            cv_type: CV strategy ("purged", "walk_forward", "standard")
            fold_metrics: List of per-fold metric dicts
            mean_metrics: Aggregated metrics across folds
            config: Optional CV configuration
        """
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("cv_type", cv_type)
            mlflow.log_param("n_folds", len(fold_metrics))
            
            if config:
                for key, value in config.items():
                    mlflow.log_param(f"cv_{key}", value)
            
            # Log mean metrics
            for metric_name, value in mean_metrics.items():
                mlflow.log_metric(f"mean_{metric_name}", value)
            
            # Log per-fold metrics
            for fold_idx, fold_dict in enumerate(fold_metrics):
                for metric_name, value in fold_dict.items():
                    mlflow.log_metric(f"fold{fold_idx}_{metric_name}", value)
