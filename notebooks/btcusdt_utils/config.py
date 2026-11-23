"""Centralized configuration for BTCUSDT AutoML notebooks.

Consolidates all constants, hyperparameters, and paths into
immutable dataclasses. Eliminates magic numbers and hardcoded
values scattered across notebook cells.

Usage
-----
```python
from btcusdt_utils.config import NotebookConfig

cfg = NotebookConfig()
print(cfg.aggtrades_path)  # `smart_trading_dev`.`01_bronze`.`aggtrades`
print(cfg.training.train_ratio)  # 0.60
print(cfg.training.optuna_trials)  # 20
```

For custom configurations:
```python
cfg = NotebookConfig(
    train_ratio=0.70,
    optuna_trials=50,
    experiment_suffix="_v2",
)
```
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

# Import Unity Catalog structure from constants.py
from btcusdt_utils.constants import (
    CATALOG,
    BRONZE,
    SILVER,
    GOLD,
    VOLUME,
)


# ═══════════════════════════════════════════════════════════════
# Unity Catalog & Data Paths
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CatalogConfig:
    """Unity Catalog structure and table references."""
    
    catalog: str = CATALOG
    volume_schema: str = VOLUME
    bronze: str = BRONZE
    silver: str = SILVER
    gold: str = GOLD
    
    # Table names (without schema prefix)
    aggtrades_table: str = "aggtrades"
    news_table: str = "raw_news"
    
    @property
    def aggtrades_path(self) -> str:
        """Fully qualified path: `catalog`.`schema`.`table`."""
        return f"`{self.catalog}`.`{self.bronze}`.`{self.aggtrades_table}`"
    
    @property
    def news_path(self) -> str:
        """Fully qualified path: `catalog`.`schema`.`table`."""
        return f"`{self.catalog}`.`{self.bronze}`.`{self.news_table}`"
    
    @property
    def silver_features_table(self) -> str:
        """Silver layer features table name."""
        return "btcusdt_features_1min"
    
    @property
    def silver_features_path(self) -> str:
        """Fully qualified path to silver features table."""
        return f"`{self.catalog}`.`{self.silver}`.`{self.silver_features_table}`"


# ═══════════════════════════════════════════════════════════════
# Training & Hyperparameter Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TrainingConfig:
    """Model training and hyperparameter tuning configuration."""
    
    # Train/Val/Test split ratios (must sum to 1.0)
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20
    
    # Optuna hyperparameter search
    optuna_trials: int = 20
    optuna_timeout_sec: int = 600
    
    # Random state for reproducibility
    random_state: int = 42
    
    # Early stopping patience (validation set)
    early_stopping_rounds: int = 50
    
    def __post_init__(self) -> None:
        """Validate that split ratios sum to 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"Split ratios must sum to 1.0 (got {total:.4f})"
            )


# ═══════════════════════════════════════════════════════════════
# Visualization & Reporting Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VisualizationConfig:
    """Visualization and reporting settings."""
    
    # Feature importance display
    top_n_features: int = 20
    
    # Rolling metrics window (for temporal analysis)
    rolling_window_size: int = 100
    
    # Print formatting
    section_divider: str = "=" * 70
    subsection_divider: str = "─" * 70
    
    # Chart dimensions (pixels)
    default_chart_width: int = 900
    default_chart_height: int = 500


# ═══════════════════════════════════════════════════════════════
# MLflow Experiment Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MLflowConfig:
    """MLflow experiment tracking configuration."""
    
    # Username (auto-detected from environment)
    username: str | None = None
    
    # Experiment name suffix (default: base experiment)
    experiment_suffix: str = "_manual_ml"
    
    # Model registry settings
    default_registry_alias: Literal["candidate", "champion"] = "candidate"
    
    # Minimum AUC thresholds for model promotion
    champion_auc_threshold: float = 0.55  # Above this = champion
    candidate_auc_threshold: float = 0.52  # Above this = candidate
    
    @property
    def effective_username(self) -> str:
        """Resolve username from explicit value or environment.
        
        Priority:
        1. Explicit username passed to __init__
        2. BTCUSDT_USERNAME env var
        3. Spark config (spark.databricks.userName)
        4. Fallback: nthuan2109@gmail.com
        """
        if self.username:
            return self.username
        
        # Try environment variable
        env_user = os.getenv("BTCUSDT_USERNAME")
        if env_user:
            return env_user
        
        # Try Spark config (if available)
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if spark:
                return spark.conf.get("spark.databricks.userName")
        except Exception:
            pass
        
        # Fallback
        return "nthuan2109@gmail.com"
    
    @property
    def experiment_name(self) -> str:
        """Full MLflow experiment path."""
        return f"/Users/{self.effective_username}/btcusdt_direction{self.experiment_suffix}"


# ═══════════════════════════════════════════════════════════════
# Phase-Specific Configuration (from refactoring assessment)
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BarSamplingConfig:
    """Phase 1: Bar sampling strategy configuration."""
    
    # Time bar intervals (minutes)
    time_intervals: tuple[int, ...] = (1, 5, 15, 30)
    
    # Volume/dollar/tick bar threshold multipliers
    threshold_multipliers: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)
    
    # Quality gate: minimum bars for evaluation
    min_bars_for_quality: int = 5000
    
    # CUSUM filter thresholds (in std dev units)
    cusum_thresholds: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0)


@dataclass(frozen=True)
class TripleBarrierConfig:
    """Phase 2: Triple barrier labeling configuration."""
    
    # PT/SL multiplier sweep ranges
    pt_mults: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
    sl_mults: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
    
    # Max holding period sweep (bars)
    max_holds: tuple[int, ...] = (10, 30, 60, 120, 240, 480)
    
    # Volatility EWMA span sweep
    vol_spans: tuple[int, ...] = (20, 50, 100, 200)
    
    # Quality gates for label configuration
    min_samples: int = 100
    min_balance: float = 0.20  # At least 20% minority class
    min_uniqueness: float = 0.10  # AFML Ch. 4 uniqueness metric
    
    # Memory management: keep top-N label arrays
    top_n_keep: int = 20


# ═══════════════════════════════════════════════════════════════
# Master Configuration (combines all sub-configs)
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NotebookConfig:
    """Master configuration for BTCUSDT AutoML notebooks.
    
    Combines all sub-configurations into a single immutable object.
    All fields have sensible defaults and can be overridden.
    
    Examples
    --------
    Default configuration:
    >>> cfg = NotebookConfig()
    >>> cfg.aggtrades_path
    '`smart_trading_dev`.`01_bronze`.`aggtrades`'
    
    Custom configuration:
    >>> cfg = NotebookConfig(
    ...     train_ratio=0.70,
    ...     optuna_trials=50,
    ... )
    
    Access sub-configs:
    >>> cfg.catalog.bronze
    '01_bronze'
    >>> cfg.training.optuna_trials
    20
    """
    
    # Sub-configurations (can be overridden with custom instances)
    catalog: CatalogConfig = None
    training: TrainingConfig = None
    viz: VisualizationConfig = None
    mlflow: MLflowConfig = None
    bar_sampling: BarSamplingConfig = None
    triple_barrier: TripleBarrierConfig = None
    
    # Convenience overrides (applied to sub-configs if provided)
    train_ratio: float | None = None
    val_ratio: float | None = None
    test_ratio: float | None = None
    optuna_trials: int | None = None
    optuna_timeout_sec: int | None = None
    top_n_features: int | None = None
    rolling_window_size: int | None = None
    experiment_suffix: str | None = None
    username: str | None = None
    
    def __post_init__(self) -> None:
        """Initialize sub-configs with overrides applied."""
        # Use object.__setattr__ since dataclass is frozen
        
        # Catalog config (no overrides)
        if self.catalog is None:
            object.__setattr__(self, "catalog", CatalogConfig())
        
        # Training config (with overrides)
        if self.training is None:
            training_kwargs = {}
            if self.train_ratio is not None:
                training_kwargs["train_ratio"] = self.train_ratio
            if self.val_ratio is not None:
                training_kwargs["val_ratio"] = self.val_ratio
            if self.test_ratio is not None:
                training_kwargs["test_ratio"] = self.test_ratio
            if self.optuna_trials is not None:
                training_kwargs["optuna_trials"] = self.optuna_trials
            if self.optuna_timeout_sec is not None:
                training_kwargs["optuna_timeout_sec"] = self.optuna_timeout_sec
            object.__setattr__(self, "training", TrainingConfig(**training_kwargs))
        
        # Viz config (with overrides)
        if self.viz is None:
            viz_kwargs = {}
            if self.top_n_features is not None:
                viz_kwargs["top_n_features"] = self.top_n_features
            if self.rolling_window_size is not None:
                viz_kwargs["rolling_window_size"] = self.rolling_window_size
            object.__setattr__(self, "viz", VisualizationConfig(**viz_kwargs))
        
        # MLflow config (with overrides)
        if self.mlflow is None:
            mlflow_kwargs = {}
            if self.username is not None:
                mlflow_kwargs["username"] = self.username
            if self.experiment_suffix is not None:
                mlflow_kwargs["experiment_suffix"] = self.experiment_suffix
            object.__setattr__(self, "mlflow", MLflowConfig(**mlflow_kwargs))
        
        # Bar sampling config (no overrides)
        if self.bar_sampling is None:
            object.__setattr__(self, "bar_sampling", BarSamplingConfig())
        
        # Triple barrier config (no overrides)
        if self.triple_barrier is None:
            object.__setattr__(self, "triple_barrier", TripleBarrierConfig())
    
    # ═══════════════════════════════════════════════════════════════
    # Convenience properties (flatten common access patterns)
    # ═══════════════════════════════════════════════════════════════
    
    @property
    def aggtrades_path(self) -> str:
        """Shortcut: catalog.aggtrades_path."""
        return self.catalog.aggtrades_path
    
    @property
    def news_path(self) -> str:
        """Shortcut: catalog.news_path."""
        return self.catalog.news_path
    
    @property
    def silver_features_path(self) -> str:
        """Shortcut: catalog.silver_features_path."""
        return self.catalog.silver_features_path
    
    @property
    def experiment_name(self) -> str:
        """Shortcut: mlflow.experiment_name."""
        return self.mlflow.experiment_name
    
    @property
    def section_divider(self) -> str:
        """Shortcut: viz.section_divider."""
        return self.viz.section_divider
    
    @property
    def subsection_divider(self) -> str:
        """Shortcut: viz.subsection_divider."""
        return self.viz.subsection_divider
    
    def print_summary(self) -> None:
        """Print a formatted summary of the current configuration."""
        print("\n" + self.section_divider)
        print("CONFIGURATION SUMMARY")
        print(self.section_divider)
        
        print("\nUnity Catalog:")
        print(f"  Catalog:          {self.catalog.catalog}")
        print(f"  Bronze:           {self.catalog.bronze}")
        print(f"  Silver:           {self.catalog.silver}")
        print(f"  Gold:             {self.catalog.gold}")
        
        print("\nData Paths:")
        print(f"  Aggtrades:        {self.aggtrades_path}")
        print(f"  News:             {self.news_path}")
        print(f"  Silver features:  {self.silver_features_path}")
        
        print("\nTraining Config:")
        print(f"  Train/Val/Test:   "
              f"{self.training.train_ratio:.0%}/"
              f"{self.training.val_ratio:.0%}/"
              f"{self.training.test_ratio:.0%}")
        print(f"  Optuna trials:    {self.training.optuna_trials}")
        print(f"  Optuna timeout:   {self.training.optuna_timeout_sec}s")
        print(f"  Random state:     {self.training.random_state}")
        
        print("\nVisualization:")
        print(f"  Top N features:   {self.viz.top_n_features}")
        print(f"  Rolling window:   {self.viz.rolling_window_size}")
        
        print("\nMLflow:")
        print(f"  Username:         {self.mlflow.effective_username}")
        print(f"  Experiment:       {self.experiment_name}")
        print(f"  Champion AUC ≥:   {self.mlflow.champion_auc_threshold:.3f}")
        print(f"  Candidate AUC ≥:  {self.mlflow.candidate_auc_threshold:.3f}")
        
        print(self.section_divider)
