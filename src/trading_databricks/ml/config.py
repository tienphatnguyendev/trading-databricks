from dataclasses import dataclass, field


@dataclass(frozen=True)
class CUSUMConfig:
    ewma_lambda: float = 0.94
    threshold_multiplier: float = 1.0
    min_inter_event_bars: int = 5
    lookback_days: int = 30

@dataclass(frozen=True)
class LabelConfig:
    upper_barrier_multiplier: float = 2.0
    lower_barrier_multiplier: float = 1.0
    vertical_barrier_days: int = 5
    zero_label_threshold: float = 0.1

@dataclass(frozen=True)
class FeatureConfig:
    technical_windows: tuple[int, ...] = (5, 10, 14, 20, 26, 50)
    volatility_windows: tuple[int, ...] = (20, 50)
    volume_windows: tuple[int, ...] = (5, 20)
    news_lookback_hours: int = 24

@dataclass(frozen=True)
class CPCVConfig:
    n_groups: int = 6
    k_test: int = 2
    embargo_pct: float = 0.01

@dataclass(frozen=True)
class BetSizingConfig:
    meta_threshold: float = 0.50
    primary_threshold: float = 0.55
    tiers: tuple[tuple[float, float], ...] = (
        (0.10, 0.00), (0.25, 0.15), (0.50, 0.35), (0.75, 0.60), (1.01, 0.85)
    )

@dataclass(frozen=True)
class TrainingConfig:
    algorithm: str = "RandomForestClassifier"
    n_iter_search: int = 50
    random_state: int = 42

@dataclass(frozen=True)
class InferenceConfig:
    cadence_hours: int = 4

@dataclass(frozen=True)
class MLPipelineConfig:
    cusum: CUSUMConfig = field(default_factory=CUSUMConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    cpcv: CPCVConfig = field(default_factory=CPCVConfig)
    bet_sizing: BetSizingConfig = field(default_factory=BetSizingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    symbol: str = "BTCUSDT"
    catalog: str = "trading-signal-gen-dev"
