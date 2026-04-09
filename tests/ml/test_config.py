import pytest
from trading_databricks.ml.config import (
    CUSUMConfig, LabelConfig, FeatureConfig, CPCVConfig, 
    BetSizingConfig, TrainingConfig, InferenceConfig, MLPipelineConfig
)

def test_cusum_config_defaults():
    cfg = CUSUMConfig()
    assert cfg.ewma_lambda == 0.94
    assert cfg.threshold_multiplier == 1.0
    assert cfg.min_inter_event_bars == 5
    assert cfg.lookback_days == 30

def test_feature_config_defaults():
    cfg = FeatureConfig()
    assert cfg.technical_windows == (5, 10, 14, 20, 26, 50)
    assert cfg.news_lookback_hours == 24

def test_ml_pipeline_config_composition():
    cfg = MLPipelineConfig()
    assert cfg.symbol == "BTCUSDT"
    assert cfg.catalog == "trading-signal-gen-dev"
    assert isinstance(cfg.cusum, CUSUMConfig)
    assert isinstance(cfg.feature, FeatureConfig)

def test_configs_are_frozen():
    cfg = CUSUMConfig()
    with pytest.raises(Exception):
        cfg.ewma_lambda = 0.99  # type: ignore
