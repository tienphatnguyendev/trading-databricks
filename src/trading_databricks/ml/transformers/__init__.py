"""sklearn-compatible transformer wrappers for the ML domain."""
from __future__ import annotations

from trading_databricks.ml.transformers.base import PolarsTransformerMixin
from trading_databricks.ml.transformers.cusum_transformer import CUSUMEventDetector
from trading_databricks.ml.transformers.feature_transformers import (
    NewsFeatureTransformer,
    TechnicalFeatureTransformer,
    VolatilityFeatureTransformer,
    VolumeFeatureTransformer,
)
from trading_databricks.ml.transformers.feature_union import PolarsFeatureUnion
from trading_databricks.ml.transformers.label_transformer import TripleBarrierLabeler
from trading_databricks.ml.transformers.numpy_bridge import PolarsToNumpy
from trading_databricks.ml.transformers.pipeline import build_training_pipeline
from trading_databricks.ml.transformers.search import build_hyperparameter_search
from trading_databricks.ml.transformers.mlflow_model import TradingPipelineModel

__all__ = [
    "PolarsTransformerMixin",
    "CUSUMEventDetector",
    "TripleBarrierLabeler",
    "TechnicalFeatureTransformer",
    "VolatilityFeatureTransformer",
    "VolumeFeatureTransformer",
    "NewsFeatureTransformer",
    "PolarsFeatureUnion",
    "PolarsToNumpy",
    "build_training_pipeline",
    "build_hyperparameter_search",
    "TradingPipelineModel",
]
