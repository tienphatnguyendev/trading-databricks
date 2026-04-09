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
]
