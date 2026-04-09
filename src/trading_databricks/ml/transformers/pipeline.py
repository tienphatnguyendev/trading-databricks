from __future__ import annotations

import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from trading_databricks.ml.config import MLPipelineConfig
from trading_databricks.ml.training.cpcv import CPCVSplitter
from trading_databricks.ml.transformers import (
    PolarsFeatureUnion,
    PolarsToNumpy,
    TechnicalFeatureTransformer,
    VolatilityFeatureTransformer,
    VolumeFeatureTransformer,
)


def build_training_pipeline(
    config: MLPipelineConfig,
    bars: pl.DataFrame,
    events: pl.DataFrame,
    labels: pl.DataFrame,
) -> tuple[Pipeline, CPCVSplitter]:
    """Builds the sklearn Pipeline and CPCV splitter for training."""

    feature_union = PolarsFeatureUnion([
        ("technical", TechnicalFeatureTransformer(config.feature)),
        ("volatility", VolatilityFeatureTransformer(config.feature)),
        ("volume", VolumeFeatureTransformer(config.feature)),
    ])

    # Fit the union with the bars context
    feature_union.fit(events, y=None, bars=bars)

    pipeline = Pipeline([
        ("features", feature_union),
        ("to_numpy", PolarsToNumpy()),
        ("classifier", RandomForestClassifier(random_state=config.training.random_state)),
    ])

    # Ensure CPCVSplitter gets integer arrays
    t_start = labels["bar_index"].to_numpy()
    t_end = labels["vertical_bar"].to_numpy()

    splitter = CPCVSplitter(
        config=config.cpcv,
        t_start=t_start,
        t_end=t_end
    )

    return pipeline, splitter
