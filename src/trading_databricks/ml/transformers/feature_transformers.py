from __future__ import annotations

import polars as pl

from trading_databricks.ml.config import FeatureConfig
from trading_databricks.ml.features import compute_news, compute_technical, compute_volatility, compute_volume
from trading_databricks.ml.transformers.base import PolarsTransformerMixin

class TechnicalFeatureTransformer(PolarsTransformerMixin):
    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
    
    def _fit_polars(self, df: pl.DataFrame, y=None, **fit_params):
        if "bars" not in fit_params:
            raise ValueError("TechnicalFeatureTransformer requires 'bars' passed to fit().")
        self.bars_ = fit_params["bars"]
        return self
    
    def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        if not hasattr(self, "bars_"):
            raise ValueError("Estimator not fitted yet.")
        return compute_technical(df.lazy(), self.bars_.lazy(), config=self.config).collect()

class VolatilityFeatureTransformer(PolarsTransformerMixin):
    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
    
    def _fit_polars(self, df: pl.DataFrame, y=None, **fit_params):
        if "bars" not in fit_params:
            raise ValueError("VolatilityFeatureTransformer requires 'bars' passed to fit().")
        self.bars_ = fit_params["bars"]
        return self
    
    def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        if not hasattr(self, "bars_"):
            raise ValueError("Estimator not fitted yet.")
        return compute_volatility(df.lazy(), self.bars_.lazy(), config=self.config).collect()

class VolumeFeatureTransformer(PolarsTransformerMixin):
    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
    
    def _fit_polars(self, df: pl.DataFrame, y=None, **fit_params):
        if "bars" not in fit_params:
            raise ValueError("VolumeFeatureTransformer requires 'bars' passed to fit().")
        self.bars_ = fit_params["bars"]
        return self
    
    def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        if not hasattr(self, "bars_"):
            raise ValueError("Estimator not fitted yet.")
        return compute_volume(df.lazy(), self.bars_.lazy(), config=self.config).collect()

class NewsFeatureTransformer(PolarsTransformerMixin):
    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
    
    def _fit_polars(self, df: pl.DataFrame, y=None, **fit_params):
        # news_features=None is valid — compute_news handles it with zero-fill
        self.news_features_ = fit_params.get("news_features", None)
        self._fitted = True
        return self

    def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        if not hasattr(self, "_fitted"):
            raise ValueError("Estimator not fitted yet.")
        news_lf = self.news_features_.lazy() if self.news_features_ is not None else None
        return compute_news(df.lazy(), news_lf, config=self.config).collect()
