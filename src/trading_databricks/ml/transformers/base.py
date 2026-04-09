from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


class PolarsTransformerMixin(BaseEstimator, TransformerMixin):
    def fit(
        self,
        X: pl.DataFrame | np.ndarray,
        y: object = None,
        **fit_params: object,
    ) -> PolarsTransformerMixin:
        if isinstance(X, pl.DataFrame):
            self.feature_names_in_ = X.columns
        elif isinstance(X, np.ndarray):
            self.feature_names_in_ = [f"f{i}" for i in range(X.shape[1])]
        else:
            raise TypeError("Input must be a Polars DataFrame or NumPy array")
        return self._fit_polars(self._to_polars(X), y, **fit_params)

    def transform(
        self,
        X: pl.DataFrame | np.ndarray,
    ) -> pl.DataFrame:
        return self._transform_polars(self._to_polars(X))

    def _to_polars(self, X: pl.DataFrame | np.ndarray) -> pl.DataFrame:
        if isinstance(X, pl.DataFrame):
            return X
        if not hasattr(self, "feature_names_in_"):
            raise ValueError("Estimator not fitted yet.")
        return pl.DataFrame(X, schema=self.feature_names_in_)

    def _fit_polars(
        self,
        df: pl.DataFrame,
        y: object = None,
        **fit_params: object,
    ) -> PolarsTransformerMixin:
        return self

    def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        return df
