from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from trading_databricks.ml.transformers.base import PolarsTransformerMixin

def test_base_transformer_fit_stores_columns() -> None:
    class DummyTransformer(PolarsTransformerMixin):
        def _fit_polars(self, df: pl.DataFrame, y=None, **fit_params):
            return self
        def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
            return df
    
    t = DummyTransformer()
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    t.fit(df)
    assert t.feature_names_in_ == ["a", "b"]

def test_base_transformer_numpy_roundtrip() -> None:
    class DummyTransformer(PolarsTransformerMixin):
        def _fit_polars(self, df: pl.DataFrame, y=None, **fit_params):
            return self
        def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns((pl.col("f0") * 2).alias("f0"))
            
    t = DummyTransformer()
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    t.fit(arr)
    assert t.feature_names_in_ == ["f0", "f1"]
    
    result = t.transform(arr)
    assert isinstance(result, pl.DataFrame)
    assert result.select("f0").to_series().to_list() == [2.0, 6.0]
