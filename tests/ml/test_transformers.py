from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from trading_databricks.ml.transformers.base import PolarsTransformerMixin
from datetime import datetime
from trading_databricks.ml.config import CUSUMConfig
from trading_databricks.ml.cusum import detect_cusum_events

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

def test_cusum_transformer() -> None:
    from trading_databricks.ml.transformers.cusum_transformer import CUSUMEventDetector
    bars = pl.DataFrame({
        "bar_index": [1, 2, 3],
        "symbol": ["BTCUSDT"] * 3,
        "close": [100.0, 105.0, 110.0],
        "bar_end_time": [datetime(2026, 1, 1), datetime(2026, 1, 2), datetime(2026, 1, 3)]
    })
    
    config = CUSUMConfig(lookback_days=0)
    expected = detect_cusum_events(bars, config=config)
    
    detector = CUSUMEventDetector(config=config)
    result = detector.fit(bars).transform(bars)
    
    # drop non-deterministic columns
    assert result.drop(["event_id", "_ingested_at"]).equals(expected.drop(["event_id", "_ingested_at"]))
