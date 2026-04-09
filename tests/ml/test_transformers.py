from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from trading_databricks.ml.config import CUSUMConfig, LabelConfig
from trading_databricks.ml.cusum import detect_cusum_events
from trading_databricks.ml.labeling import compute_triple_barrier_labels
from trading_databricks.ml.transformers.base import PolarsTransformerMixin


def test_base_transformer_fit_stores_columns() -> None:
    class DummyTransformer(PolarsTransformerMixin):
        def _fit_polars(
            self,
            df: pl.DataFrame,
            y: object = None,
            **fit_params: object,
        ) -> PolarsTransformerMixin:
            return self

        def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
            return df

    t = DummyTransformer()
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    t.fit(df)
    assert t.feature_names_in_ == ["a", "b"]


def test_base_transformer_numpy_roundtrip() -> None:
    class DummyTransformer(PolarsTransformerMixin):
        def _fit_polars(
            self,
            df: pl.DataFrame,
            y: object = None,
            **fit_params: object,
        ) -> PolarsTransformerMixin:
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
        "bar_end_time": [
            datetime(2026, 1, 1),
            datetime(2026, 1, 2),
            datetime(2026, 1, 3),
        ],
    })

    config = CUSUMConfig(lookback_days=0)
    expected = detect_cusum_events(bars, config=config)

    detector = CUSUMEventDetector(config=config)
    result = detector.fit(bars).transform(bars)

    # drop non-deterministic columns
    assert result.drop(["event_id", "_ingested_at"]).equals(
        expected.drop(["event_id", "_ingested_at"])
    )


def test_label_transformer() -> None:
    from trading_databricks.ml.transformers.label_transformer import TripleBarrierLabeler
    events = pl.DataFrame({
        "event_id": ["e1"],
        "symbol": ["BTCUSDT"],
        "event_ts": [datetime(2026, 1, 1)],
        "bar_index": [1],
        "log_return": [0.01],
        "cusum_side": ["up"],
        "cusum_value": [0.05],
        "threshold_used": [0.02],
        "ewma_vol": [0.01],
        "avg_bars_per_day": [10.0],
    })
    bars = pl.DataFrame({
        "bar_index": [1, 2],
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "close": [100.0, 110.0],
        "bar_end_time": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
    })

    config = LabelConfig(vertical_barrier_days=1)
    expected = compute_triple_barrier_labels(events, bars, config=config)

    labeler = TripleBarrierLabeler(config=config)
    result = labeler.fit(events, bars=bars).transform(events)

    assert result.equals(expected)


def test_label_transformer_missing_bars() -> None:
    from trading_databricks.ml.transformers.label_transformer import TripleBarrierLabeler
    events = pl.DataFrame({"a": [1]})
    labeler = TripleBarrierLabeler()
    with pytest.raises(ValueError, match="requires 'bars' passed to fit"):
        labeler.fit(events)
