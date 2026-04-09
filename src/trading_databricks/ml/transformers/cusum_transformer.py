from __future__ import annotations

import polars as pl

from trading_databricks.ml.config import CUSUMConfig
from trading_databricks.ml.cusum import detect_cusum_events
from trading_databricks.ml.transformers.base import PolarsTransformerMixin


class CUSUMEventDetector(PolarsTransformerMixin):
    def __init__(self, config: CUSUMConfig | None = None) -> None:
        self.config = config or CUSUMConfig()

    def _fit_polars(
        self,
        df: pl.DataFrame,
        y: object = None,
        **fit_params: object,
    ) -> CUSUMEventDetector:
        return self

    def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        return detect_cusum_events(df, config=self.config)
