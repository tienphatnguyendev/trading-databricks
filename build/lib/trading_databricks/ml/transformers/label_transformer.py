from __future__ import annotations

import polars as pl

from trading_databricks.ml.config import LabelConfig
from trading_databricks.ml.labeling import compute_triple_barrier_labels
from trading_databricks.ml.transformers.base import PolarsTransformerMixin


class TripleBarrierLabeler(PolarsTransformerMixin):
    def __init__(self, config: LabelConfig | None = None) -> None:
        self.config = config or LabelConfig()

    def _fit_polars(
        self,
        df: pl.DataFrame,
        y: object = None,
        **fit_params: object,
    ) -> TripleBarrierLabeler:
        if "bars" not in fit_params:
            raise ValueError("TripleBarrierLabeler requires 'bars' passed to fit().")
        self.bars_ = fit_params["bars"]  # type: ignore[assignment]
        return self

    def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        if not hasattr(self, "bars_"):
            raise ValueError("Estimator not fitted yet.")
        return compute_triple_barrier_labels(df, self.bars_, config=self.config)
