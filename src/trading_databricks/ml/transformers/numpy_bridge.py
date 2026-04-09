"""PolarsToNumpy: strips metadata columns and returns a NumPy array."""
from __future__ import annotations

import numpy as np
import polars as pl

from trading_databricks.ml.transformers.base import PolarsTransformerMixin


class PolarsToNumpy(PolarsTransformerMixin):
    """Strip metadata columns and return a NumPy ndarray for sklearn classifiers."""

    _METADATA_COLS: frozenset[str] = frozenset({
        "event_id", "symbol", "event_ts", "computed_at", "bar_index",
        "bar_end_time", "cpcv_group", "dataset_version",
    })

    def _fit_polars(
        self,
        df: pl.DataFrame,
        y: object = None,
        **fit_params: object,
    ) -> PolarsToNumpy:
        self.output_features_ = [c for c in df.columns if c not in self._METADATA_COLS]
        return self

    def _transform_polars(self, df: pl.DataFrame) -> np.ndarray:  # type: ignore[override]
        if not hasattr(self, "output_features_"):
            raise ValueError("Estimator not fitted yet.")
        return df.select(self.output_features_).to_numpy()
