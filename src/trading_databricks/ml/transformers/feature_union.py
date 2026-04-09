"""PolarsFeatureUnion: runs multiple transformers and joins features on event_id."""
from __future__ import annotations

import polars as pl

from trading_databricks.ml.transformers.base import PolarsTransformerMixin


class PolarsFeatureUnion(PolarsTransformerMixin):
    """Run multiple PolarsTransformerMixin instances, joining results on event_id."""

    def __init__(
        self,
        transformers: list[tuple[str, PolarsTransformerMixin]],
    ) -> None:
        self.transformers = transformers

    def _fit_polars(
        self,
        df: pl.DataFrame,
        y: object = None,
        **fit_params: object,
    ) -> PolarsFeatureUnion:
        for _name, t in self.transformers:
            t.fit(df, y, **fit_params)
        return self

    def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        if "event_id" not in df.columns:
            raise ValueError(
                "PolarsFeatureUnion requires 'event_id' in the input DataFrame."
            )
        result = df.clone()
        existing_cols: set[str] = set(result.columns)
        for _name, t in self.transformers:
            features = t.transform(df)
            if "event_id" not in features.columns:
                raise ValueError(
                    f"Transformer '{_name}' must return a DataFrame with 'event_id'."
                )
            if features["event_id"].n_unique() != len(features):
                raise ValueError(
                    f"Transformer '{_name}' returned duplicate event_id values; "
                    "PolarsFeatureUnion requires one row per event_id."
                )
            # Only join new columns (avoid duplicating event_id, symbol, event_ts, etc.)
            new_cols = [
                c for c in features.columns
                if c not in existing_cols and c != "event_id"
            ]
            result = result.join(
                features.select(["event_id"] + new_cols),
                on="event_id",
                how="left",
            )
            existing_cols.update(new_cols)
        return result
