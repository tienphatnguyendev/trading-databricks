from __future__ import annotations

import pickle

import mlflow
import pandas as pd
import polars as pl

from trading_databricks.ml.config import MLPipelineConfig


class TradingPipelineModel(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc wrapper for the trading pipeline.
    
    At inference, the predict method receives events (as a pandas DataFrame per MLflow contract),
    converts to Polars, runs the pipeline, and computes bet sizing.
    """

    def __init__(self, pipeline=None, config: MLPipelineConfig | None = None):
        self.pipeline = pipeline
        self.config = config

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Loads the serialized pipeline from MLflow artifacts."""
        with open(context.artifacts["pipeline"], "rb") as f:
            self.pipeline = pickle.load(f)

        with open(context.artifacts["config"], "rb") as f:
            self.config = pickle.load(f)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext | None,
        model_input: pd.DataFrame,
        params: dict | None = None
    ) -> pd.DataFrame:
        """Runs inference and bet sizing."""

        events = pl.from_pandas(model_input)

        # Get probability of class 1 (long)
        probas = self.pipeline.predict_proba(events)[:, 1]

        # Apply bet sizing logic
        bet_sizes = []
        for p in probas:
            # Simplified bet sizing for wrapper
            if self.config and p > self.config.bet_sizing.primary_threshold:
                bet_sizes.append(1.0)
            elif self.config and p < (1.0 - self.config.bet_sizing.primary_threshold):
                bet_sizes.append(-1.0)
            else:
                bet_sizes.append(0.0)

        result = model_input.copy()
        result["bet_size"] = bet_sizes
        result["probability_long"] = probas

        return result
