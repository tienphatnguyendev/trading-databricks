import pickle
from unittest.mock import MagicMock

import pandas as pd

from trading_databricks.ml.config import MLPipelineConfig
from trading_databricks.ml.transformers.mlflow_model import TradingPipelineModel


class DummyPipeline:
    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.2, 0.8]] * len(X))

def test_mlflow_model_predict():
    config = MLPipelineConfig()
    pipeline = DummyPipeline()
    model = TradingPipelineModel(pipeline=pipeline, config=config)

    input_df = pd.DataFrame({"feature1": [1.0, 2.0]})
    # context is None in direct calls unless loaded via MLflow
    result = model.predict(context=None, model_input=input_df)

    assert "bet_size" in result.columns
    assert len(result) == 2

def test_mlflow_model_load_context(tmp_path):
    """Verify load_context correctly initializes pipeline and config."""
    pipeline = DummyPipeline()
    config = MLPipelineConfig()

    pipe_path = tmp_path / "pipeline.pkl"
    config_path = tmp_path / "config.pkl"

    with open(pipe_path, "wb") as f:
        pickle.dump(pipeline, f)
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    context = MagicMock()
    context.artifacts = {
        "pipeline": str(pipe_path),
        "config": str(config_path)
    }

    model = TradingPipelineModel()
    model.load_context(context)

    assert model.pipeline is not None
    assert model.config is not None
