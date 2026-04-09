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
