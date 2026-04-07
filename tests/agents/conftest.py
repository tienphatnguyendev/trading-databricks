import pytest
import pandas as pd
import numpy as np
import mlflow
import tempfile
from sklearn.base import BaseEstimator, ClassifierMixin

class DummyGatekeeperModel(BaseEstimator, ClassifierMixin):
    """A dummy sklearn model that always predicts class 1 (impactful) for testing."""
    def fit(self, X, y):
        self.classes_ = np.array([False, True])
        return self
        
    def predict_proba(self, X):
        # Return probability 1.0 for the positive class (index 1)
        # Handle single strings or iterables
        n_samples = len(X) if isinstance(X, (list, pd.Series, np.ndarray)) else 1
        return np.array([[0.0, 1.0]] * n_samples)
        
    def predict(self, X):
        n_samples = len(X) if isinstance(X, (list, pd.Series, np.ndarray)) else 1
        return np.array([True] * n_samples)

@pytest.fixture(scope="session")
def dummy_model_uri():
    """Creates a dummy MLflow model and returns its URI for testing."""
    model = DummyGatekeeperModel()
    model.fit(["dummy text"], [True])
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # We use file:// URI for local MLflow tracking during tests
        mlflow.set_tracking_uri(f"file://{tmp_dir}/mlruns")
        with mlflow.start_run():
            model_info = mlflow.sklearn.log_model(model, "dummy-model")
            yield model_info.model_uri
