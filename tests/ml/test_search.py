import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from trading_databricks.ml.config import CPCVConfig, TrainingConfig
from trading_databricks.ml.training.cpcv import CPCVSplitter
from trading_databricks.ml.transformers.search import build_hyperparameter_search


def test_build_hyperparameter_search():
    pipeline = Pipeline([("classifier", RandomForestClassifier())])
    cpcv_config = CPCVConfig(n_groups=3, k_test=1, embargo_pct=0.01)
    splitter = CPCVSplitter(
        config=cpcv_config,
        t_start=np.array([0, 1, 2]),
        t_end=np.array([1, 2, 3])
    )
    config = TrainingConfig(n_iter_search=5, random_state=42)

    search = build_hyperparameter_search(pipeline, splitter, config)

    assert search.n_iter == 5
    assert search.random_state == 42
    assert search.cv == splitter
    assert search.scoring == "neg_log_loss"
    assert "classifier__n_estimators" in search.param_distributions
