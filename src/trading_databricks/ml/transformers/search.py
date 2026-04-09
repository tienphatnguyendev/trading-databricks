from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from trading_databricks.ml.config import TrainingConfig
from trading_databricks.ml.training.cpcv import CPCVSplitter

def build_hyperparameter_search(
    pipeline: Pipeline,
    splitter: CPCVSplitter,
    config: TrainingConfig,
    param_distributions: dict[str, list] | None = None,
) -> RandomizedSearchCV:
    """Wraps the pipeline in a RandomizedSearchCV utilizing CPCV."""
    
    if param_distributions is None:
        param_distributions = {
            "classifier__n_estimators": [100, 200, 500],
            "classifier__max_depth": [3, 5, 10, None],
            "classifier__min_samples_leaf": [1, 5, 10, 20],
        }
        
    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=config.n_iter_search,
        cv=splitter,
        scoring="neg_log_loss",
        random_state=config.random_state,
        n_jobs=1,  # Polars is internally parallel; avoid oversubscription
    )
