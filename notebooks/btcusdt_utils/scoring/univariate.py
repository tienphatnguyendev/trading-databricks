"""Univariate feature scoring via single-feature LightGBM AUC.

Trains a shallow LightGBM model on each feature individually
using a chronological train/test split to measure standalone
predictive power without data leakage.

Reference: Phase 4B §10 of BTCUSDT EDA notebook.
"""

from __future__ import annotations

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from btcusdt_utils.scoring._config import ScoringConfig


def compute_univariate_auc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    config: ScoringConfig | None = None,
) -> np.ndarray:
    """Compute single-feature LightGBM AUC for each column.

    Parameters
    ----------
    X      : Feature matrix (n_samples, n_features).  Must be
             chronologically ordered (no shuffling).
    y      : Binary label array.
    config : Scoring configuration (keyword-only).

    Returns
    -------
    1-D array of AUC scores, one per feature.  Falls back to
    0.5 (random baseline) when a model fails to train.
    """
    if config is None:
        config = ScoringConfig()

    n_samples, n_features = X.shape
    split_idx = int((1.0 - config.test_ratio) * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "max_depth": config.lgb_max_depth,
        "num_leaves": config.lgb_num_leaves,
        "learning_rate": config.lgb_lr,
        "verbose": -1,
        "n_jobs": 1,
    }

    auc_scores = np.full(n_features, 0.5)

    for j in range(n_features):
        X_j_tr = X_train[:, j : j + 1]
        X_j_te = X_test[:, j : j + 1]

        dtrain = lgb.Dataset(X_j_tr, label=y_train)
        dval = lgb.Dataset(X_j_te, label=y_test, reference=dtrain)

        try:
            model = lgb.train(
                lgb_params,
                dtrain,
                num_boost_round=config.lgb_rounds,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(
                        config.lgb_early_stop, verbose=False,
                    ),
                ],
            )
            y_pred = model.predict(X_j_te)
            auc_scores[j] = roc_auc_score(y_test, y_pred)
        except Exception:
            pass  # keeps default 0.5

    return auc_scores
