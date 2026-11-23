"""Classification metrics and naive baseline strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute classification metrics.

    Returns dict with accuracy, precision, recall, f1, roc_auc, log_loss.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    if y_prob is None:
        y_prob = y_pred.astype(float)

    try:
        auc = (
            0.5 if len(np.unique(y_prob)) < 2
            else roc_auc_score(y_true, y_prob)
        )
    except ValueError:
        auc = 0.5

    try:
        ll = log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15))
    except ValueError:
        ll = float("nan")

    return {
        "accuracy": round(acc, 5),
        "precision": round(prec, 5),
        "recall": round(rec, 5),
        "f1": round(f1, 5),
        "roc_auc": round(auc, 5),
        "log_loss": round(ll, 5),
    }


@dataclass(frozen=True)
class BaselineResult:
    """Structured result for a single naive baseline strategy."""

    name: str
    y_pred: np.ndarray
    metrics: dict[str, float]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"acc={self.metrics['accuracy']:.3f}, "
            f"f1={self.metrics['f1']:.3f}, "
            f"auc={self.metrics['roc_auc']:.3f})"
        )


def compute_baselines(
    y_test: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int = 42,
) -> dict[str, BaselineResult]:
    """Compute 5 naive baseline strategies.

    Strategies: Majority Class, Random Coin Flip, Weighted Random,
    Last Label (persistence), Always Positive.
    """
    rng = np.random.default_rng(seed)
    n_test = len(y_test)
    train_counts = np.bincount(y_train, minlength=2)
    train_freq = train_counts / len(y_train)
    majority_class = int(np.argmax(train_counts))

    results: dict[str, BaselineResult] = {}

    # 1. Majority Class
    y_maj = np.full(n_test, majority_class, dtype=int)
    results["Majority Class"] = BaselineResult(
        name="Majority Class", y_pred=y_maj,
        metrics=compute_metrics(y_test, y_maj),
    )

    # 2. Random Coin Flip
    y_coin = rng.integers(0, 2, size=n_test)
    results["Random Coin Flip"] = BaselineResult(
        name="Random Coin Flip", y_pred=y_coin,
        metrics=compute_metrics(y_test, y_coin, rng.random(size=n_test)),
    )

    # 3. Weighted Random
    y_w = rng.choice([0, 1], size=n_test, p=train_freq)
    y_s_w = np.clip(train_freq[1] + 0.1 * (rng.random(n_test) - 0.5), 0, 1)
    results["Weighted Random"] = BaselineResult(
        name="Weighted Random", y_pred=y_w,
        metrics=compute_metrics(y_test, y_w, y_s_w),
    )

    # 4. Last Label (persistence)
    y_persist = np.empty(n_test, dtype=int)
    y_persist[0] = y_train[-1]
    y_persist[1:] = y_test[:-1]
    results["Last Label"] = BaselineResult(
        name="Last Label", y_pred=y_persist,
        metrics=compute_metrics(y_test, y_persist),
    )

    # 5. Always Positive
    y_pos = np.ones(n_test, dtype=int)
    results["Always Positive"] = BaselineResult(
        name="Always Positive", y_pred=y_pos,
        metrics=compute_metrics(y_test, y_pos),
    )

    return results
