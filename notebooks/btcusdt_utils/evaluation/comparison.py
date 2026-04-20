"""Model comparison table builder."""

from __future__ import annotations

from typing import Any

import pandas as pd

from btcusdt_utils.evaluation.metrics import BaselineResult


def build_comparison_table(
    model_results: dict[str, dict[str, Any]],
    baselines: dict[str, BaselineResult],
    *,
    best_model_name: str | None = None,
) -> pd.DataFrame:
    """Build a comparison table of baselines and trained models.

    Parameters
    ----------
    model_results   : {name: {"test_metrics": {...}}}
    baselines       : dict from :func:`compute_baselines`
    best_model_name : highlight this model with a star (★)

    Returns
    -------
    pd.DataFrame with Model, Type, Accuracy, F1, ROC-AUC.
    """
    rows: list[dict[str, str]] = []

    for bl in baselines.values():
        rows.append({
            "Model": bl.name,
            "Type": "Baseline",
            "Accuracy": f"{bl.metrics['accuracy']:.1%}",
            "F1": f"{bl.metrics['f1']:.3f}",
            "ROC-AUC": f"{bl.metrics['roc_auc']:.3f}",
        })

    for name, res in model_results.items():
        tm = res["test_metrics"]
        tag = " \u2605" if name == best_model_name else ""
        rows.append({
            "Model": f"{name}{tag}",
            "Type": "ML Model",
            "Accuracy": f"{tm['accuracy']:.1%}",
            "F1": f"{tm['f1']:.3f}",
            "ROC-AUC": f"{tm['roc_auc']:.3f}",
        })

    return pd.DataFrame(rows)
