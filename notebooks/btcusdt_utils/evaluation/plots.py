"""Plotly visualization utilities for model evaluation.

All functions return ``go.Figure`` objects using the project
dark theme from ``btcusdt_utils.viz``.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
)
from sklearn.calibration import calibration_curve

from btcusdt_utils.viz import (
    _BG, _BG_CARD, _TEXT, _TEXT_MUT, _GRID,
    _RED, _GREEN, _BLUE, _ORANGE, _PURPLE, _CYAN,
    dark_fig,
)

_COLORS = [_BLUE, _GREEN, _ORANGE, _PURPLE, _CYAN, _RED]


def plot_roc_curves(
    y_test: np.ndarray,
    model_probs: dict[str, np.ndarray],
) -> go.Figure:
    """ROC curves for multiple models + diagonal reference."""
    fig = dark_fig()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color=_TEXT_MUT, width=1),
        name="Random (AUC=0.50)",
    ))
    for i, (name, y_prob) in enumerate(model_probs.items()):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={auc:.3f})",
            line=dict(color=_COLORS[i % len(_COLORS)], width=2),
        ))
    fig.update_layout(
        title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR",
        width=800, height=600, legend=dict(x=0.55, y=0.05),
    )
    return fig


def plot_precision_recall(
    y_test: np.ndarray,
    model_probs: dict[str, np.ndarray],
) -> go.Figure:
    """Precision–Recall curves for multiple models."""
    fig = dark_fig()
    for i, (name, y_prob) in enumerate(model_probs.items()):
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            name=f"{name} (AP={ap:.3f})",
            line=dict(color=_COLORS[i % len(_COLORS)], width=2),
        ))
    prevalence = float(np.mean(y_test))
    fig.add_hline(
        y=prevalence, line_dash="dash", line_color=_TEXT_MUT,
        annotation_text=f"Prevalence={prevalence:.3f}",
    )
    fig.update_layout(
        title="Precision–Recall Curves",
        xaxis_title="Recall", yaxis_title="Precision",
        width=800, height=600,
    )
    return fig


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Confusion Matrix",
    labels: tuple[str, str] = ("DOWN (0)", "UP (1)"),
) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm / cm.sum() * 100
    text = [
        [f"{cm[i][j]:,}<br>({cm_pct[i][j]:.1f}%)" for j in range(2)]
        for i in range(2)
    ]
    fig = dark_fig()
    fig.add_trace(go.Heatmap(
        z=cm, x=list(labels), y=list(labels),
        text=text, texttemplate="%{text}",
        colorscale=[[0, _BG], [1, _BLUE]], showscale=False,
    ))
    fig.update_layout(
        title=title, xaxis_title="Predicted", yaxis_title="Actual",
        width=500, height=500,
    )
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    *,
    top_n: int = 20,
    title: str = "Feature Importance (Top 20)",
) -> go.Figure:
    """Horizontal bar chart of feature importance."""
    idx = np.argsort(importances)[-top_n:]
    fig = dark_fig()
    fig.add_trace(go.Bar(
        x=importances[idx],
        y=[feature_names[i] for i in idx],
        orientation="h", marker_color=_BLUE,
    ))
    fig.update_layout(
        title=title, xaxis_title="Importance",
        width=900, height=max(400, top_n * 25),
        margin=dict(l=150),
    )
    return fig


def plot_calibration(
    y_test: np.ndarray,
    model_probs: dict[str, np.ndarray],
    *,
    n_bins: int = 10,
) -> go.Figure:
    """Calibration (reliability) curves for multiple models."""
    fig = dark_fig()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color=_TEXT_MUT, width=1),
        name="Perfectly calibrated",
    ))
    for i, (name, y_prob) in enumerate(model_probs.items()):
        prob_true, prob_pred = calibration_curve(
            y_test, y_prob, n_bins=n_bins, strategy="uniform",
        )
        fig.add_trace(go.Scatter(
            x=prob_pred, y=prob_true, mode="lines+markers",
            name=name,
            line=dict(color=_COLORS[i % len(_COLORS)], width=2),
            marker=dict(size=6),
        ))
    fig.update_layout(
        title="Calibration Curves",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        width=700, height=600,
    )
    return fig
