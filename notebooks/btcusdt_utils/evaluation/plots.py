"""Plotly visualization utilities for model evaluation.

All functions return ``go.Figure`` objects using the project
dark theme from ``btcusdt_utils.viz``.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED VISUALIZATION UTILITIES (AFML Ch. 7 - Cross-Validation Analysis)
# ═══════════════════════════════════════════════════════════════════════════════
# Extracted from: BTCUSDT EDA Parameter Optimization, Phases 6C & 7C
# Author: Databricks AI Assistant
# Date: 2026-05-01

def plot_cv_comparison(
    wf_results: dict[str, list[dict]],
    purged_aucs: np.ndarray | None = None,
    unpurged_aucs: np.ndarray | None = None,
    mdi_rank: np.ndarray | None = None,
    mi_rank_lookup: dict | None = None,
    all_feature_names: list[str] | None = None,
    n_splits: int = 5,
    title: str = "Cross-Validation Comparison",
    colors: dict[str, str] | None = None,
) -> go.Figure:
    """Create 2×2 dashboard comparing walk-forward CV with K-fold CV results.
    
    Panels:
      (1,1) Walk-forward AUC by split (all feature sets)
      (1,2) Feature importance: MDI rank vs MI rank scatter (top 20)
      (2,1) Purged vs Unpurged K-fold AUC comparison
      (2,2) Walk-forward AUC trend with linear fits
    
    Args:
        wf_results: Walk-forward results dict, format:
            {"Feature Set Name": [{"split": 0, "auc": 0.52, ...}, ...]}
        purged_aucs: K-fold AUCs with purging, shape (n_splits,) (optional)
        unpurged_aucs: K-fold AUCs without purging, shape (n_splits,) (optional)
        mdi_rank: Feature indices sorted by MDI importance, shape (n_features,) (optional)
        mi_rank_lookup: Dict mapping feature idx to MI rank (optional)
        all_feature_names: Feature names list (optional)
        n_splits: Number of K-fold splits (default: 5)
        title: Dashboard title (default: "Cross-Validation Comparison")
        colors: Color mapping for feature sets (optional)
    
    Returns:
        Plotly Figure object (2×2 subplots)
    
    Example:
        >>> wf_results = {
        ...     "Full": [{"split": 0, "auc": 0.50}, {"split": 1, "auc": 0.52}],
        ...     "Top-10": [{"split": 0, "auc": 0.51}, {"split": 1, "auc": 0.53}],
        ... }
        >>> fig = plot_cv_comparison(wf_results)
        >>> fig.show()
    """
    if colors is None:
        colors = {"Full": "#58A6FF", "Top-10": "#52C41A", "Top-21": "#F0883E"}
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Walk-Forward AUC by Split",
            "Feature Importance: MDI vs MI Rank (Top 20)",
            "Purged vs Unpurged K-Fold AUC",
            "Walk-Forward AUC Trend (Signal Stability)",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )
    
    # ── Panel (1,1): Walk-forward AUC by split ──────────────────
    for fs_name, fs_res in wf_results.items():
        aucs = [r["auc"] for r in fs_res]
        color = colors.get(fs_name, "#58A6FF")
        fig.add_trace(
            go.Scatter(
                x=[f"Split {r['split']}" for r in fs_res],
                y=aucs,
                mode="lines+markers",
                name=fs_name,
                line=dict(color=color, width=2),
                marker=dict(size=8),
                showlegend=True,
            ),
            row=1, col=1,
        )
    
    # Random baseline
    fig.add_hline(
        y=0.5, row=1, col=1,
        line_dash="dash", line_color="#8B949E", line_width=1.5,
        annotation_text="Random (0.50)",
        annotation_font_color="#8B949E",
    )
    
    # ── Panel (1,2): MDI vs MI rank scatter ─────────────────────
    if mdi_rank is not None and mi_rank_lookup is not None and all_feature_names is not None:
        top20_mdi = mdi_rank[:20]
        mdi_ranks_plot = np.arange(1, 21)
        mi_ranks_plot = [mi_rank_lookup.get(idx, len(mdi_rank) - 1) + 1 for idx in top20_mdi]
        feat_names_plot = [all_feature_names[idx] for idx in top20_mdi]
        
        fig.add_trace(
            go.Scatter(
                x=mdi_ranks_plot,
                y=mi_ranks_plot,
                mode="markers+text",
                text=[n[:12] for n in feat_names_plot],
                textposition="top center",
                textfont=dict(size=8, color="#E6EDF3"),
                marker=dict(
                    size=10,
                    color=["#52C41A" if mi <= 21 else "#FF4D4F" for mi in mi_ranks_plot],
                    opacity=0.8,
                ),
                showlegend=False,
            ),
            row=1, col=2,
        )
        # Diagonal reference
        fig.add_trace(
            go.Scatter(
                x=[1, 20], y=[1, 20],
                mode="lines",
                line=dict(color="#8B949E", dash="dash", width=1),
                showlegend=False,
            ),
            row=1, col=2,
        )
    
    # ── Panel (2,1): Purged vs Unpurged K-fold ──────────────────
    if purged_aucs is not None and unpurged_aucs is not None:
        fold_labels = [f"Fold {k}" for k in range(n_splits)]
        
        fig.add_trace(
            go.Bar(
                x=fold_labels, y=purged_aucs,
                name="Purged",
                marker_color="#58A6FF",
                showlegend=True,
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=fold_labels, y=unpurged_aucs,
                name="Unpurged",
                marker_color="#F0883E",
                opacity=0.7,
                showlegend=True,
            ),
            row=2, col=1,
        )
        fig.add_hline(
            y=0.5, row=2, col=1,
            line_dash="dash", line_color="#8B949E", line_width=1.5,
        )
    
    # ── Panel (2,2): AUC trend with linear fits ─────────────────
    for fs_name, fs_res in wf_results.items():
        aucs = [r["auc"] for r in fs_res]
        splits = list(range(len(aucs)))
        color = colors.get(fs_name, "#58A6FF")
        
        # Linear fit
        coefs = np.polyfit(splits, aucs, 1)
        trend_line = np.polyval(coefs, splits)
        
        fig.add_trace(
            go.Scatter(
                x=[f"WF {s}" for s in splits],
                y=aucs,
                mode="markers",
                marker=dict(size=10, color=color),
                showlegend=False,
            ),
            row=2, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[f"WF {s}" for s in splits],
                y=trend_line.tolist(),
                mode="lines",
                line=dict(color=color, dash="dash", width=1.5),
                name=f"{fs_name} trend",
                showlegend=False,
            ),
            row=2, col=2,
        )
    
    fig.add_hline(
        y=0.5, row=2, col=2,
        line_dash="dash", line_color="#8B949E", line_width=1.5,
    )
    
    # ── Layout ──────────────────────────────────────────────────
    fig.update_layout(
        title_text=title,
        title_font_color="#E6EDF3",
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        font_color="#E6EDF3",
        height=800,
        width=1200,
        barmode="group",
        legend=dict(
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor="#21262D",
            font_color="#E6EDF3",
        ),
    )
    fig.update_xaxes(gridcolor="#21262D")
    fig.update_yaxes(gridcolor="#21262D")
    
    # Axis labels
    fig.update_xaxes(title_text="Split", row=1, col=1)
    fig.update_yaxes(title_text="AUC", row=1, col=1)
    fig.update_xaxes(title_text="MDI Rank", row=1, col=2)
    fig.update_yaxes(title_text="MI Rank", row=1, col=2)
    fig.update_xaxes(title_text="Fold", row=2, col=1)
    fig.update_yaxes(title_text="AUC", row=2, col=1)
    fig.update_xaxes(title_text="Walk-Forward Split", row=2, col=2)
    fig.update_yaxes(title_text="AUC", row=2, col=2)
    
    return fig


def plot_meta_labeling_dashboard(
    meta_prob: np.ndarray,
    meta_labels: np.ndarray,
    trade_returns: np.ndarray,
    bet_sweep: list[dict],
    persist_acc: float,
    auc_progression: dict[str, float] | None = None,
    best_threshold: float | None = None,
    title: str = "Meta-Labeling Analysis",
) -> go.Figure:
    """Create 2×2 dashboard for meta-labeling model analysis.
    
    Panels:
      (1,1) Equity curve: always trade vs meta-filtered
      (1,2) Win rate vs trade fraction (bet sizing sweep)
      (2,1) Meta-probability distribution by correctness
      (2,2) AUC progression across phases
    
    Args:
        meta_prob: Meta-label probabilities (P(primary correct)), shape (N,)
        meta_labels: Actual meta-labels (1=correct, 0=wrong), shape (N,)
        trade_returns: Trade returns (+1 if correct, -1 if wrong), shape (N,)
        bet_sweep: Bet sizing sweep results, list of dicts with keys:
            {"threshold", "frac", "win_rate", "sharpe", ...}
        persist_acc: Persistence model accuracy (for reference line)
        auc_progression: Dict mapping phase names to AUC values (optional)
        best_threshold: Optimal threshold from bet sweep (for equity curve)
        title: Dashboard title
    
    Returns:
        Plotly Figure object (2×2 subplots)
    
    Example:
        >>> meta_prob = model.predict_proba(X_test)[:, 1]
        >>> meta_labels = (y_pred == y_test).astype(int)
        >>> trade_returns = np.where(y_pred == y_test, 1.0, -1.0)
        >>> bet_sweep = [
        ...     {"threshold": 0.5, "frac": 1.0, "win_rate": 0.87, "sharpe": 1.2},
        ...     {"threshold": 0.7, "frac": 0.8, "win_rate": 0.90, "sharpe": 1.5},
        ... ]
        >>> fig = plot_meta_labeling_dashboard(
        ...     meta_prob, meta_labels, trade_returns, bet_sweep, persist_acc=0.87
        ... )
        >>> fig.show()
    """
    if best_threshold is None:
        best_threshold = max(bet_sweep, key=lambda b: b["sharpe"])["threshold"]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Equity Curve: Always Trade vs Meta-Filtered",
            "Win Rate vs Trade Fraction",
            "Meta-Label Probability Distribution",
            "AUC Progression (Phases)",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )
    
    # ── Panel (1,1): Equity curves ──────────────────────────────
    mask_filtered = meta_prob >= best_threshold
    eq_always = np.cumsum(trade_returns)
    eq_filtered = np.cumsum(np.where(mask_filtered, trade_returns, 0.0))
    
    bar_idx = np.arange(len(meta_prob))
    fig.add_trace(
        go.Scatter(
            x=bar_idx, y=eq_always,
            mode="lines", name="Always Trade",
            line=dict(color="#8B949E", width=1.5),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=bar_idx, y=eq_filtered,
            mode="lines", name=f"Meta-Filtered (t>={best_threshold:.2f})",
            line=dict(color="#52C41A", width=2),
        ),
        row=1, col=1,
    )
    
    # ── Panel (1,2): Win rate vs trade fraction ─────────────────
    fig.add_trace(
        go.Scatter(
            x=[b["frac"] for b in bet_sweep],
            y=[b["win_rate"] for b in bet_sweep],
            mode="lines+markers",
            marker=dict(size=6, color="#58A6FF"),
            line=dict(color="#58A6FF", width=2),
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_hline(
        y=persist_acc, row=1, col=2,
        line_dash="dash", line_color="#F0883E",
        annotation_text=f"Persist {persist_acc:.1%}",
        annotation_font_color="#F0883E",
    )
    
    # ── Panel (2,1): Probability distribution ───────────────────
    fig.add_trace(
        go.Histogram(
            x=meta_prob[meta_labels == 1],
            name="Correct (meta=1)",
            marker_color="#52C41A",
            opacity=0.7,
            nbinsx=30,
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=meta_prob[meta_labels == 0],
            name="Wrong (meta=0)",
            marker_color="#FF4D4F",
            opacity=0.7,
            nbinsx=30,
        ),
        row=2, col=1,
    )
    
    # ── Panel (2,2): AUC progression ────────────────────────────
    if auc_progression is not None:
        fig.add_trace(
            go.Bar(
                x=list(auc_progression.keys()),
                y=list(auc_progression.values()),
                marker_color=["#8B949E", "#8B949E", "#F0883E", "#58A6FF", "#52C41A"][:len(auc_progression)],
                showlegend=False,
            ),
            row=2, col=2,
        )
        fig.add_hline(
            y=0.5, row=2, col=2,
            line_dash="dash", line_color="#8B949E",
        )
    
    # ── Layout ──────────────────────────────────────────────────
    fig.update_layout(
        title_text=title,
        title_font_color="#E6EDF3",
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        font_color="#E6EDF3",
        height=800,
        width=1200,
        barmode="overlay",
        legend=dict(
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor="#21262D",
        ),
    )
    fig.update_xaxes(gridcolor="#21262D")
    fig.update_yaxes(gridcolor="#21262D")
    
    # Axis labels
    fig.update_xaxes(title_text="Bar Index", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_xaxes(title_text="Trade Fraction", row=1, col=2)
    fig.update_yaxes(title_text="Win Rate", row=1, col=2)
    fig.update_xaxes(title_text="Meta-Probability", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Phase", row=2, col=2)
    fig.update_yaxes(title_text="AUC", row=2, col=2)
    
    return fig


def plot_regime_analysis(
    split_indices: list[tuple[np.ndarray, np.ndarray, int, int]],
    y_labels: np.ndarray,
    returns: np.ndarray | None = None,
    split_names: list[str] | None = None,
    title: str = "Regime Analysis by Split",
) -> go.Figure:
    """Analyze regime characteristics across CV splits.
    
    Plots label balance and (optionally) volatility for each split to identify
    regime shifts that affect model performance.
    
    Args:
        split_indices: List of (train_idx, test_idx, n_purged, n_embargo) tuples
        y_labels: Label array, shape (N,)
        returns: Return array for volatility analysis, shape (N,) (optional)
        split_names: Custom split names (default: "Split 0", "Split 1", ...)
        title: Plot title
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> split_indices = [
        ...     (train_0, test_0, 10, 20),
        ...     (train_1, test_1, 12, 18),
        ... ]
        >>> fig = plot_regime_analysis(split_indices, y_labels, returns)
        >>> fig.show()
    """
    n_splits = len(split_indices)
    if split_names is None:
        split_names = [f"Split {i}" for i in range(n_splits)]
    
    # Compute label balance and volatility per split
    label_balance = []
    volatilities = []
    
    for _, test_idx, _, _ in split_indices:
        y_test = y_labels[test_idx]
        balance = float(y_test.mean())
        label_balance.append(balance)
        
        if returns is not None:
            ret_test = returns[test_idx]
            vol = float(ret_test.std())
            volatilities.append(vol)
    
    # Create figure
    if returns is not None:
        # Two subplots: label balance + volatility
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Label Balance by Split", "Volatility by Split"],
            horizontal_spacing=0.15,
        )
        
        # Label balance
        fig.add_trace(
            go.Bar(
                x=split_names,
                y=label_balance,
                marker_color="#58A6FF",
                name="Label Balance",
            ),
            row=1, col=1,
        )
        fig.add_hline(
            y=0.5, row=1, col=1,
            line_dash="dash", line_color="#8B949E",
        )
        
        # Volatility
        fig.add_trace(
            go.Bar(
                x=split_names,
                y=volatilities,
                marker_color="#F0883E",
                name="Volatility",
            ),
            row=1, col=2,
        )
        
        fig.update_yaxes(title_text="Label Balance (P(y=1))", row=1, col=1)
        fig.update_yaxes(title_text="Return Std Dev", row=1, col=2)
    else:
        # Single plot: label balance only
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=split_names,
                y=label_balance,
                marker_color="#58A6FF",
                name="Label Balance",
            )
        )
        fig.add_hline(
            y=0.5,
            line_dash="dash", line_color="#8B949E",
        )
        fig.update_yaxes(title_text="Label Balance (P(y=1))")
    
    # Layout
    fig.update_layout(
        title_text=title,
        title_font_color="#E6EDF3",
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        font_color="#E6EDF3",
        height=400,
        width=1200 if returns is not None else 800,
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#21262D")
    fig.update_yaxes(gridcolor="#21262D")
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED VISUALIZATION UTILITIES (AFML Ch. 7 - Cross-Validation Analysis)
# ═══════════════════════════════════════════════════════════════════════════════
# Extracted from: BTCUSDT EDA Parameter Optimization, Phases 6C & 7C
# Author: Databricks AI Assistant
# Date: 2026-05-01

def plot_cv_comparison(
    wf_results: dict[str, list[dict]],
    purged_aucs: np.ndarray | None = None,
    unpurged_aucs: np.ndarray | None = None,
    mdi_rank: np.ndarray | None = None,
    mi_rank_lookup: dict | None = None,
    all_feature_names: list[str] | None = None,
    n_splits: int = 5,
    title: str = "Cross-Validation Comparison",
    colors: dict[str, str] | None = None,
) -> go.Figure:
    """Create 2×2 dashboard comparing walk-forward CV with K-fold CV results.
    
    Panels:
      (1,1) Walk-forward AUC by split (all feature sets)
      (1,2) Feature importance: MDI rank vs MI rank scatter (top 20)
      (2,1) Purged vs Unpurged K-fold AUC comparison
      (2,2) Walk-forward AUC trend with linear fits
    
    Args:
        wf_results: Walk-forward results dict, format:
            {"Feature Set Name": [{"split": 0, "auc": 0.52, ...}, ...]}
        purged_aucs: K-fold AUCs with purging, shape (n_splits,) (optional)
        unpurged_aucs: K-fold AUCs without purging, shape (n_splits,) (optional)
        mdi_rank: Feature indices sorted by MDI importance, shape (n_features,) (optional)
        mi_rank_lookup: Dict mapping feature idx to MI rank (optional)
        all_feature_names: Feature names list (optional)
        n_splits: Number of K-fold splits (default: 5)
        title: Dashboard title (default: "Cross-Validation Comparison")
        colors: Color mapping for feature sets (optional)
    
    Returns:
        Plotly Figure object (2×2 subplots)
    
    Example:
        >>> wf_results = {
        ...     "Full": [{"split": 0, "auc": 0.50}, {"split": 1, "auc": 0.52}],
        ...     "Top-10": [{"split": 0, "auc": 0.51}, {"split": 1, "auc": 0.53}],
        ... }
        >>> fig = plot_cv_comparison(wf_results)
        >>> fig.show()
    """
    if colors is None:
        colors = {"Full": "#58A6FF", "Top-10": "#52C41A", "Top-21": "#F0883E"}
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Walk-Forward AUC by Split",
            "Feature Importance: MDI vs MI Rank (Top 20)",
            "Purged vs Unpurged K-Fold AUC",
            "Walk-Forward AUC Trend (Signal Stability)",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )
    
    # ── Panel (1,1): Walk-forward AUC by split ──────────────────
    for fs_name, fs_res in wf_results.items():
        aucs = [r["auc"] for r in fs_res]
        color = colors.get(fs_name, "#58A6FF")
        fig.add_trace(
            go.Scatter(
                x=[f"Split {r['split']}" for r in fs_res],
                y=aucs,
                mode="lines+markers",
                name=fs_name,
                line=dict(color=color, width=2),
                marker=dict(size=8),
                showlegend=True,
            ),
            row=1, col=1,
        )
    
    # Random baseline
    fig.add_hline(
        y=0.5, row=1, col=1,
        line_dash="dash", line_color="#8B949E", line_width=1.5,
        annotation_text="Random (0.50)",
        annotation_font_color="#8B949E",
    )
    
    # ── Panel (1,2): MDI vs MI rank scatter ─────────────────────
    if mdi_rank is not None and mi_rank_lookup is not None and all_feature_names is not None:
        top20_mdi = mdi_rank[:20]
        mdi_ranks_plot = np.arange(1, 21)
        mi_ranks_plot = [mi_rank_lookup.get(idx, len(mdi_rank) - 1) + 1 for idx in top20_mdi]
        feat_names_plot = [all_feature_names[idx] for idx in top20_mdi]
        
        fig.add_trace(
            go.Scatter(
                x=mdi_ranks_plot,
                y=mi_ranks_plot,
                mode="markers+text",
                text=[n[:12] for n in feat_names_plot],
                textposition="top center",
                textfont=dict(size=8, color="#E6EDF3"),
                marker=dict(
                    size=10,
                    color=["#52C41A" if mi <= 21 else "#FF4D4F" for mi in mi_ranks_plot],
                    opacity=0.8,
                ),
                showlegend=False,
            ),
            row=1, col=2,
        )
        # Diagonal reference
        fig.add_trace(
            go.Scatter(
                x=[1, 20], y=[1, 20],
                mode="lines",
                line=dict(color="#8B949E", dash="dash", width=1),
                showlegend=False,
            ),
            row=1, col=2,
        )
    
    # ── Panel (2,1): Purged vs Unpurged K-fold ──────────────────
    if purged_aucs is not None and unpurged_aucs is not None:
        fold_labels = [f"Fold {k}" for k in range(n_splits)]
        
        fig.add_trace(
            go.Bar(
                x=fold_labels, y=purged_aucs,
                name="Purged",
                marker_color="#58A6FF",
                showlegend=True,
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=fold_labels, y=unpurged_aucs,
                name="Unpurged",
                marker_color="#F0883E",
                opacity=0.7,
                showlegend=True,
            ),
            row=2, col=1,
        )
        fig.add_hline(
            y=0.5, row=2, col=1,
            line_dash="dash", line_color="#8B949E", line_width=1.5,
        )
    
    # ── Panel (2,2): AUC trend with linear fits ─────────────────
    for fs_name, fs_res in wf_results.items():
        aucs = [r["auc"] for r in fs_res]
        splits = list(range(len(aucs)))
        color = colors.get(fs_name, "#58A6FF")
        
        # Linear fit
        coefs = np.polyfit(splits, aucs, 1)
        trend_line = np.polyval(coefs, splits)
        
        fig.add_trace(
            go.Scatter(
                x=[f"WF {s}" for s in splits],
                y=aucs,
                mode="markers",
                marker=dict(size=10, color=color),
                showlegend=False,
            ),
            row=2, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[f"WF {s}" for s in splits],
                y=trend_line.tolist(),
                mode="lines",
                line=dict(color=color, dash="dash", width=1.5),
                name=f"{fs_name} trend",
                showlegend=False,
            ),
            row=2, col=2,
        )
    
    fig.add_hline(
        y=0.5, row=2, col=2,
        line_dash="dash", line_color="#8B949E", line_width=1.5,
    )
    
    # ── Layout ──────────────────────────────────────────────────
    fig.update_layout(
        title_text=title,
        title_font_color="#E6EDF3",
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        font_color="#E6EDF3",
        height=800,
        width=1200,
        barmode="group",
        legend=dict(
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor="#21262D",
            font_color="#E6EDF3",
        ),
    )
    fig.update_xaxes(gridcolor="#21262D")
    fig.update_yaxes(gridcolor="#21262D")
    
    # Axis labels
    fig.update_xaxes(title_text="Split", row=1, col=1)
    fig.update_yaxes(title_text="AUC", row=1, col=1)
    fig.update_xaxes(title_text="MDI Rank", row=1, col=2)
    fig.update_yaxes(title_text="MI Rank", row=1, col=2)
    fig.update_xaxes(title_text="Fold", row=2, col=1)
    fig.update_yaxes(title_text="AUC", row=2, col=1)
    fig.update_xaxes(title_text="Walk-Forward Split", row=2, col=2)
    fig.update_yaxes(title_text="AUC", row=2, col=2)
    
    return fig


def plot_meta_labeling_dashboard(
    meta_prob: np.ndarray,
    meta_labels: np.ndarray,
    trade_returns: np.ndarray,
    bet_sweep: list[dict],
    persist_acc: float,
    auc_progression: dict[str, float] | None = None,
    best_threshold: float | None = None,
    title: str = "Meta-Labeling Analysis",
) -> go.Figure:
    """Create 2×2 dashboard for meta-labeling model analysis.
    
    Panels:
      (1,1) Equity curve: always trade vs meta-filtered
      (1,2) Win rate vs trade fraction (bet sizing sweep)
      (2,1) Meta-probability distribution by correctness
      (2,2) AUC progression across phases
    
    Args:
        meta_prob: Meta-label probabilities (P(primary correct)), shape (N,)
        meta_labels: Actual meta-labels (1=correct, 0=wrong), shape (N,)
        trade_returns: Trade returns (+1 if correct, -1 if wrong), shape (N,)
        bet_sweep: Bet sizing sweep results, list of dicts with keys:
            {"threshold", "frac", "win_rate", "sharpe", ...}
        persist_acc: Persistence model accuracy (for reference line)
        auc_progression: Dict mapping phase names to AUC values (optional)
        best_threshold: Optimal threshold from bet sweep (for equity curve)
        title: Dashboard title
    
    Returns:
        Plotly Figure object (2×2 subplots)
    
    Example:
        >>> meta_prob = model.predict_proba(X_test)[:, 1]
        >>> meta_labels = (y_pred == y_test).astype(int)
        >>> trade_returns = np.where(y_pred == y_test, 1.0, -1.0)
        >>> bet_sweep = [
        ...     {"threshold": 0.5, "frac": 1.0, "win_rate": 0.87, "sharpe": 1.2},
        ...     {"threshold": 0.7, "frac": 0.8, "win_rate": 0.90, "sharpe": 1.5},
        ... ]
        >>> fig = plot_meta_labeling_dashboard(
        ...     meta_prob, meta_labels, trade_returns, bet_sweep, persist_acc=0.87
        ... )
        >>> fig.show()
    """
    if best_threshold is None:
        best_threshold = max(bet_sweep, key=lambda b: b["sharpe"])["threshold"]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Equity Curve: Always Trade vs Meta-Filtered",
            "Win Rate vs Trade Fraction",
            "Meta-Label Probability Distribution",
            "AUC Progression (Phases)",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )
    
    # ── Panel (1,1): Equity curves ──────────────────────────────
    mask_filtered = meta_prob >= best_threshold
    eq_always = np.cumsum(trade_returns)
    eq_filtered = np.cumsum(np.where(mask_filtered, trade_returns, 0.0))
    
    bar_idx = np.arange(len(meta_prob))
    fig.add_trace(
        go.Scatter(
            x=bar_idx, y=eq_always,
            mode="lines", name="Always Trade",
            line=dict(color="#8B949E", width=1.5),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=bar_idx, y=eq_filtered,
            mode="lines", name=f"Meta-Filtered (t>={best_threshold:.2f})",
            line=dict(color="#52C41A", width=2),
        ),
        row=1, col=1,
    )
    
    # ── Panel (1,2): Win rate vs trade fraction ─────────────────
    fig.add_trace(
        go.Scatter(
            x=[b["frac"] for b in bet_sweep],
            y=[b["win_rate"] for b in bet_sweep],
            mode="lines+markers",
            marker=dict(size=6, color="#58A6FF"),
            line=dict(color="#58A6FF", width=2),
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_hline(
        y=persist_acc, row=1, col=2,
        line_dash="dash", line_color="#F0883E",
        annotation_text=f"Persist {persist_acc:.1%}",
        annotation_font_color="#F0883E",
    )
    
    # ── Panel (2,1): Probability distribution ───────────────────
    fig.add_trace(
        go.Histogram(
            x=meta_prob[meta_labels == 1],
            name="Correct (meta=1)",
            marker_color="#52C41A",
            opacity=0.7,
            nbinsx=30,
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=meta_prob[meta_labels == 0],
            name="Wrong (meta=0)",
            marker_color="#FF4D4F",
            opacity=0.7,
            nbinsx=30,
        ),
        row=2, col=1,
    )
    
    # ── Panel (2,2): AUC progression ────────────────────────────
    if auc_progression is not None:
        fig.add_trace(
            go.Bar(
                x=list(auc_progression.keys()),
                y=list(auc_progression.values()),
                marker_color=["#8B949E", "#8B949E", "#F0883E", "#58A6FF", "#52C41A"][:len(auc_progression)],
                showlegend=False,
            ),
            row=2, col=2,
        )
        fig.add_hline(
            y=0.5, row=2, col=2,
            line_dash="dash", line_color="#8B949E",
        )
    
    # ── Layout ──────────────────────────────────────────────────
    fig.update_layout(
        title_text=title,
        title_font_color="#E6EDF3",
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        font_color="#E6EDF3",
        height=800,
        width=1200,
        barmode="overlay",
        legend=dict(
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor="#21262D",
        ),
    )
    fig.update_xaxes(gridcolor="#21262D")
    fig.update_yaxes(gridcolor="#21262D")
    
    # Axis labels
    fig.update_xaxes(title_text="Bar Index", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_xaxes(title_text="Trade Fraction", row=1, col=2)
    fig.update_yaxes(title_text="Win Rate", row=1, col=2)
    fig.update_xaxes(title_text="Meta-Probability", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Phase", row=2, col=2)
    fig.update_yaxes(title_text="AUC", row=2, col=2)
    
    return fig


def plot_regime_analysis(
    split_indices: list[tuple[np.ndarray, np.ndarray, int, int]],
    y_labels: np.ndarray,
    returns: np.ndarray | None = None,
    split_names: list[str] | None = None,
    title: str = "Regime Analysis by Split",
) -> go.Figure:
    """Analyze regime characteristics across CV splits.
    
    Plots label balance and (optionally) volatility for each split to identify
    regime shifts that affect model performance.
    
    Args:
        split_indices: List of (train_idx, test_idx, n_purged, n_embargo) tuples
        y_labels: Label array, shape (N,)
        returns: Return array for volatility analysis, shape (N,) (optional)
        split_names: Custom split names (default: "Split 0", "Split 1", ...)
        title: Plot title
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> split_indices = [
        ...     (train_0, test_0, 10, 20),
        ...     (train_1, test_1, 12, 18),
        ... ]
        >>> fig = plot_regime_analysis(split_indices, y_labels, returns)
        >>> fig.show()
    """
    n_splits = len(split_indices)
    if split_names is None:
        split_names = [f"Split {i}" for i in range(n_splits)]
    
    # Compute label balance and volatility per split
    label_balance = []
    volatilities = []
    
    for _, test_idx, _, _ in split_indices:
        y_test = y_labels[test_idx]
        balance = float(y_test.mean())
        label_balance.append(balance)
        
        if returns is not None:
            ret_test = returns[test_idx]
            vol = float(ret_test.std())
            volatilities.append(vol)
    
    # Create figure
    if returns is not None:
        # Two subplots: label balance + volatility
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Label Balance by Split", "Volatility by Split"],
            horizontal_spacing=0.15,
        )
        
        # Label balance
        fig.add_trace(
            go.Bar(
                x=split_names,
                y=label_balance,
                marker_color="#58A6FF",
                name="Label Balance",
            ),
            row=1, col=1,
        )
        fig.add_hline(
            y=0.5, row=1, col=1,
            line_dash="dash", line_color="#8B949E",
        )
        
        # Volatility
        fig.add_trace(
            go.Bar(
                x=split_names,
                y=volatilities,
                marker_color="#F0883E",
                name="Volatility",
            ),
            row=1, col=2,
        )
        
        fig.update_yaxes(title_text="Label Balance (P(y=1))", row=1, col=1)
        fig.update_yaxes(title_text="Return Std Dev", row=1, col=2)
    else:
        # Single plot: label balance only
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=split_names,
                y=label_balance,
                marker_color="#58A6FF",
                name="Label Balance",
            )
        )
        fig.add_hline(
            y=0.5,
            line_dash="dash", line_color="#8B949E",
        )
        fig.update_yaxes(title_text="Label Balance (P(y=1))")
    
    # Layout
    fig.update_layout(
        title_text=title,
        title_font_color="#E6EDF3",
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        font_color="#E6EDF3",
        height=400,
        width=1200 if returns is not None else 800,
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#21262D")
    fig.update_yaxes(gridcolor="#21262D")
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED VISUALIZATION UTILITIES (AFML Ch. 7 - Cross-Validation Analysis)
# ═══════════════════════════════════════════════════════════════════════════════
# Extracted from: BTCUSDT EDA Parameter Optimization, Phases 6C & 7C
# Author: Databricks AI Assistant
# Date: 2026-05-01

def plot_cv_comparison(
    wf_results: dict[str, list[dict]],
    purged_aucs: np.ndarray | None = None,
    unpurged_aucs: np.ndarray | None = None,
    mdi_rank: np.ndarray | None = None,
    mi_rank_lookup: dict | None = None,
    all_feature_names: list[str] | None = None,
    n_splits: int = 5,
    title: str = "Cross-Validation Comparison",
    colors: dict[str, str] | None = None,
) -> go.Figure:
    """Create 2×2 dashboard comparing walk-forward CV with K-fold CV results.
    
    Panels:
      (1,1) Walk-forward AUC by split (all feature sets)
      (1,2) Feature importance: MDI rank vs MI rank scatter (top 20)
      (2,1) Purged vs Unpurged K-fold AUC comparison
      (2,2) Walk-forward AUC trend with linear fits
    
    Args:
        wf_results: Walk-forward results dict, format:
            {"Feature Set Name": [{"split": 0, "auc": 0.52, ...}, ...]}
        purged_aucs: K-fold AUCs with purging, shape (n_splits,) (optional)
        unpurged_aucs: K-fold AUCs without purging, shape (n_splits,) (optional)
        mdi_rank: Feature indices sorted by MDI importance, shape (n_features,) (optional)
        mi_rank_lookup: Dict mapping feature idx to MI rank (optional)
        all_feature_names: Feature names list (optional)
        n_splits: Number of K-fold splits (default: 5)
        title: Dashboard title (default: "Cross-Validation Comparison")
        colors: Color mapping for feature sets (optional)
    
    Returns:
        Plotly Figure object (2×2 subplots)
    
    Example:
        >>> wf_results = {
        ...     "Full": [{"split": 0, "auc": 0.50}, {"split": 1, "auc": 0.52}],
        ...     "Top-10": [{"split": 0, "auc": 0.51}, {"split": 1, "auc": 0.53}],
        ... }
        >>> fig = plot_cv_comparison(wf_results)
        >>> fig.show()
    """
    if colors is None:
        colors = {"Full": "#58A6FF", "Top-10": "#52C41A", "Top-21": "#F0883E"}
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Walk-Forward AUC by Split",
            "Feature Importance: MDI vs MI Rank (Top 20)",
            "Purged vs Unpurged K-Fold AUC",
            "Walk-Forward AUC Trend (Signal Stability)",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )
    
    # ── Panel (1,1): Walk-forward AUC by split ──────────────────
    for fs_name, fs_res in wf_results.items():
        aucs = [r["auc"] for r in fs_res]
        color = colors.get(fs_name, "#58A6FF")
        fig.add_trace(
            go.Scatter(
                x=[f"Split {r['split']}" for r in fs_res],
                y=aucs,
                mode="lines+markers",
                name=fs_name,
                line=dict(color=color, width=2),
                marker=dict(size=8),
                showlegend=True,
            ),
            row=1, col=1,
        )
    
    # Random baseline
    fig.add_hline(
        y=0.5, row=1, col=1,
        line_dash="dash", line_color="#8B949E", line_width=1.5,
        annotation_text="Random (0.50)",
        annotation_font_color="#8B949E",
    )
    
    # ── Panel (1,2): MDI vs MI rank scatter ─────────────────────
    if mdi_rank is not None and mi_rank_lookup is not None and all_feature_names is not None:
        top20_mdi = mdi_rank[:20]
        mdi_ranks_plot = np.arange(1, 21)
        mi_ranks_plot = [mi_rank_lookup.get(idx, len(mdi_rank) - 1) + 1 for idx in top20_mdi]
        feat_names_plot = [all_feature_names[idx] for idx in top20_mdi]
        
        fig.add_trace(
            go.Scatter(
                x=mdi_ranks_plot,
                y=mi_ranks_plot,
                mode="markers+text",
                text=[n[:12] for n in feat_names_plot],
                textposition="top center",
                textfont=dict(size=8, color="#E6EDF3"),
                marker=dict(
                    size=10,
                    color=["#52C41A" if mi <= 21 else "#FF4D4F" for mi in mi_ranks_plot],
                    opacity=0.8,
                ),
                showlegend=False,
            ),
            row=1, col=2,
        )
        # Diagonal reference
        fig.add_trace(
            go.Scatter(
                x=[1, 20], y=[1, 20],
                mode="lines",
                line=dict(color="#8B949E", dash="dash", width=1),
                showlegend=False,
            ),
            row=1, col=2,
        )
    
    # ── Panel (2,1): Purged vs Unpurged K-fold ──────────────────
    if purged_aucs is not None and unpurged_aucs is not None:
        fold_labels = [f"Fold {k}" for k in range(n_splits)]
        
        fig.add_trace(
            go.Bar(
                x=fold_labels, y=purged_aucs,
                name="Purged",
                marker_color="#58A6FF",
                showlegend=True,
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=fold_labels, y=unpurged_aucs,
                name="Unpurged",
                marker_color="#F0883E",
                opacity=0.7,
                showlegend=True,
            ),
            row=2, col=1,
        )
        fig.add_hline(
            y=0.5, row=2, col=1,
            line_dash="dash", line_color="#8B949E", line_width=1.5,
        )
    
    # ── Panel (2,2): AUC trend with linear fits ─────────────────
    for fs_name, fs_res in wf_results.items():
        aucs = [r["auc"] for r in fs_res]
        splits = list(range(len(aucs)))
        color = colors.get(fs_name, "#58A6FF")
        
        # Linear fit
        coefs = np.polyfit(splits, aucs, 1)
        trend_line = np.polyval(coefs, splits)
        
        fig.add_trace(
            go.Scatter(
                x=[f"WF {s}" for s in splits],
                y=aucs,
                mode="markers",
                marker=dict(size=10, color=color),
                showlegend=False,
            ),
            row=2, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[f"WF {s}" for s in splits],
                y=trend_line.tolist(),
                mode="lines",
                line=dict(color=color, dash="dash", width=1.5),
                name=f"{fs_name} trend",
                showlegend=False,
            ),
            row=2, col=2,
        )
    
    fig.add_hline(
        y=0.5, row=2, col=2,
        line_dash="dash", line_color="#8B949E", line_width=1.5,
    )
    
    # ── Layout ──────────────────────────────────────────────────
    fig.update_layout(
        title_text=title,
        title_font_color="#E6EDF3",
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        font_color="#E6EDF3",
        height=800,
        width=1200,
        barmode="group",
        legend=dict(
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor="#21262D",
            font_color="#E6EDF3",
        ),
    )
    fig.update_xaxes(gridcolor="#21262D")
    fig.update_yaxes(gridcolor="#21262D")
    
    # Axis labels
    fig.update_xaxes(title_text="Split", row=1, col=1)
    fig.update_yaxes(title_text="AUC", row=1, col=1)
    fig.update_xaxes(title_text="MDI Rank", row=1, col=2)
    fig.update_yaxes(title_text="MI Rank", row=1, col=2)
    fig.update_xaxes(title_text="Fold", row=2, col=1)
    fig.update_yaxes(title_text="AUC", row=2, col=1)
    fig.update_xaxes(title_text="Walk-Forward Split", row=2, col=2)
    fig.update_yaxes(title_text="AUC", row=2, col=2)
    
    return fig


def plot_meta_labeling_dashboard(
    meta_prob: np.ndarray,
    meta_labels: np.ndarray,
    trade_returns: np.ndarray,
    bet_sweep: list[dict],
    persist_acc: float,
    auc_progression: dict[str, float] | None = None,
    best_threshold: float | None = None,
    title: str = "Meta-Labeling Analysis",
) -> go.Figure:
    """Create 2×2 dashboard for meta-labeling model analysis.
    
    Panels:
      (1,1) Equity curve: always trade vs meta-filtered
      (1,2) Win rate vs trade fraction (bet sizing sweep)
      (2,1) Meta-probability distribution by correctness
      (2,2) AUC progression across phases
    
    Args:
        meta_prob: Meta-label probabilities (P(primary correct)), shape (N,)
        meta_labels: Actual meta-labels (1=correct, 0=wrong), shape (N,)
        trade_returns: Trade returns (+1 if correct, -1 if wrong), shape (N,)
        bet_sweep: Bet sizing sweep results, list of dicts with keys:
            {"threshold", "frac", "win_rate", "sharpe", ...}
        persist_acc: Persistence model accuracy (for reference line)
        auc_progression: Dict mapping phase names to AUC values (optional)
        best_threshold: Optimal threshold from bet sweep (for equity curve)
        title: Dashboard title
    
    Returns:
        Plotly Figure object (2×2 subplots)
    
    Example:
        >>> meta_prob = model.predict_proba(X_test)[:, 1]
        >>> meta_labels = (y_pred == y_test).astype(int)
        >>> trade_returns = np.where(y_pred == y_test, 1.0, -1.0)
        >>> bet_sweep = [
        ...     {"threshold": 0.5, "frac": 1.0, "win_rate": 0.87, "sharpe": 1.2},
        ...     {"threshold": 0.7, "frac": 0.8, "win_rate": 0.90, "sharpe": 1.5},
        ... ]
        >>> fig = plot_meta_labeling_dashboard(
        ...     meta_prob, meta_labels, trade_returns, bet_sweep, persist_acc=0.87
        ... )
        >>> fig.show()
    """
    if best_threshold is None:
        best_threshold = max(bet_sweep, key=lambda b: b["sharpe"])["threshold"]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Equity Curve: Always Trade vs Meta-Filtered",
            "Win Rate vs Trade Fraction",
            "Meta-Label Probability Distribution",
            "AUC Progression (Phases)",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )
    
    # ── Panel (1,1): Equity curves ──────────────────────────────
    mask_filtered = meta_prob >= best_threshold
    eq_always = np.cumsum(trade_returns)
    eq_filtered = np.cumsum(np.where(mask_filtered, trade_returns, 0.0))
    
    bar_idx = np.arange(len(meta_prob))
    fig.add_trace(
        go.Scatter(
            x=bar_idx, y=eq_always,
            mode="lines", name="Always Trade",
            line=dict(color="#8B949E", width=1.5),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=bar_idx, y=eq_filtered,
            mode="lines", name=f"Meta-Filtered (t>={best_threshold:.2f})",
            line=dict(color="#52C41A", width=2),
        ),
        row=1, col=1,
    )
    
    # ── Panel (1,2): Win rate vs trade fraction ─────────────────
    fig.add_trace(
        go.Scatter(
            x=[b["frac"] for b in bet_sweep],
            y=[b["win_rate"] for b in bet_sweep],
            mode="lines+markers",
            marker=dict(size=6, color="#58A6FF"),
            line=dict(color="#58A6FF", width=2),
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_hline(
        y=persist_acc, row=1, col=2,
        line_dash="dash", line_color="#F0883E",
        annotation_text=f"Persist {persist_acc:.1%}",
        annotation_font_color="#F0883E",
    )
    
    # ── Panel (2,1): Probability distribution ───────────────────
    fig.add_trace(
        go.Histogram(
            x=meta_prob[meta_labels == 1],
            name="Correct (meta=1)",
            marker_color="#52C41A",
            opacity=0.7,
            nbinsx=30,
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=meta_prob[meta_labels == 0],
            name="Wrong (meta=0)",
            marker_color="#FF4D4F",
            opacity=0.7,
            nbinsx=30,
        ),
        row=2, col=1,
    )
    
    # ── Panel (2,2): AUC progression ────────────────────────────
    if auc_progression is not None:
        fig.add_trace(
            go.Bar(
                x=list(auc_progression.keys()),
                y=list(auc_progression.values()),
                marker_color=["#8B949E", "#8B949E", "#F0883E", "#58A6FF", "#52C41A"][:len(auc_progression)],
                showlegend=False,
            ),
            row=2, col=2,
        )
        fig.add_hline(
            y=0.5, row=2, col=2,
            line_dash="dash", line_color="#8B949E",
        )
    
    # ── Layout ──────────────────────────────────────────────────
    fig.update_layout(
        title_text=title,
        title_font_color="#E6EDF3",
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        font_color="#E6EDF3",
        height=800,
        width=1200,
        barmode="overlay",
        legend=dict(
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor="#21262D",
        ),
    )
    fig.update_xaxes(gridcolor="#21262D")
    fig.update_yaxes(gridcolor="#21262D")
    
    # Axis labels
    fig.update_xaxes(title_text="Bar Index", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_xaxes(title_text="Trade Fraction", row=1, col=2)
    fig.update_yaxes(title_text="Win Rate", row=1, col=2)
    fig.update_xaxes(title_text="Meta-Probability", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Phase", row=2, col=2)
    fig.update_yaxes(title_text="AUC", row=2, col=2)
    
    return fig


def plot_regime_analysis(
    split_indices: list[tuple[np.ndarray, np.ndarray, int, int]],
    y_labels: np.ndarray,
    returns: np.ndarray | None = None,
    split_names: list[str] | None = None,
    title: str = "Regime Analysis by Split",
) -> go.Figure:
    """Analyze regime characteristics across CV splits.
    
    Plots label balance and (optionally) volatility for each split to identify
    regime shifts that affect model performance.
    
    Args:
        split_indices: List of (train_idx, test_idx, n_purged, n_embargo) tuples
        y_labels: Label array, shape (N,)
        returns: Return array for volatility analysis, shape (N,) (optional)
        split_names: Custom split names (default: "Split 0", "Split 1", ...)
        title: Plot title
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> split_indices = [
        ...     (train_0, test_0, 10, 20),
        ...     (train_1, test_1, 12, 18),
        ... ]
        >>> fig = plot_regime_analysis(split_indices, y_labels, returns)
        >>> fig.show()
    """
    n_splits = len(split_indices)
    if split_names is None:
        split_names = [f"Split {i}" for i in range(n_splits)]
    
    # Compute label balance and volatility per split
    label_balance = []
    volatilities = []
    
    for _, test_idx, _, _ in split_indices:
        y_test = y_labels[test_idx]
        balance = float(y_test.mean())
        label_balance.append(balance)
        
        if returns is not None:
            ret_test = returns[test_idx]
            vol = float(ret_test.std())
            volatilities.append(vol)
    
    # Create figure
    if returns is not None:
        # Two subplots: label balance + volatility
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Label Balance by Split", "Volatility by Split"],
            horizontal_spacing=0.15,
        )
        
        # Label balance
        fig.add_trace(
            go.Bar(
                x=split_names,
                y=label_balance,
                marker_color="#58A6FF",
                name="Label Balance",
            ),
            row=1, col=1,
        )
        fig.add_hline(
            y=0.5, row=1, col=1,
            line_dash="dash", line_color="#8B949E",
        )
        
        # Volatility
        fig.add_trace(
            go.Bar(
                x=split_names,
                y=volatilities,
                marker_color="#F0883E",
                name="Volatility",
            ),
            row=1, col=2,
        )
        
        fig.update_yaxes(title_text="Label Balance (P(y=1))", row=1, col=1)
        fig.update_yaxes(title_text="Return Std Dev", row=1, col=2)
    else:
        # Single plot: label balance only
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=split_names,
                y=label_balance,
                marker_color="#58A6FF",
                name="Label Balance",
            )
        )
        fig.add_hline(
            y=0.5,
            line_dash="dash", line_color="#8B949E",
        )
        fig.update_yaxes(title_text="Label Balance (P(y=1))")
    
    # Layout
    fig.update_layout(
        title_text=title,
        title_font_color="#E6EDF3",
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        font_color="#E6EDF3",
        height=400,
        width=1200 if returns is not None else 800,
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#21262D")
    fig.update_yaxes(gridcolor="#21262D")
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED VISUALIZATION UTILITIES (AFML Ch. 7 - Cross-Validation Analysis)
# ═══════════════════════════════════════════════════════════════════════════════
# Extracted from: BTCUSDT EDA Parameter Optimization, Phases 6C & 7C
# Author: Databricks AI Assistant
# Date: 2026-05-01

def plot_cv_comparison(
    wf_results: dict[str, list[dict]],
    purged_aucs: np.ndarray | None = None,
    unpurged_aucs: np.ndarray | None = None,
    mdi_rank: np.ndarray | None = None,
    mi_rank_lookup: dict | None = None,
    all_feature_names: list[str] | None = None,
    n_splits: int = 5,
    title: str = "Cross-Validation Comparison",
    colors: dict[str, str] | None = None,
) -> go.Figure:
    """Create 2×2 dashboard comparing walk-forward CV with K-fold CV results.
    
    Panels:
      (1,1) Walk-forward AUC by split (all feature sets)
      (1,2) Feature importance: MDI rank vs MI rank scatter (top 20)
      (2,1) Purged vs Unpurged K-fold AUC comparison
      (2,2) Walk-forward AUC trend with linear fits
    
    Args:
        wf_results: Walk-forward results dict, format:
            {"Feature Set Name": [{"split": 0, "auc": 0.52, ...}, ...]}
        purged_aucs: K-fold AUCs with purging, shape (n_splits,) (optional)
        unpurged_aucs: K-fold AUCs without purging, shape (n_splits,) (optional)
        mdi_rank: Feature indices sorted by MDI importance, shape (n_features,) (optional)
        mi_rank_lookup: Dict mapping feature idx to MI rank (optional)
        all_feature_names: Feature names list (optional)
        n_splits: Number of K-fold splits (default: 5)
        title: Dashboard title (default: "Cross-Validation Comparison")
        colors: Color mapping for feature sets (optional)
    
    Returns:
        Plotly Figure object (2×2 subplots)
    
    Example:
        >>> wf_results = {
        ...     "Full": [{"split": 0, "auc": 0.50}, {"split": 1, "auc": 0.52}],
        ...     "Top-10": [{"split": 0, "auc": 0.51}, {"split": 1, "auc": 0.53}],
        ... }
        >>> fig = plot_cv_comparison(wf_results)
        >>> fig.show()
    """
    if colors is None:
        colors = {"Full": "#58A6FF", "Top-10": "#52C41A", "Top-21": "#F0883E"}
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Walk-Forward AUC by Split",
            "Feature Importance: MDI vs MI Rank (Top 20)",
            "Purged vs Unpurged K-Fold AUC",
            "Walk-Forward AUC Trend (Signal Stability)",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )
    
    # ── Panel (1,1): Walk-forward AUC by split ──────────────────
    for fs_name, fs_res in wf_results.items():
        aucs = [r["auc"] for r in fs_res]
        color = colors.get(fs_name, "#58A6FF")
        fig.add_trace(
            go.Scatter(
                x=[f"Split {r['split']}" for r in fs_res],
                y=aucs,
                mode="lines+markers",
                name=fs_name,
                line=dict(color=color, width=2),
                marker=dict(size=8),
                showlegend=True,
            ),
            row=1, col=1,
        )
    
    # Random baseline
    fig.add_hline(
        y=0.5, row=1, col=1,
        line_dash="dash", line_color="#8B949E", line_width=1.5,
        annotation_text="Random (0.50)",
        annotation_font_color="#8B949E",
    )
    
    # ── Panel (1,2): MDI vs MI rank scatter ─────────────────────
    if mdi_rank is not None and mi_rank_lookup is not None and all_feature_names is not None:
        top20_mdi = mdi_rank[:20]
        mdi_ranks_plot = np.arange(1, 21)
        mi_ranks_plot = [mi_rank_lookup.get(idx, len(mdi_rank) - 1) + 1 for idx in top20_mdi]
        feat_names_plot = [all_feature_names[idx] for idx in top20_mdi]
        
        fig.add_trace(
            go.Scatter(
                x=mdi_ranks_plot,
                y=mi_ranks_plot,
                mode="markers+text",
                text=[n[:12] for n in feat_names_plot],
                textposition="top center",
                textfont=dict(size=8, color="#E6EDF3"),
                marker=dict(
                    size=10,
                    color=["#52C41A" if mi <= 21 else "#FF4D4F" for mi in mi_ranks_plot],
                    opacity=0.8,
                ),
                showlegend=False,
            ),
            row=1, col=2,
        )
        # Diagonal reference
        fig.add_trace(
            go.Scatter(
                x=[1, 20], y=[1, 20],
                mode="lines",
                line=dict(color="#8B949E", dash="dash", width=1),
                showlegend=False,
            ),
            row=1, col=2,
        )
    
    # ── Panel (2,1): Purged vs Unpurged K-fold ──────────────────
    if purged_aucs is not None and unpurged_aucs is not None:
        fold_labels = [f"Fold {k}" for k in range(n_splits)]
        
        fig.add_trace(
            go.Bar(
                x=fold_labels, y=purged_aucs,
                name="Purged",
                marker_color="#58A6FF",
                showlegend=True,
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=fold_labels, y=unpurged_aucs,
                name="Unpurged",
                marker_color="#F0883E",
                opacity=0.7,
                showlegend=True,
            ),
            row=2, col=1,
        )
        fig.add_hline(
            y=0.5, row=2, col=1,
            line_dash="dash", line_color="#8B949E", line_width=1.5,
        )
    
    # ── Panel (2,2): AUC trend with linear fits ─────────────────
    for fs_name, fs_res in wf_results.items():
        aucs = [r["auc"] for r in fs_res]
        splits = list(range(len(aucs)))
        color = colors.get(fs_name, "#58A6FF")
        
        # Linear fit
        coefs = np.polyfit(splits, aucs, 1)
        trend_line = np.polyval(coefs, splits)
        
        fig.add_trace(
            go.Scatter(
                x=[f"WF {s}" for s in splits],
                y=aucs,
                mode="markers",
                marker=dict(size=10, color=color),
                showlegend=False,
            ),
            row=2, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[f"WF {s}" for s in splits],
                y=trend_line.tolist(),
                mode="lines",
                line=dict(color=color, dash="dash", width=1.5),
                name=f"{fs_name} trend",
                showlegend=False,
            ),
            row=2, col=2,
        )
    
    fig.add_hline(
        y=0.5, row=2, col=2,
        line_dash="dash", line_color="#8B949E", line_width=1.5,
    )
    
    # ── Layout ──────────────────────────────────────────────────
    fig.update_layout(
        title_text=title,
        title_font_color="#E6EDF3",
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        font_color="#E6EDF3",
        height=800,
        width=1200,
        barmode="group",
        legend=dict(
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor="#21262D",
            font_color="#E6EDF3",
        ),
    )
    fig.update_xaxes(gridcolor="#21262D")
    fig.update_yaxes(gridcolor="#21262D")
    
    # Axis labels
    fig.update_xaxes(title_text="Split", row=1, col=1)
    fig.update_yaxes(title_text="AUC", row=1, col=1)
    fig.update_xaxes(title_text="MDI Rank", row=1, col=2)
    fig.update_yaxes(title_text="MI Rank", row=1, col=2)
    fig.update_xaxes(title_text="Fold", row=2, col=1)
    fig.update_yaxes(title_text="AUC", row=2, col=1)
    fig.update_xaxes(title_text="Walk-Forward Split", row=2, col=2)
    fig.update_yaxes(title_text="AUC", row=2, col=2)
    
    return fig


def plot_meta_labeling_dashboard(
    meta_prob: np.ndarray,
    meta_labels: np.ndarray,
    trade_returns: np.ndarray,
    bet_sweep: list[dict],
    persist_acc: float,
    auc_progression: dict[str, float] | None = None,
    best_threshold: float | None = None,
    title: str = "Meta-Labeling Analysis",
) -> go.Figure:
    """Create 2×2 dashboard for meta-labeling model analysis.
    
    Panels:
      (1,1) Equity curve: always trade vs meta-filtered
      (1,2) Win rate vs trade fraction (bet sizing sweep)
      (2,1) Meta-probability distribution by correctness
      (2,2) AUC progression across phases
    
    Args:
        meta_prob: Meta-label probabilities (P(primary correct)), shape (N,)
        meta_labels: Actual meta-labels (1=correct, 0=wrong), shape (N,)
        trade_returns: Trade returns (+1 if correct, -1 if wrong), shape (N,)
        bet_sweep: Bet sizing sweep results, list of dicts with keys:
            {"threshold", "frac", "win_rate", "sharpe", ...}
        persist_acc: Persistence model accuracy (for reference line)
        auc_progression: Dict mapping phase names to AUC values (optional)
        best_threshold: Optimal threshold from bet sweep (for equity curve)
        title: Dashboard title
    
    Returns:
        Plotly Figure object (2×2 subplots)
    
    Example:
        >>> meta_prob = model.predict_proba(X_test)[:, 1]
        >>> meta_labels = (y_pred == y_test).astype(int)
        >>> trade_returns = np.where(y_pred == y_test, 1.0, -1.0)
        >>> bet_sweep = [
        ...     {"threshold": 0.5, "frac": 1.0, "win_rate": 0.87, "sharpe": 1.2},
        ...     {"threshold": 0.7, "frac": 0.8, "win_rate": 0.90, "sharpe": 1.5},
        ... ]
        >>> fig = plot_meta_labeling_dashboard(
        ...     meta_prob, meta_labels, trade_returns, bet_sweep, persist_acc=0.87
        ... )
        >>> fig.show()
    """
    if best_threshold is None:
        best_threshold = max(bet_sweep, key=lambda b: b["sharpe"])["threshold"]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Equity Curve: Always Trade vs Meta-Filtered",
            "Win Rate vs Trade Fraction",
            "Meta-Label Probability Distribution",
            "AUC Progression (Phases)",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )
    
    # ── Panel (1,1): Equity curves ──────────────────────────────
    mask_filtered = meta_prob >= best_threshold
    eq_always = np.cumsum(trade_returns)
    eq_filtered = np.cumsum(np.where(mask_filtered, trade_returns, 0.0))
    
    bar_idx = np.arange(len(meta_prob))
    fig.add_trace(
        go.Scatter(
            x=bar_idx, y=eq_always,
            mode="lines", name="Always Trade",
            line=dict(color="#8B949E", width=1.5),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=bar_idx, y=eq_filtered,
            mode="lines", name=f"Meta-Filtered (t>={best_threshold:.2f})",
            line=dict(color="#52C41A", width=2),
        ),
        row=1, col=1,
    )
    
    # ── Panel (1,2): Win rate vs trade fraction ─────────────────
    fig.add_trace(
        go.Scatter(
            x=[b["frac"] for b in bet_sweep],
            y=[b["win_rate"] for b in bet_sweep],
            mode="lines+markers",
            marker=dict(size=6, color="#58A6FF"),
            line=dict(color="#58A6FF", width=2),
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_hline(
        y=persist_acc, row=1, col=2,
        line_dash="dash", line_color="#F0883E",
        annotation_text=f"Persist {persist_acc:.1%}",
        annotation_font_color="#F0883E",
    )
    
    # ── Panel (2,1): Probability distribution ───────────────────
    fig.add_trace(
        go.Histogram(
            x=meta_prob[meta_labels == 1],
            name="Correct (meta=1)",
            marker_color="#52C41A",
            opacity=0.7,
            nbinsx=30,
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=meta_prob[meta_labels == 0],
            name="Wrong (meta=0)",
            marker_color="#FF4D4F",
            opacity=0.7,
            nbinsx=30,
        ),
        row=2, col=1,
    )
    
    # ── Panel (2,2): AUC progression ────────────────────────────
    if auc_progression is not None:
        fig.add_trace(
            go.Bar(
                x=list(auc_progression.keys()),
                y=list(auc_progression.values()),
                marker_color=["#8B949E", "#8B949E", "#F0883E", "#58A6FF", "#52C41A"][:len(auc_progression)],
                showlegend=False,
            ),
            row=2, col=2,
        )
        fig.add_hline(
            y=0.5, row=2, col=2,
            line_dash="dash", line_color="#8B949E",
        )
    
    # ── Layout ──────────────────────────────────────────────────
    fig.update_layout(
        title_text=title,
        title_font_color="#E6EDF3",
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        font_color="#E6EDF3",
        height=800,
        width=1200,
        barmode="overlay",
        legend=dict(
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor="#21262D",
        ),
    )
    fig.update_xaxes(gridcolor="#21262D")
    fig.update_yaxes(gridcolor="#21262D")
    
    # Axis labels
    fig.update_xaxes(title_text="Bar Index", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_xaxes(title_text="Trade Fraction", row=1, col=2)
    fig.update_yaxes(title_text="Win Rate", row=1, col=2)
    fig.update_xaxes(title_text="Meta-Probability", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Phase", row=2, col=2)
    fig.update_yaxes(title_text="AUC", row=2, col=2)
    
    return fig


def plot_regime_analysis(
    split_indices: list[tuple[np.ndarray, np.ndarray, int, int]],
    y_labels: np.ndarray,
    returns: np.ndarray | None = None,
    split_names: list[str] | None = None,
    title: str = "Regime Analysis by Split",
) -> go.Figure:
    """Analyze regime characteristics across CV splits.
    
    Plots label balance and (optionally) volatility for each split to identify
    regime shifts that affect model performance.
    
    Args:
        split_indices: List of (train_idx, test_idx, n_purged, n_embargo) tuples
        y_labels: Label array, shape (N,)
        returns: Return array for volatility analysis, shape (N,) (optional)
        split_names: Custom split names (default: "Split 0", "Split 1", ...)
        title: Plot title
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> split_indices = [
        ...     (train_0, test_0, 10, 20),
        ...     (train_1, test_1, 12, 18),
        ... ]
        >>> fig = plot_regime_analysis(split_indices, y_labels, returns)
        >>> fig.show()
    """
    n_splits = len(split_indices)
    if split_names is None:
        split_names = [f"Split {i}" for i in range(n_splits)]
    
    # Compute label balance and volatility per split
    label_balance = []
    volatilities = []
    
    for _, test_idx, _, _ in split_indices:
        y_test = y_labels[test_idx]
        balance = float(y_test.mean())
        label_balance.append(balance)
        
        if returns is not None:
            ret_test = returns[test_idx]
            vol = float(ret_test.std())
            volatilities.append(vol)
    
    # Create figure
    if returns is not None:
        # Two subplots: label balance + volatility
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Label Balance by Split", "Volatility by Split"],
            horizontal_spacing=0.15,
        )
        
        # Label balance
        fig.add_trace(
            go.Bar(
                x=split_names,
                y=label_balance,
                marker_color="#58A6FF",
                name="Label Balance",
            ),
            row=1, col=1,
        )
        fig.add_hline(
            y=0.5, row=1, col=1,
            line_dash="dash", line_color="#8B949E",
        )
        
        # Volatility
        fig.add_trace(
            go.Bar(
                x=split_names,
                y=volatilities,
                marker_color="#F0883E",
                name="Volatility",
            ),
            row=1, col=2,
        )
        
        fig.update_yaxes(title_text="Label Balance (P(y=1))", row=1, col=1)
        fig.update_yaxes(title_text="Return Std Dev", row=1, col=2)
    else:
        # Single plot: label balance only
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=split_names,
                y=label_balance,
                marker_color="#58A6FF",
                name="Label Balance",
            )
        )
        fig.add_hline(
            y=0.5,
            line_dash="dash", line_color="#8B949E",
        )
        fig.update_yaxes(title_text="Label Balance (P(y=1))")
    
    # Layout
    fig.update_layout(
        title_text=title,
        title_font_color="#E6EDF3",
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        font_color="#E6EDF3",
        height=400,
        width=1200 if returns is not None else 800,
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#21262D")
    fig.update_yaxes(gridcolor="#21262D")
    
    return fig
