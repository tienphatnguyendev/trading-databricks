"""Plotly dark theme configuration (GitHub Dark inspired)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Color palette ─────────────────────────────────────────────
_BG       = "#0D1117"
_BG_CARD  = "#161B22"
_TEXT     = "#E6EDF3"
_TEXT_MUT = "#8B949E"
_GRID     = "#21262D"
_RED      = "#FF4D4F"
_GREEN    = "#52C41A"
_BLUE     = "#58A6FF"
_ORANGE   = "#F0883E"
_PURPLE   = "#BC8CFF"
_CYAN     = "#39D2C0"
_FONT     = dict(family="Inter, Arial, sans-serif", color=_TEXT)

DARK_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor=_BG,
    paper_bgcolor=_BG_CARD,
    font=_FONT,
    xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
    yaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
    margin=dict(l=60, r=30, t=50, b=50),
)


def dark_fig(rows: int = 1, cols: int = 1, **kwargs) -> go.Figure:
    """Create a Plotly figure pre-configured with dark theme."""
    fig = make_subplots(rows=rows, cols=cols, **kwargs)
    fig.update_layout(**DARK_LAYOUT)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 VISUALIZATION UTILITIES (Triple Barrier Parameter Sweep)
# ══════════════════════════════════════════════════════════════════════════════
# Dashboard builders for PT×SL heatmap and touch-bar histogram visualization.

def plot_ptsl_heatmap(
    pt_vals: list[float],
    sl_vals: list[float],
    z_matrix: list[list[float]],
    annot_text: list[list[str]],
    *,
    title: str = "PT × SL Score Heatmap",
    colorscale: str = "Viridis",
) -> go.Figure:
    """Create PT×SL heatmap with annotations.
    
    Args:
        pt_vals: Profit-taking multipliers (x-axis)
        sl_vals: Stop-loss multipliers (y-axis)
        z_matrix: Score values (2D array: SL × PT)
        annot_text: Annotation strings (2D array: SL × PT)
        title: Chart title
        colorscale: Plotly colorscale name
    
    Returns:
        Plotly Figure with dark theme
    
    Example:
        >>> pt = [0.5, 1.0, 1.5]
        >>> sl = [0.5, 1.0, 1.5]
        >>> scores = [[0.45, 0.47, 0.46], [0.46, 0.48, 0.47], [0.44, 0.46, 0.45]]
        >>> annot = [["0.45", "0.47", "0.46"], ["0.46", "0.48", "0.47"], ["0.44", "0.46", "0.45"]]
        >>> fig = plot_ptsl_heatmap(pt, sl, scores, annot)
        >>> fig.show()
    """
    fig = dark_fig()
    
    fig.add_trace(go.Heatmap(
        x=[f"PT={pt:.1f}σ" for pt in pt_vals],
        y=[f"SL={sl:.1f}σ" for sl in sl_vals],
        z=z_matrix,
        text=annot_text,
        texttemplate="%{text}",
        textfont={"size": 10, "color": _TEXT},
        colorscale=colorscale,
        colorbar=dict(
            title=dict(text="Score", side="right"),
            tickfont={"color": _TEXT},
        ),
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color=_TEXT, size=16)),
        xaxis=dict(title="Profit-Taking", gridcolor=_GRID),
        yaxis=dict(title="Stop-Loss", gridcolor=_GRID),
        width=700,
        height=600,
    )
    
    return fig


def plot_touch_bar_histogram(
    touch_bar_arrays: dict[str, np.ndarray],
    config_names: list[str],
    *,
    median_ref: float | None = None,
    title: str = "Touch-Bar Distribution (Top Configs)",
    colors: list[str] | None = None,
) -> go.Figure:
    """Overlay histograms of touch-bar counts for top configurations.
    
    Args:
        touch_bar_arrays: Dict mapping config name -> touch_bars array
        config_names: Ordered list of config names to plot
        median_ref: Optional reference median to highlight (e.g., 3.0)
        title: Chart title
        colors: Optional color list (default: uses theme palette)
    
    Returns:
        Plotly Figure with dark theme
    
    Example:
        >>> arrays = {
        ...     "PT1.0_SL1.0": np.array([1, 2, 3, 2, 1]),
        ...     "PT2.0_SL2.0": np.array([2, 3, 4, 3, 2]),
        ... }
        >>> fig = plot_touch_bar_histogram(arrays, list(arrays.keys()), median_ref=3.0)
        >>> fig.show()
    """
    if colors is None:
        colors = [_BLUE, _GREEN, _ORANGE, _PURPLE, _CYAN]
    
    fig = dark_fig()
    
    for i, name in enumerate(config_names):
        if name not in touch_bar_arrays:
            continue
        touch_bars = touch_bar_arrays[name]
        color = colors[i % len(colors)]
        median_val = float(np.median(touch_bars))
        
        fig.add_trace(go.Histogram(
            x=touch_bars,
            name=f"{name} (med={median_val:.1f})",
            marker_color=color,
            opacity=0.7,
            nbinsx=30,
        ))
    
    if median_ref is not None:
        fig.add_vline(
            x=median_ref,
            line_dash="dash",
            line_color=_RED,
            line_width=2,
            annotation_text=f"Target med={median_ref:.1f}",
            annotation_font_color=_RED,
            annotation_position="top",
        )
    
    fig.update_layout(
        title=dict(text=title, font=dict(color=_TEXT, size=16)),
        xaxis=dict(title="Bars Until Touch", gridcolor=_GRID),
        yaxis=dict(title="Frequency", gridcolor=_GRID),
        barmode="overlay",
        width=900,
        height=600,
        legend=dict(
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor=_GRID,
        ),
    )
    
    return fig


def plot_sweep_dashboard(
    heatmap_data: tuple[list[float], list[float], list[list[float]], list[list[str]]],
    touch_bar_data: dict[str, np.ndarray],
    *,
    top_configs: int | list[str] = 3,
    median_ref: float | None = None,
    title: str = "Phase 2: Triple Barrier Parameter Sweep",
) -> go.Figure:
    """Combined 1×2 dashboard: PT×SL heatmap + touch-bar histograms.
    
    Args:
        heatmap_data: Tuple of (pt_vals, sl_vals, z_matrix, annot_text)
        touch_bar_data: Dict mapping config name -> touch_bars array
        top_configs: Number of top configs OR list of config names to plot
        median_ref: Optional reference median for touch-bar histogram
        title: Dashboard title
    
    Returns:
        Plotly Figure (1×2 subplots) with dark theme
    
    Example:
        >>> heatmap_data = ([0.5, 1.0], [0.5, 1.0], [[0.45, 0.47], [0.46, 0.48]], [["0.45", "0.47"], ["0.46", "0.48"]])
        >>> touch_data = {"PT1.0_SL1.0": np.array([1, 2, 3])}
        >>> fig = plot_sweep_dashboard(heatmap_data, touch_data, top_configs=1)
        >>> fig.show()
    """
    pt_vals, sl_vals, z_matrix, annot_text = heatmap_data
    
    # Determine which configs to plot
    if isinstance(top_configs, int):
        config_names = list(touch_bar_data.keys())[:top_configs]
    else:
        config_names = top_configs
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["PT × SL Score Heatmap", "Touch-Bar Distribution (Top Configs)"],
        horizontal_spacing=0.12,
    )
    
    # ── Panel (1,1): PT×SL heatmap ─────────────────────────────────────────
    fig.add_trace(
        go.Heatmap(
            x=[f"PT={pt:.1f}σ" for pt in pt_vals],
            y=[f"SL={sl:.1f}σ" for sl in sl_vals],
            z=z_matrix,
            text=annot_text,
            texttemplate="%{text}",
            textfont={"size": 10, "color": _TEXT},
            colorscale="Viridis",
            colorbar=dict(
                title=dict(text="Score", side="right"),
                tickfont={"color": _TEXT},
                x=0.46,  # Position colorbar between panels
            ),
            showscale=True,
        ),
        row=1, col=1,
    )
    
    # ── Panel (1,2): Touch-bar histograms ──────────────────────────────────
    colors = [_BLUE, _GREEN, _ORANGE, _PURPLE, _CYAN]
    for i, name in enumerate(config_names):
        if name not in touch_bar_data:
            continue
        touch_bars = touch_bar_data[name]
        color = colors[i % len(colors)]
        median_val = float(np.median(touch_bars))
        
        fig.add_trace(
            go.Histogram(
                x=touch_bars,
                name=f"{name} (med={median_val:.1f})",
                marker_color=color,
                opacity=0.7,
                nbinsx=30,
                showlegend=True,
            ),
            row=1, col=2,
        )
    
    if median_ref is not None:
        fig.add_vline(
            x=median_ref,
            row=1, col=2,
            line_dash="dash",
            line_color=_RED,
            line_width=2,
            annotation_text=f"Target={median_ref:.1f}",
            annotation_font_color=_RED,
            annotation_position="top right",
        )
    
    # ── Layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, font=dict(color=_TEXT, size=18)),
        plot_bgcolor=_BG,
        paper_bgcolor=_BG_CARD,
        font=dict(color=_TEXT),
        barmode="overlay",
        width=1400,
        height=600,
        legend=dict(
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor=_GRID,
            x=1.0,
            y=1.0,
        ),
    )
    
    fig.update_xaxes(title="Profit-Taking", gridcolor=_GRID, row=1, col=1)
    fig.update_yaxes(title="Stop-Loss", gridcolor=_GRID, row=1, col=1)
    fig.update_xaxes(title="Bars Until Touch", gridcolor=_GRID, row=1, col=2)
    fig.update_yaxes(title="Frequency", gridcolor=_GRID, row=1, col=2)
    
    return fig
