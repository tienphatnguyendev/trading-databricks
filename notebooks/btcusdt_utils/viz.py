"""Plotly dark theme configuration (GitHub Dark inspired)."""

from __future__ import annotations

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
