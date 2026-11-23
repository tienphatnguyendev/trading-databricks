"""Context retriever node for MAS agent pipeline.

V1 stub: returns NULL for market context fields until TRD-5 is complete.
"""
from __future__ import annotations

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.state import MASState

def context_retriever(state: MASState, *, config: MASConfig) -> MASState:
    """Fetch pre-treatment market context X.
    
    V1 stub: gold.dollar_bars not yet available (TRD-5 dependency).
    Returns NULL for all market context fields.
    Schema fields 18-20 in CAUSAL_FEATURES_SCHEMA remain nullable for this reason.
    
    V2 (post-TRD-5): query gold.dollar_bars for bar nearest to
    state["published_at"], extract vol, trend, close.
    """
    return {
        **state,
        "market_vol_at_publish": None,
        "market_trend_at_publish": None,
        "dollar_bar_close": None,
    }
