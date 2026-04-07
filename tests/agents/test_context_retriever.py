from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.nodes.context_retriever import context_retriever
from trading_databricks.agents.state import MASState
from trading_databricks.agents.tools import fetch_market_context

def test_context_retriever_stub():
    """Verify context_retriever returns None for market fields."""
    initial_state: MASState = {
        "headline": "Bitcoin surges",
        "published_at": "2026-04-07T12:00:00Z"
    }
    config = MASConfig()
    
    new_state = context_retriever(initial_state, config=config)
    
    assert new_state["headline"] == "Bitcoin surges"
    assert new_state["published_at"] == "2026-04-07T12:00:00Z"
    assert new_state["market_vol_at_publish"] is None
    assert new_state["market_trend_at_publish"] is None
    assert new_state["dollar_bar_close"] is None

def test_fetch_market_context_tool():
    """Verify fetch_market_context tool returns None for market fields."""
    result = fetch_market_context("2026-04-07T12:00:00Z")
    
    assert result == {
        "market_vol_at_publish": None,
        "market_trend_at_publish": None,
        "dollar_bar_close": None,
    }