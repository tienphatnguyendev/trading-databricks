from trading_databricks.agents.state import MASState

def test_mas_state_instantiation():
    """Verify MASState can be instantiated with all required fields."""
    state: MASState = {
        "headline": "Test",
        "few_shot_examples": [],
        "retrieval_story_ids": [],
        "published_at": "2026-04-07T12:00:00Z",
        "market_vol_at_publish": None,
        "market_trend_at_publish": None,
        "dollar_bar_close": None,
    }
    assert state["headline"] == "Test"
    assert state["market_vol_at_publish"] is None
from typing import get_type_hints

def test_mas_state_has_extraction_path():
    hints = get_type_hints(MASState)
    assert "extraction_path" in hints
    assert hints["extraction_path"] is str
