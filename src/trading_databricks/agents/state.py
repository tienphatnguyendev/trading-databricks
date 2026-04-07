"""Centralized LangGraph state definitions for MAS agents."""
from __future__ import annotations

from typing import TypedDict

class MASState(TypedDict, total=False):
    """Type hint for MAS LangGraph state.
    
    total=False makes all fields optional, as nodes only interact with
    portions of the state.
    """
    headline: str
    few_shot_examples: list[dict]
    retrieval_story_ids: list[str]
    published_at: str
    market_vol_at_publish: float | None
    market_trend_at_publish: str | None
    dollar_bar_close: float | None
