"""Centralized LangGraph state definitions for MAS agents."""
from __future__ import annotations

from typing import TypedDict


class MASState(TypedDict, total=False):
    """Type hint for MAS LangGraph state.

    total=False makes all fields optional, as nodes only interact with
    portions of the state.

    Input fields (caller-supplied):
        headline: Article headline text
        body: Full article body text (always str, empty string if absent — never None)
        published_at: ISO timestamp string

    Context fields (set by context_retriever):
        market_vol_at_publish, market_trend_at_publish, dollar_bar_close

    Retrieval fields (set by few_shot_retriever):
        few_shot_examples, retrieval_story_ids

    Routing fields (set by uncertainty_router):
        extraction_path

    Extraction fields (set by causality_extractor):
        extraction

    Validation fields (set by mechanism_validator):
        validation_status, rejected_mechanism_raw, evidence_relevance_score
    """

    # --- input ---
    headline: str
    body: str                        # always str; use "" if absent, never None
    published_at: str

    # --- context (context_retriever) ---
    market_vol_at_publish: float | None
    market_trend_at_publish: str | None
    dollar_bar_close: float | None

    # --- retrieval (few_shot_retriever) ---
    few_shot_examples: list[dict]
    retrieval_story_ids: list[str]

    # --- routing (uncertainty_router) ---
    extraction_path: str

    # --- extraction (causality_extractor) ---
    extraction: dict | None

    # --- validation (mechanism_validator) ---
    validation_status: str | None
    rejected_mechanism_raw: str | None
    evidence_relevance_score: float | None
