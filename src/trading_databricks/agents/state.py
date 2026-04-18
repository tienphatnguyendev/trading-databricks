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
        symbol: Asset symbol (e.g., "BTCUSDT", "ETHUSDT") — used by rag_retriever for OHLC join

    Context fields (set by context_retriever):
        market_vol_at_publish, market_trend_at_publish, dollar_bar_close

    Retrieval fields (set by few_shot_retriever):
        few_shot_examples, retrieval_story_ids

    Routing fields (set by uncertainty_router):
        extraction_path

    Sub-pipeline fields (added 2026-04-18):
        evidence_extractor → evidence, evidence_status
        rag_retriever      → precedents, precedent_status
        hypothesis_generator → extraction, extraction_status
        judge_panel        → judge_results, judge_failures
        reflector          → reflection_count, extraction_status (overwrites)

    Validation fields (set by mechanism_validator):
        validation_status, rejected_mechanism_raw, evidence_relevance_score
    """

    # --- input ---
    headline: str
    body: str                        # always str; use "" if absent, never None
    published_at: str
    symbol: str                      # e.g., "BTCUSDT" — used by rag_retriever

    # --- context (context_retriever) ---
    market_vol_at_publish: float | None
    market_trend_at_publish: str | None
    dollar_bar_close: float | None

    # --- retrieval (few_shot_retriever) ---
    few_shot_examples: list[dict]
    retrieval_story_ids: list[str]

    # --- routing (uncertainty_router) ---
    extraction_path: str

    # --- sub-pipeline: evidence_extractor ---
    evidence: list[dict]
    evidence_status: str             # "ok" | "empty" | "failed"

    # --- sub-pipeline: rag_retriever ---
    precedents: list[dict]
    precedent_status: str            # "ok" | "no_match" | "unavailable"

    # --- sub-pipeline: hypothesis_generator ---
    extraction: dict | None
    extraction_status: str           # "ok" | "ok_after_reflection" | "judge_failed_accepted" | "unjudged" | "parse_failed" | "failed"

    # --- sub-pipeline: judge_panel ---
    judge_results: dict | None
    judge_failures: dict | None      # populated when judges fail; fed back into hypothesis_generator on reflection

    # --- sub-pipeline: reflector ---
    reflection_count: int
    _route: str

    # --- validation (mechanism_validator) ---
    validation_status: str | None
    rejected_mechanism_raw: str | None
    evidence_relevance_score: float | None
