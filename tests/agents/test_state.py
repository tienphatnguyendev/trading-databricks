import pytest
from typing import get_type_hints
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

def test_mas_state_has_extraction_path():
    hints = get_type_hints(MASState)
    assert "extraction_path" in hints
    assert hints["extraction_path"] is str


def test_mas_state_has_body_and_extraction_fields():
    """state.py must have body, extraction, and validator output fields."""
    state: MASState = {
        "headline": "Test",
        "body": "Full article text here.",
        "extraction": {"treatment_class": "Regulation.etf_ruling"},
        "validation_status": "accepted",
        "rejected_mechanism_raw": None,
        "evidence_relevance_score": 0.87,
    }
    assert state["body"] == "Full article text here."
    assert state["extraction"]["treatment_class"] == "Regulation.etf_ruling"
    assert state["validation_status"] == "accepted"
    assert state["evidence_relevance_score"] == pytest.approx(0.87)

def test_mas_state_accepts_new_subpipeline_keys():
    """Verify TypedDict accepts the 7 new keys added for causality discovery sub-pipeline."""
    from trading_databricks.agents.state import MASState

    state: MASState = {
        "evidence": [{"actor": "BlackRock", "action": "files"}],
        "evidence_status": "ok",
        "precedents": [{"headline": "x", "return_24h": 0.04}],
        "precedent_status": "ok",
        "extraction_status": "ok",
        "judge_results": {"direction": {"passed": True, "score": 0.9}},
        "judge_failures": None,
        "reflection_count": 0,
    }
    assert state["evidence_status"] == "ok"
    assert state["reflection_count"] == 0
