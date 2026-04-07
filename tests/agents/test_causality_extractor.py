"""Tests for the causality_extractor node."""
import pytest
from trading_databricks.agents.nodes.causality_extractor import (
    parse_json_response,
    majority_vote
)

def test_parse_json_response():
    assert parse_json_response('{"a": 1}') == {"a": 1}
    assert parse_json_response('```json\n{"b": 2}\n```') == {"b": 2}
    assert parse_json_response('invalid') is None

def test_majority_vote_consensus():
    extractions = [
        {"treatment_class": "A", "mechanism": "m", "expected_direction": "bullish", "confidence": 0.8},
        {"treatment_class": "A", "mechanism": "m", "expected_direction": "bullish", "confidence": 0.7},
        {"treatment_class": "B", "mechanism": "n", "expected_direction": "bearish", "confidence": 0.9}
    ]
    winner = majority_vote(extractions)
    assert winner["confidence"] == 0.8  # First occurrence of consensus

def test_majority_vote_tie_breaker():
    extractions = [
        {"treatment_class": "A", "mechanism": "m", "expected_direction": "bullish", "confidence": 0.6},
        {"treatment_class": "B", "mechanism": "n", "expected_direction": "bearish", "confidence": 0.9},
        {"treatment_class": "C", "mechanism": "o", "expected_direction": "volatile", "confidence": 0.7}
    ]
    winner = majority_vote(extractions)
    assert winner["treatment_class"] == "B"
    assert winner["confidence"] == pytest.approx(0.6)  # Penalized: max(0, 0.9 - 0.3)
