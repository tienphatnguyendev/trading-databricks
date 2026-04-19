"""Unit tests for _interpret_judge_response — covers all three judge branches.

These tests call the parser directly, one level below the _invoke_judge mock
boundary used by test_judge_panel.py. They catch contract bugs in the YAML
prompt that higher-level mocks would hide.
"""
from __future__ import annotations

import pytest

from trading_databricks.agents.nodes.judge_panel import _interpret_judge_response


# --- direction judge ---

def test_direction_up_passes():
    result = _interpret_judge_response("judge_direction", "up")
    assert result == {"passed": True, "score": "up", "reason": ""}


def test_direction_down_passes():
    result = _interpret_judge_response("judge_direction", "down")
    assert result == {"passed": True, "score": "down", "reason": ""}


def test_direction_neutral_passes():
    result = _interpret_judge_response("judge_direction", "neutral")
    assert result == {"passed": True, "score": "neutral", "reason": ""}


def test_direction_unknown_fails_with_reason():
    result = _interpret_judge_response("judge_direction", "MAYBE")
    assert result["passed"] is False
    assert "unparseable" in result["reason"]


# --- mechanism judge ---

def test_mechanism_match_passes():
    result = _interpret_judge_response("judge_mechanism", "match")
    assert result["passed"] is True
    assert result["score"] == "match"


def test_mechanism_mismatch_fails():
    result = _interpret_judge_response("judge_mechanism", "mismatch")
    assert result["passed"] is False


def test_mechanism_name_instead_of_verdict_fails():
    """If LLM returns a mechanism name instead of match/mismatch, it must fail."""
    result = _interpret_judge_response("judge_mechanism", "etf_inflow_demand")
    assert result["passed"] is False


# --- grounding judge ---

def test_grounding_valid_float_score_only():
    """_interpret_judge_response must NOT apply the threshold — only return the score."""
    result = _interpret_judge_response("judge_grounding", "0.75")
    assert result["score"] == pytest.approx(0.75)
    assert result["passed"] is None   # threshold applied by judge_panel caller


def test_grounding_zero_score():
    result = _interpret_judge_response("judge_grounding", "0.0")
    assert result["score"] == pytest.approx(0.0)
    assert result["passed"] is None


def test_grounding_invalid_text_fails():
    result = _interpret_judge_response("judge_grounding", "high confidence")
    assert result["passed"] is False
    assert result["score"] is None
    assert "unparseable" in result["reason"]


# --- _invoke_judge integration (renders real YAML prompt) ---

from unittest.mock import MagicMock, patch


@patch("trading_databricks.agents.nodes.judge_panel._get_client")
@patch("mlflow.load_prompt", side_effect=Exception("not registered"))
def test_invoke_judge_mechanism_passes_on_match_response(mock_load_prompt, mock_get_client):
    """Verify _invoke_judge can render judge_mechanism without KeyError and returns
    passed=True when the LLM responds 'match'.

    This test fails before the YAML fix because render_messages raises
    KeyError: 'whitelist_injection'.
    """
    from trading_databricks.agents.nodes.judge_panel import _invoke_judge

    mock_client = MagicMock()
    mock_client.predict.return_value = {"choices": [{"message": {"content": "match"}}]}
    mock_get_client.return_value = mock_client

    result = _invoke_judge(
        "judge_mechanism",
        headline="BlackRock files for spot Ether ETF",
        body="BlackRock files for spot Ether ETF with the SEC.",
        extraction={"mechanism": "etf_inflow_demand", "expected_direction": "bullish"},
        endpoint="test-ep",
        timeout_s=10.0,
    )
    assert result["passed"] is True
    assert result["score"] == "match"
