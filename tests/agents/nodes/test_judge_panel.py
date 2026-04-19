"""Tests for judge_panel node."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from trading_databricks.agents.config import MASConfig


def _state():
    return {
        "headline": "BlackRock files for spot Ether ETF",
        "body": "BlackRock files for spot Ether ETF.",
        "extraction": {
            "treatment_class": "Regulation.etf_ruling",
            "mechanism": "etf_inflow_demand",
            "expected_direction": "bullish",
            "evidence_spans": ["BlackRock files for spot Ether ETF"],
            "confidence": 0.78,
        },
    }


@patch("trading_databricks.agents.nodes.judge_panel._invoke_judge")
def test_judge_panel_all_pass(mock_invoke):
    from trading_databricks.agents.nodes.judge_panel import judge_panel

    # NOTE: direction judge returns "up"/"down"/"neutral"; judge_panel maps
    # extraction.expected_direction ("bullish") → "up" before comparing.
    mock_invoke.side_effect = [
        {"passed": True, "score": "up", "reason": ""},
        {"passed": True, "score": "match", "reason": ""},
        {"passed": True, "score": 0.9, "reason": ""},
    ]

    out = judge_panel(_state(), config=MASConfig())
    assert out["judge_failures"] is None
    assert out["judge_results"]["direction"]["passed"] is True
    assert out["judge_results"]["mechanism"]["passed"] is True
    assert out["judge_results"]["grounding"]["passed"] is True


@patch("trading_databricks.agents.nodes.judge_panel._invoke_judge")
def test_judge_panel_strict_and_one_fails(mock_invoke):
    from trading_databricks.agents.nodes.judge_panel import judge_panel

    # Grounding score 0.4 < threshold 0.5 → judge_panel rewrites reason to
    # "score 0.40 < threshold 0.5". The mock's "reason" is overwritten.
    mock_invoke.side_effect = [
        {"passed": True, "score": "up", "reason": ""},
        {"passed": True, "score": "match", "reason": ""},
        {"passed": False, "score": 0.4, "reason": "raw reason discarded"},
    ]

    out = judge_panel(_state(), config=MASConfig())
    assert "grounding" in out["judge_failures"]
    assert "threshold" in out["judge_failures"]["grounding"]


@patch("trading_databricks.agents.nodes.judge_panel._invoke_judge")
def test_judge_panel_individual_judge_unavailable(mock_invoke):
    from trading_databricks.agents.nodes.judge_panel import judge_panel

    mock_invoke.side_effect = [
        {"passed": True, "score": "up", "reason": ""},
        RuntimeError("judge LLM down"),
        {"passed": True, "score": 0.9, "reason": ""},
    ]

    out = judge_panel(_state(), config=MASConfig())
    assert out["judge_results"]["mechanism"]["passed"] is False
    assert out["judge_results"]["mechanism"]["score"] is None
    assert out["judge_results"]["mechanism"]["reason"] == "judge_unavailable"
    assert "mechanism" in out["judge_failures"]


@patch("trading_databricks.agents.nodes.judge_panel._invoke_judge")
def test_judge_panel_all_unavailable_marks_unjudged(mock_invoke):
    from trading_databricks.agents.nodes.judge_panel import judge_panel

    mock_invoke.side_effect = [RuntimeError("x"), RuntimeError("y"), RuntimeError("z")]

    out = judge_panel(_state(), config=MASConfig())
    assert out["judge_results"] == {}
    assert out["extraction_status"] == "unjudged"
