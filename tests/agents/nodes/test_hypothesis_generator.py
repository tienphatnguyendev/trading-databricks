"""Tests for hypothesis_generator node."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from trading_databricks.agents.config import MASConfig

VALID_EXTRACTION = {
    "treatment_class": "Regulation.etf_ruling",
    "trigger_event": "BlackRock files for spot Ether ETF",
    "mechanism": "etf_inflow_demand",
    "expected_direction": "bullish",
    "evidence_spans": ["BlackRock files for spot Ether ETF"],
    "confidence": 0.78,
    "reasoning": "Filing signals institutional ETF demand.",
}

_TAXONOMY = {"Regulation": ["etf_ruling"]}
_WHITELIST = {("Regulation", "etf_inflow_demand"): ["bullish"]}


def _base_state():
    return {
        "headline": "BlackRock files for spot Ether ETF",
        "body": "BlackRock files for spot Ether ETF with the SEC.",
        "evidence": [{"actor": "BlackRock", "action": "files", "span": "BlackRock files for spot Ether ETF"}],
        "precedents": [{"headline": "Fidelity files Bitcoin ETF", "return_24h": 0.043, "direction": "up"}],
        "few_shot_examples": [],
        "extraction_path": "standard",
        "reflection_count": 0,
    }


@patch("trading_databricks.agents.nodes.hypothesis_generator._get_client")
def test_hypothesis_generator_ok(mock_get_client):
    from trading_databricks.agents.nodes.hypothesis_generator import hypothesis_generator

    mock_client = MagicMock()
    mock_client.predict.return_value = {
        "choices": [{"message": {"content": json.dumps(VALID_EXTRACTION)}}]
    }
    mock_get_client.return_value = mock_client

    out = hypothesis_generator(_base_state(), config=MASConfig(), taxonomy=_TAXONOMY, whitelist=_WHITELIST)
    assert out["extraction_status"] == "ok"
    assert out["extraction"]["mechanism"] == "etf_inflow_demand"


@patch("trading_databricks.agents.nodes.hypothesis_generator._get_client")
def test_hypothesis_generator_retries_on_parse_failure(mock_get_client):
    from trading_databricks.agents.nodes.hypothesis_generator import hypothesis_generator

    mock_client = MagicMock()
    mock_client.predict.side_effect = [
        {"choices": [{"message": {"content": "garbage not json"}}]},
        {"choices": [{"message": {"content": json.dumps(VALID_EXTRACTION)}}]},
    ]
    mock_get_client.return_value = mock_client

    out = hypothesis_generator(_base_state(), config=MASConfig(), taxonomy=_TAXONOMY, whitelist=_WHITELIST)
    assert mock_client.predict.call_count == 2
    assert out["extraction_status"] == "ok"


@patch("trading_databricks.agents.nodes.hypothesis_generator._get_client")
def test_hypothesis_generator_parse_failed_after_two_attempts(mock_get_client):
    from trading_databricks.agents.nodes.hypothesis_generator import hypothesis_generator

    mock_client = MagicMock()
    mock_client.predict.return_value = {"choices": [{"message": {"content": "still garbage"}}]}
    mock_get_client.return_value = mock_client

    out = hypothesis_generator(_base_state(), config=MASConfig(), taxonomy={}, whitelist={})
    assert mock_client.predict.call_count == 2
    assert out["extraction_status"] == "parse_failed"
    assert out["extraction"] is None


@patch("trading_databricks.agents.nodes.hypothesis_generator._get_client")
def test_hypothesis_generator_includes_judge_failures_in_reflection(mock_get_client):
    from trading_databricks.agents.nodes.hypothesis_generator import hypothesis_generator

    mock_client = MagicMock()
    mock_client.predict.return_value = {"choices": [{"message": {"content": json.dumps(VALID_EXTRACTION)}}]}
    mock_get_client.return_value = mock_client

    state = _base_state()
    state["judge_failures"] = {"grounding": "evidence not verbatim in body"}
    state["reflection_count"] = 1

    out = hypothesis_generator(state, config=MASConfig(), taxonomy=_TAXONOMY, whitelist=_WHITELIST)
    args, kwargs = mock_client.predict.call_args
    messages = kwargs["inputs"]["messages"]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "grounding" in user_content
    assert "evidence not verbatim" in user_content
    assert out["extraction_status"] == "ok_after_reflection"


@patch("trading_databricks.agents.nodes.hypothesis_generator._get_client")
def test_hypothesis_generator_failed_on_llm_exception(mock_get_client):
    from trading_databricks.agents.nodes.hypothesis_generator import hypothesis_generator

    mock_client = MagicMock()
    mock_client.predict.side_effect = RuntimeError("LLM down")
    mock_get_client.return_value = mock_client

    out = hypothesis_generator(_base_state(), config=MASConfig(), taxonomy={}, whitelist={})
    assert out["extraction_status"] == "failed"
    assert out["extraction"] is None


@patch("trading_databricks.agents.nodes.hypothesis_generator._get_client")
def test_hypothesis_generator_accepts_preloaded_taxonomy_whitelist(mock_get_client):
    from trading_databricks.agents.nodes.hypothesis_generator import hypothesis_generator

    mock_client = MagicMock()
    mock_client.predict.return_value = {
        "choices": [{"message": {"content": json.dumps(VALID_EXTRACTION)}}]
    }
    mock_get_client.return_value = mock_client

    out = hypothesis_generator(
        _base_state(),
        config=MASConfig(),
        taxonomy=_TAXONOMY,
        whitelist=_WHITELIST,
    )
    assert out["extraction_status"] == "ok"
    assert out["extraction"]["mechanism"] == "etf_inflow_demand"
