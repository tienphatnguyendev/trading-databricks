"""Tests for evidence_extractor node."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trading_databricks.agents.config import MASConfig


@patch("trading_databricks.agents.nodes.evidence_extractor._get_client")
def test_evidence_extractor_ok(mock_get_client):
    from trading_databricks.agents.nodes.evidence_extractor import evidence_extractor

    mock_client = MagicMock()
    mock_client.predict.return_value = {
        "choices": [{"message": {"content": '[{"actor": "BlackRock", "action": "files", "object": "spot Ether ETF", "span": "BlackRock files for spot Ether ETF", "timestamp_hint": null}]'}}]
    }
    mock_get_client.return_value = mock_client

    state = {"headline": "BlackRock files for spot Ether ETF", "body": "BlackRock files for spot Ether ETF with the SEC."}
    out = evidence_extractor(state, config=MASConfig())

    assert out["evidence_status"] == "ok"
    assert len(out["evidence"]) == 1
    assert out["evidence"][0]["actor"] == "BlackRock"


@patch("trading_databricks.agents.nodes.evidence_extractor._get_client")
def test_evidence_extractor_empty_on_malformed_json(mock_get_client):
    from trading_databricks.agents.nodes.evidence_extractor import evidence_extractor

    mock_client = MagicMock()
    mock_client.predict.return_value = {
        "choices": [{"message": {"content": "not json at all"}}]
    }
    mock_get_client.return_value = mock_client

    out = evidence_extractor({"headline": "x", "body": "y"}, config=MASConfig())

    assert out["evidence_status"] == "empty"
    assert out["evidence"] == []


@patch("trading_databricks.agents.nodes.evidence_extractor._get_client")
def test_evidence_extractor_failed_on_llm_exception(mock_get_client):
    from trading_databricks.agents.nodes.evidence_extractor import evidence_extractor

    mock_client = MagicMock()
    mock_client.predict.side_effect = RuntimeError("LLM down")
    mock_get_client.return_value = mock_client

    out = evidence_extractor({"headline": "x", "body": "y"}, config=MASConfig())

    assert out["evidence_status"] == "failed"
    assert out["evidence"] == []
    assert out["extraction_status"] == "failed"
