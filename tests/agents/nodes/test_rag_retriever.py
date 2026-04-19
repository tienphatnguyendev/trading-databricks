"""Tests for rag_retriever node."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trading_databricks.agents.config import MASConfig


def _state_with_evidence():
    return {
        "headline": "BlackRock files for spot Ether ETF",
        "body": "BlackRock files for spot Ether ETF.",
        "symbol": "ETHUSDT",
        "published_at": "2026-04-18T10:00:00Z",
        "evidence": [{"actor": "BlackRock", "action": "files", "span": "BlackRock files for spot Ether ETF"}],
        "evidence_status": "ok",
    }


@patch("trading_databricks.agents.nodes.rag_retriever._join_market_outcomes")
@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose")
def test_rag_retriever_ok(mock_query, mock_join):
    from trading_databricks.agents.nodes.rag_retriever import rag_retriever

    mock_query.return_value = [
        {"headline": "Fidelity files Bitcoin ETF", "body_excerpt": "...", "timestamp": "2024-06-15T12:00:00Z"},
    ]
    mock_join.return_value = [
        {"headline": "Fidelity files Bitcoin ETF", "body_excerpt": "...", "timestamp": "2024-06-15T12:00:00Z", "return_24h": 0.043, "direction": "up"},
    ]

    out = rag_retriever(_state_with_evidence(), config=MASConfig())
    assert out["precedent_status"] == "ok"
    assert len(out["precedents"]) == 1
    assert out["precedents"][0]["direction"] == "up"


@patch("trading_databricks.agents.nodes.rag_retriever._join_market_outcomes")
@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose")
def test_rag_retriever_no_match(mock_query, mock_join):
    from trading_databricks.agents.nodes.rag_retriever import rag_retriever

    mock_query.return_value = []
    out = rag_retriever(_state_with_evidence(), config=MASConfig())
    mock_join.assert_not_called()
    assert out["precedent_status"] == "no_match"
    assert out["precedents"] == []


@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose")
def test_rag_retriever_unavailable_on_exception(mock_query):
    from trading_databricks.agents.nodes.rag_retriever import rag_retriever

    mock_query.side_effect = RuntimeError("Vector Search down")
    out = rag_retriever(_state_with_evidence(), config=MASConfig())
    assert out["precedent_status"] == "unavailable"
    assert out["precedents"] == []


@patch("trading_databricks.agents.nodes.rag_retriever._join_market_outcomes")
@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose")
def test_rag_retriever_drops_precedents_with_failed_join(mock_query, mock_join):
    from trading_databricks.agents.nodes.rag_retriever import rag_retriever

    mock_query.return_value = [
        {"headline": "A", "timestamp": "t1"},
        {"headline": "B", "timestamp": "t2"},
    ]
    # Join only returns one — the other was dropped due to missing OHLC
    mock_join.return_value = [
        {"headline": "A", "timestamp": "t1", "return_24h": 0.01, "direction": "up"},
    ]

    out = rag_retriever(_state_with_evidence(), config=MASConfig())
    assert out["precedent_status"] == "ok"
    assert len(out["precedents"]) == 1


@patch("trading_databricks.agents.nodes.rag_retriever._join_market_outcomes")
@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose")
def test_rag_retriever_no_match_when_all_joins_fail(mock_query, mock_join):
    from trading_databricks.agents.nodes.rag_retriever import rag_retriever

    mock_query.return_value = [{"headline": "A", "timestamp": "t1"}]
    mock_join.return_value = []  # all dropped

    out = rag_retriever(_state_with_evidence(), config=MASConfig())
    assert out["precedent_status"] == "no_match"
    assert out["precedents"] == []
