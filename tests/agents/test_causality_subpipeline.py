"""End-to-end integration test for the causality sub-pipeline.

Runs build_mas_graph() with per-node mocked LLM and Vector Search externals
and verifies the happy path, reflection path, and budget-exhausted path produce
the expected terminal state.

Mocking strategy:
  - evidence_extractor LLM  → patch evidence_extractor._get_client
  - hypothesis_generator LLM → patch hypothesis_generator._get_client
  - judge_panel judges       → patch judge_panel._invoke_judge
  - RAG                      → patch rag_retriever._query_firehose / _join_market_outcomes
  - validator embed + whitelist → patch graph.mlflow.deployments.get_deploy_client + graph.load_whitelist
  - taxonomy/whitelist at construction → patch graph.load_taxonomy + graph.load_whitelist + graph.get_spark_session
"""
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
    "reasoning": "Filing signals institutional ETF demand grounded in precedent.",
}

_TAXONOMY = {"Regulation": ["etf_ruling"]}
_WHITELIST = {("Regulation", "etf_inflow_demand"): ["bullish"]}


def _entry_state():
    return {
        "headline": "BlackRock files for spot Ether ETF",
        "body": "BlackRock files for spot Ether ETF with the SEC.",
        "published_at": "2026-04-18T10:00:00Z",
        "symbol": "ETHUSDT",
    }


@patch("trading_databricks.agents.nodes.judge_panel._invoke_judge")
@patch("trading_databricks.agents.nodes.hypothesis_generator._get_client")
@patch("trading_databricks.agents.nodes.evidence_extractor._get_client")
@patch("trading_databricks.agents.nodes.rag_retriever._join_market_outcomes")
@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose")
def test_subpipeline_happy_path(
    mock_firehose, mock_join, mock_ev_client, mock_hyp_client, mock_judge,
):
    from trading_databricks.agents.graph import build_mas_graph

    # Evidence LLM
    ev_client = MagicMock()
    ev_client.predict.return_value = {"choices": [{"message": {"content": '[{"actor": "BlackRock", "action": "files", "object": "spot Ether ETF", "span": "BlackRock files for spot Ether ETF", "timestamp_hint": null}]'}}]}
    mock_ev_client.return_value = ev_client

    # RAG
    mock_firehose.return_value = [{"headline": "Fidelity files Bitcoin ETF", "timestamp": "2024-06-15T12:00:00Z"}]
    mock_join.return_value = [{"headline": "Fidelity files Bitcoin ETF", "timestamp": "2024-06-15T12:00:00Z", "return_24h": 0.043, "direction": "up"}]

    # Hypothesis LLM
    hyp_client = MagicMock()
    hyp_client.predict.return_value = {"choices": [{"message": {"content": json.dumps(VALID_EXTRACTION)}}]}
    mock_hyp_client.return_value = hyp_client

    # Judges — all pass
    mock_judge.side_effect = [
        {"passed": True, "score": "up", "reason": ""},
        {"passed": True, "score": "match", "reason": ""},
        {"passed": True, "score": 0.9, "reason": ""},
    ]

    with patch("trading_databricks.agents.graph.get_spark_session"), \
         patch("trading_databricks.agents.graph.load_taxonomy", return_value=_TAXONOMY), \
         patch("trading_databricks.agents.graph.load_whitelist", return_value=_WHITELIST), \
         patch("trading_databricks.agents.graph.mlflow.deployments.get_deploy_client") as mock_val_client, \
         patch("trading_databricks.agents.nodes.context_retriever", side_effect=lambda s, config: {"market_vol_at_publish": 0.02, "market_trend_at_publish": "up", "dollar_bar_close": 3500.0}), \
         patch("trading_databricks.agents.nodes.few_shot_retriever", side_effect=lambda s, config: {"few_shot_examples": [], "retrieval_story_ids": []}), \
         patch("trading_databricks.agents.nodes.uncertainty_router", side_effect=lambda s, config: {"extraction_path": "standard"}):
        mock_val_client.return_value.predict.return_value = {"data": [{"embedding": [0.1] * 8}]}
        graph = build_mas_graph(config=MASConfig())
        result = graph.invoke(_entry_state())

    assert result["extraction"]["mechanism"] == "etf_inflow_demand"
    assert result["extraction_status"] == "ok"
    assert result["reflection_count"] == 0
    assert result["validation_status"] == "accepted"


@patch("trading_databricks.agents.nodes.judge_panel._invoke_judge")
@patch("trading_databricks.agents.nodes.hypothesis_generator._get_client")
@patch("trading_databricks.agents.nodes.evidence_extractor._get_client")
@patch("trading_databricks.agents.nodes.rag_retriever._join_market_outcomes")
@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose")
def test_subpipeline_reflects_then_passes(
    mock_firehose, mock_join, mock_ev_client, mock_hyp_client, mock_judge,
):
    from trading_databricks.agents.graph import build_mas_graph

    ev_client = MagicMock()
    ev_client.predict.return_value = {"choices": [{"message": {"content": '[{"actor": "BlackRock", "action": "files", "span": "BlackRock files for spot Ether ETF"}]'}}]}
    mock_ev_client.return_value = ev_client

    mock_firehose.return_value = [{"headline": "x", "timestamp": "t"}]
    mock_join.return_value = [{"headline": "x", "timestamp": "t", "return_24h": 0.04, "direction": "up"}]

    hyp_client = MagicMock()
    hyp_client.predict.return_value = {"choices": [{"message": {"content": json.dumps(VALID_EXTRACTION)}}]}
    mock_hyp_client.return_value = hyp_client

    # First panel — grounding fails; second panel — all pass
    mock_judge.side_effect = [
        {"passed": True, "score": "up", "reason": ""},
        {"passed": True, "score": "match", "reason": ""},
        {"passed": False, "score": 0.3, "reason": "low score"},
        {"passed": True, "score": "up", "reason": ""},
        {"passed": True, "score": "match", "reason": ""},
        {"passed": True, "score": 0.9, "reason": ""},
    ]

    with patch("trading_databricks.agents.graph.get_spark_session"), \
         patch("trading_databricks.agents.graph.load_taxonomy", return_value=_TAXONOMY), \
         patch("trading_databricks.agents.graph.load_whitelist", return_value=_WHITELIST), \
         patch("trading_databricks.agents.graph.mlflow.deployments.get_deploy_client") as mock_val_client, \
         patch("trading_databricks.agents.nodes.context_retriever", side_effect=lambda s, config: {"market_vol_at_publish": 0.02, "market_trend_at_publish": "up", "dollar_bar_close": 3500.0}), \
         patch("trading_databricks.agents.nodes.few_shot_retriever", side_effect=lambda s, config: {"few_shot_examples": [], "retrieval_story_ids": []}), \
         patch("trading_databricks.agents.nodes.uncertainty_router", side_effect=lambda s, config: {"extraction_path": "standard"}):
        mock_val_client.return_value.predict.return_value = {"data": [{"embedding": [0.1] * 8}]}
        graph = build_mas_graph(config=MASConfig())
        result = graph.invoke(_entry_state())

    assert result["extraction_status"] == "ok_after_reflection"
    assert result["reflection_count"] == 1


@patch("trading_databricks.agents.nodes.judge_panel._invoke_judge")
@patch("trading_databricks.agents.nodes.hypothesis_generator._get_client")
@patch("trading_databricks.agents.nodes.evidence_extractor._get_client")
@patch("trading_databricks.agents.nodes.rag_retriever._join_market_outcomes")
@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose")
def test_subpipeline_budget_exhausted(
    mock_firehose, mock_join, mock_ev_client, mock_hyp_client, mock_judge,
):
    from trading_databricks.agents.graph import build_mas_graph

    ev_client = MagicMock()
    ev_client.predict.return_value = {"choices": [{"message": {"content": "[]"}}]}
    mock_ev_client.return_value = ev_client

    mock_firehose.return_value = []

    hyp_client = MagicMock()
    hyp_client.predict.return_value = {"choices": [{"message": {"content": json.dumps(VALID_EXTRACTION)}}]}
    mock_hyp_client.return_value = hyp_client

    # Judges always fail — 2 rounds × 3 judges = 6 calls
    mock_judge.side_effect = [
        {"passed": False, "score": 0.3, "reason": "fail"} for _ in range(6)
    ]

    with patch("trading_databricks.agents.graph.get_spark_session"), \
         patch("trading_databricks.agents.graph.load_taxonomy", return_value=_TAXONOMY), \
         patch("trading_databricks.agents.graph.load_whitelist", return_value=_WHITELIST), \
         patch("trading_databricks.agents.graph.mlflow.deployments.get_deploy_client") as mock_val_client, \
         patch("trading_databricks.agents.nodes.context_retriever", side_effect=lambda s, config: {"market_vol_at_publish": 0.02, "market_trend_at_publish": "up", "dollar_bar_close": 3500.0}), \
         patch("trading_databricks.agents.nodes.few_shot_retriever", side_effect=lambda s, config: {"few_shot_examples": [], "retrieval_story_ids": []}), \
         patch("trading_databricks.agents.nodes.uncertainty_router", side_effect=lambda s, config: {"extraction_path": "standard"}):
        mock_val_client.return_value.predict.return_value = {"data": [{"embedding": [0.1] * 8}]}
        graph = build_mas_graph(config=MASConfig())
        result = graph.invoke(_entry_state())

    assert result["extraction_status"] == "judge_failed_accepted"
    assert result["reflection_count"] == 1
    assert hyp_client.predict.call_count == 2
