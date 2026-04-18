"""End-to-end integration test for the causality sub-pipeline.

Runs build_mas_graph() with mocked LLM and Vector Search externals and
verifies the happy path and the reflection path produce the expected
terminal state.
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

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


def _stub_outer_nodes():
    """Patch the outer nodes so the integration test focuses on the sub-pipeline."""
    return [
        patch("trading_databricks.agents.graph.get_spark_session"),
        patch("trading_databricks.agents.graph.load_whitelist"),
        patch("trading_databricks.agents.nodes.context_retriever.context_retriever", side_effect=lambda s, config: {"market_vol_at_publish": 0.02, "market_trend_at_publish": "up", "dollar_bar_close": 3500.0}),
        patch("trading_databricks.agents.nodes.few_shot_retriever.few_shot_retriever", side_effect=lambda s, config: {"few_shot_examples": [], "retrieval_story_ids": []}),
        patch("trading_databricks.agents.nodes.uncertainty_router.uncertainty_router", side_effect=lambda s, config: {"extraction_path": "standard"}),
    ]


def _entry_state():
    return {
        "headline": "BlackRock files for spot Ether ETF",
        "body": "BlackRock files for spot Ether ETF with the SEC.",
        "published_at": "2026-04-18T10:00:00Z",
        "symbol": "ETHUSDT",
    }


@patch("trading_databricks.agents.nodes.judge_panel._invoke_judge")
@patch("trading_databricks.agents.nodes.hypothesis_generator.load_taxonomy")
@patch("trading_databricks.agents.nodes.hypothesis_generator.load_whitelist")
@patch("trading_databricks.agents.nodes.hypothesis_generator.get_spark_session")
@patch("mlflow.deployments.get_deploy_client")
@patch("trading_databricks.agents.nodes.rag_retriever._join_market_outcomes")
@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose")
def test_subpipeline_happy_path(
    mock_firehose, mock_join, mock_deploy_client,
    mock_spark, mock_wl, mock_tax, mock_judge,
):
    from trading_databricks.agents.graph import build_mas_graph

    def _predict(endpoint, inputs, **kwargs):
        msgs = inputs.get("messages", [])
        content = " ".join(m.get("content", "") for m in msgs).lower()
        if "evidence" in content and "actor" in content and "treatment_class" not in content:
            # Looks like evidence extractor prompt
            return {"choices": [{"message": {"content": '[{"actor": "BlackRock", "action": "files", "object": "spot Ether ETF", "span": "BlackRock files for spot Ether ETF", "timestamp_hint": null}]'}}]}
        elif "treatment_class" in content:
            # Looks like hypothesis generator prompt
            return {"choices": [{"message": {"content": json.dumps(VALID_EXTRACTION)}}]}
        return {"data": [{"embedding": [0.1] * 8}]}

    unified_client = MagicMock()
    unified_client.predict.side_effect = _predict
    mock_deploy_client.return_value = unified_client

    # Stub RAG
    mock_firehose.return_value = [{"headline": "Fidelity files Bitcoin ETF", "timestamp": "2024-06-15T12:00:00Z"}]
    mock_join.return_value = [{"headline": "Fidelity files Bitcoin ETF", "timestamp": "2024-06-15T12:00:00Z", "return_24h": 0.043, "direction": "up"}]

    mock_tax.return_value = {"Regulation": ["etf_ruling"]}
    mock_wl.return_value = {("Regulation", "etf_inflow_demand"): ["bullish"]}

    # Stub judges — all pass
    mock_judge.side_effect = [
        {"passed": True, "score": "up", "reason": ""},
        {"passed": True, "score": "match", "reason": ""},
        {"passed": True, "score": 0.9, "reason": ""},
    ]

    # Stub the validator's external resources via graph helpers
    with patch("trading_databricks.agents.graph.get_spark_session"), \
         patch("trading_databricks.agents.graph.load_whitelist", return_value={("Regulation", "etf_inflow_demand"): ["bullish"]}), \
         patch("trading_databricks.agents.nodes.context_retriever", side_effect=lambda s, config: {"market_vol_at_publish": 0.02, "market_trend_at_publish": "up", "dollar_bar_close": 3500.0}), \
         patch("trading_databricks.agents.nodes.few_shot_retriever", side_effect=lambda s, config: {"few_shot_examples": [], "retrieval_story_ids": []}), \
         patch("trading_databricks.agents.nodes.uncertainty_router", side_effect=lambda s, config: {"extraction_path": "standard"}):
        graph = build_mas_graph(config=MASConfig())
        result = graph.invoke(_entry_state())

    assert result["extraction"]["mechanism"] == "etf_inflow_demand"
    assert result["extraction_status"] == "ok"
    assert result["reflection_count"] == 0
    assert result["validation_status"] == "accepted"


@patch("trading_databricks.agents.nodes.judge_panel._invoke_judge")
@patch("trading_databricks.agents.nodes.hypothesis_generator.load_taxonomy")
@patch("trading_databricks.agents.nodes.hypothesis_generator.load_whitelist")
@patch("trading_databricks.agents.nodes.hypothesis_generator.get_spark_session")
@patch("mlflow.deployments.get_deploy_client")
@patch("trading_databricks.agents.nodes.rag_retriever._join_market_outcomes")
@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose")
def test_subpipeline_reflects_then_passes(
    mock_firehose, mock_join, mock_deploy_client,
    mock_spark, mock_wl, mock_tax, mock_judge,
):
    from trading_databricks.agents.graph import build_mas_graph

    def _predict(endpoint, inputs, **kwargs):
        if "input" in inputs:
            # Embedding call from validator
            return {"data": [{"embedding": [0.1] * 8}]}

        msgs = inputs.get("messages", [])
        content = " ".join(m.get("content", "") for m in msgs).lower()
        if "evidence" in content and "actor" in content and "treatment_class" not in content:
            return {"choices": [{"message": {"content": '[{"actor": "BlackRock", "action": "files", "span": "BlackRock files for spot Ether ETF"}]'}}]}
        elif "treatment_class" in content:
            return {"choices": [{"message": {"content": json.dumps(VALID_EXTRACTION)}}]}
        
        # Fallback
        return {"choices": [{"message": {"content": "{}"}}]}

    unified_client = MagicMock()
    unified_client.predict.side_effect = _predict
    mock_deploy_client.return_value = unified_client

    mock_firehose.return_value = [{"headline": "x", "timestamp": "t"}]
    mock_join.return_value = [{"headline": "x", "timestamp": "t", "return_24h": 0.04, "direction": "up"}]

    mock_tax.return_value = {"Regulation": ["etf_ruling"]}
    mock_wl.return_value = {("Regulation", "etf_inflow_demand"): ["bullish"]}

    # Judges fail first call, pass second
    mock_judge.side_effect = [
        # First panel — grounding fails
        {"passed": True, "score": "up", "reason": ""},
        {"passed": True, "score": "match", "reason": ""},
        {"passed": False, "score": 0.3, "reason": "low score"},
        # Second panel — all pass
        {"passed": True, "score": "up", "reason": ""},
        {"passed": True, "score": "match", "reason": ""},
        {"passed": True, "score": 0.9, "reason": ""},
    ]

    with patch("trading_databricks.agents.graph.get_spark_session"), \
         patch("trading_databricks.agents.graph.load_whitelist", return_value={("Regulation", "etf_inflow_demand"): ["bullish"]}), \
         patch("trading_databricks.agents.nodes.context_retriever", side_effect=lambda s, config: {"market_vol_at_publish": 0.02, "market_trend_at_publish": "up", "dollar_bar_close": 3500.0}), \
         patch("trading_databricks.agents.nodes.few_shot_retriever", side_effect=lambda s, config: {"few_shot_examples": [], "retrieval_story_ids": []}), \
         patch("trading_databricks.agents.nodes.uncertainty_router", side_effect=lambda s, config: {"extraction_path": "standard"}):
        graph = build_mas_graph(config=MASConfig())
        result = graph.invoke(_entry_state())

    assert result["extraction_status"] == "ok_after_reflection"
    assert result["reflection_count"] == 1


@patch("trading_databricks.agents.nodes.judge_panel._invoke_judge")
@patch("trading_databricks.agents.nodes.hypothesis_generator.load_taxonomy")
@patch("trading_databricks.agents.nodes.hypothesis_generator.load_whitelist")
@patch("trading_databricks.agents.nodes.hypothesis_generator.get_spark_session")
@patch("mlflow.deployments.get_deploy_client")
@patch("trading_databricks.agents.nodes.rag_retriever._join_market_outcomes")
@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose")
def test_subpipeline_budget_exhausted(
    mock_firehose, mock_join, mock_deploy_client,
    mock_spark, mock_wl, mock_tax, mock_judge,
):
    from trading_databricks.agents.graph import build_mas_graph

    call_counts = {"hyp": 0}

    def _predict(endpoint, inputs, **kwargs):
        if "input" in inputs:
            return {"data": [{"embedding": [0.1] * 8}]}
        msgs = inputs.get("messages", [])
        content = " ".join(m.get("content", "") for m in msgs).lower()
        if "evidence" in content and "actor" in content and "treatment_class" not in content:
            return {"choices": [{"message": {"content": "[]"}}]}
        elif "treatment_class" in content:
            call_counts["hyp"] += 1
            return {"choices": [{"message": {"content": json.dumps(VALID_EXTRACTION)}}]}
        
        return {"choices": [{"message": {"content": "{}"}}]}

    unified_client = MagicMock()
    unified_client.predict.side_effect = _predict
    mock_deploy_client.return_value = unified_client

    mock_firehose.return_value = []
    mock_join.return_value = []

    mock_tax.return_value = {"Regulation": ["etf_ruling"]}
    mock_wl.return_value = {("Regulation", "etf_inflow_demand"): ["bullish"]}

    # Judges always fail
    mock_judge.side_effect = [
        {"passed": False, "score": 0.3, "reason": "fail"} for _ in range(6)
    ]

    with patch("trading_databricks.agents.graph.get_spark_session"), \
         patch("trading_databricks.agents.graph.load_whitelist", return_value={("Regulation", "etf_inflow_demand"): ["bullish"]}), \
         patch("trading_databricks.agents.nodes.context_retriever", side_effect=lambda s, config: {"market_vol_at_publish": 0.02, "market_trend_at_publish": "up", "dollar_bar_close": 3500.0}), \
         patch("trading_databricks.agents.nodes.few_shot_retriever", side_effect=lambda s, config: {"few_shot_examples": [], "retrieval_story_ids": []}), \
         patch("trading_databricks.agents.nodes.uncertainty_router", side_effect=lambda s, config: {"extraction_path": "standard"}):
        graph = build_mas_graph(config=MASConfig())
        result = graph.invoke(_entry_state())

    assert result["extraction_status"] == "judge_failed_accepted"
    assert result["reflection_count"] == 1  # exactly max_reflections
    assert call_counts["hyp"] == 2
