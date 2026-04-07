"""Tests for agents/graph.py — LangGraph StateGraph assembly."""
from __future__ import annotations

import pytest

from trading_databricks.agents.state import MASState


def test_route_article_fast_path():
    """route_article returns 'causality_extractor' regardless of extraction_path."""
    from trading_databricks.agents.graph import route_article

    state: MASState = {"headline": "test", "extraction_path": "fast"}
    assert route_article(state) == "causality_extractor"


def test_route_article_standard_path():
    from trading_databricks.agents.graph import route_article

    state: MASState = {"headline": "test", "extraction_path": "standard"}
    assert route_article(state) == "causality_extractor"


def test_route_article_careful_path():
    from trading_databricks.agents.graph import route_article

    state: MASState = {"headline": "test", "extraction_path": "careful"}
    assert route_article(state) == "causality_extractor"


def test_route_article_no_path():
    """Missing extraction_path also routes to causality_extractor."""
    from trading_databricks.agents.graph import route_article

    state: MASState = {"headline": "test"}
    assert route_article(state) == "causality_extractor"


from unittest.mock import MagicMock, patch

from trading_databricks.agents.config import MASConfig


def test_mechanism_validator_node_no_extraction():
    """Returns rejected_evidence when state has no extraction."""
    from trading_databricks.agents.graph import _mechanism_validator_node

    state: MASState = {"headline": "test", "body": "some body text"}
    result = _mechanism_validator_node(state, MASConfig())

    assert result["validation_status"] == "rejected_evidence"
    assert result["rejected_mechanism_raw"] is None
    assert result["evidence_relevance_score"] is None


@patch("trading_databricks.agents.graph.mlflow.deployments.get_deploy_client")
@patch("trading_databricks.agents.graph.load_whitelist")
@patch("trading_databricks.agents.graph.get_spark_session")
def test_mechanism_validator_node_accepted(mock_spark, mock_load_whitelist, mock_get_client):
    """Returns accepted when extraction matches whitelist and evidence found in body."""
    from trading_databricks.agents.graph import _mechanism_validator_node

    mock_load_whitelist.return_value = {
        ("Regulation", "institutional_access"): ["bullish"],
    }

    mock_client = MagicMock()
    mock_client.predict.return_value = {"data": [{"embedding": [0.1] * 384}]}
    mock_get_client.return_value = mock_client

    extraction = {
        "treatment_class": "Regulation.etf_ruling",
        "trigger_event": "SEC approves spot Bitcoin ETF",
        "mechanism": "institutional_access",
        "expected_direction": "bullish",
        "evidence_spans": ["sec approves spot bitcoin etf"],
        "confidence": 0.9,
    }
    body = "SEC approves spot Bitcoin ETF, enabling institutional investment."
    state: MASState = {"headline": "SEC approves ETF", "body": body, "extraction": extraction}

    result = _mechanism_validator_node(state, MASConfig())

    assert result["validation_status"] == "accepted"


@patch("trading_databricks.agents.nodes.few_shot_retriever.get_spark_session")
@patch("trading_databricks.agents.nodes.few_shot_retriever.get_accepted_chain_count")
def test_build_mas_graph_compiles(mock_count, mock_spark):
    """build_mas_graph() returns a CompiledStateGraph without raising."""
    from langgraph.graph.state import CompiledStateGraph

    from trading_databricks.agents.graph import build_mas_graph

    graph = build_mas_graph(config=MASConfig())
    assert isinstance(graph, CompiledStateGraph)
