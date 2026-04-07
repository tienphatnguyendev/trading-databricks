"""Tests for the causality_extractor node."""

import pytest

from trading_databricks.agents.nodes.causality_extractor import (
    majority_vote,
    parse_json_response,
)


def test_parse_json_response():
    assert parse_json_response('{"a": 1}') == {"a": 1}
    assert parse_json_response('```json\n{"b": 2}\n```') == {"b": 2}
    assert parse_json_response("invalid") is None


def test_majority_vote_consensus():
    extractions = [
        {
            "treatment_class": "A",
            "mechanism": "m",
            "expected_direction": "bullish",
            "confidence": 0.8,
        },
        {
            "treatment_class": "A",
            "mechanism": "m",
            "expected_direction": "bullish",
            "confidence": 0.7,
        },
        {
            "treatment_class": "B",
            "mechanism": "n",
            "expected_direction": "bearish",
            "confidence": 0.9,
        },
    ]
    winner = majority_vote(extractions)
    assert winner["confidence"] == 0.8  # First occurrence of consensus


def test_majority_vote_tie_breaker():
    extractions = [
        {
            "treatment_class": "A",
            "mechanism": "m",
            "expected_direction": "bullish",
            "confidence": 0.6,
        },
        {
            "treatment_class": "B",
            "mechanism": "n",
            "expected_direction": "bearish",
            "confidence": 0.9,
        },
        {
            "treatment_class": "C",
            "mechanism": "o",
            "expected_direction": "volatile",
            "confidence": 0.7,
        },
    ]
    winner = majority_vote(extractions)
    assert winner["treatment_class"] == "B"
    assert winner["confidence"] == pytest.approx(0.6)  # Penalized: max(0, 0.9 - 0.3)


from unittest.mock import MagicMock, patch

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.nodes.causality_extractor import causality_extractor


@patch(
    "trading_databricks.agents.nodes.causality_extractor.mlflow.deployments.get_deploy_client"
)
@patch("trading_databricks.agents.nodes.causality_extractor.get_spark_session")
@patch("trading_databricks.agents.nodes.causality_extractor.load_taxonomy")
@patch("trading_databricks.agents.nodes.causality_extractor.load_whitelist")
def test_causality_extractor_fast_path(
    mock_whitelist, mock_taxonomy, mock_spark, mock_get_client
):
    mock_taxonomy.return_value = {"A": ["b"]}
    mock_whitelist.return_value = {("A", "m"): ["bullish"]}

    mock_client = MagicMock()
    mock_client.predict.return_value = {
        "choices": [{"message": {"content": '{"treatment_class": "A.b"}'}}]
    }
    mock_get_client.return_value = mock_client

    state = {
        "headline": "Test",
        "body": "Text",
        "extraction_path": "fast",
        "few_shot_examples": [],
    }
    config = MASConfig()

    new_state = causality_extractor(state, config)

    assert new_state["extraction"] == {"treatment_class": "A.b"}

    call_kwargs = mock_client.predict.call_args[1]
    assert call_kwargs["inputs"]["n"] == 1
    assert call_kwargs["inputs"]["temperature"] == 0.0


@patch(
    "trading_databricks.agents.nodes.causality_extractor.mlflow.deployments.get_deploy_client"
)
@patch("trading_databricks.agents.nodes.causality_extractor.get_spark_session")
@patch("trading_databricks.agents.nodes.causality_extractor.load_taxonomy")
@patch("trading_databricks.agents.nodes.causality_extractor.load_whitelist")
def test_causality_extractor_careful_path(
    mock_whitelist, mock_taxonomy, mock_spark, mock_get_client
):
    mock_taxonomy.return_value = {"A": ["b"]}
    mock_whitelist.return_value = {("A", "m"): ["bullish"]}

    mock_client = MagicMock()
    mock_client.predict.return_value = {
        "choices": [
            {"message": {"content": '{"treatment_class": "A.b", "confidence": 0.8}'}},
            {"message": {"content": '{"treatment_class": "A.b", "confidence": 0.9}'}},
            {"message": {"content": '{"treatment_class": "C.d", "confidence": 0.7}'}},
        ]
    }
    mock_get_client.return_value = mock_client

    state = {
        "headline": "Test",
        "body": "Text",
        "extraction_path": "careful",
        "few_shot_examples": [],
    }
    config = MASConfig()

    new_state = causality_extractor(state, config)

    assert new_state["extraction"]["treatment_class"] == "A.b"
    assert new_state["extraction"]["confidence"] == 0.8

    call_kwargs = mock_client.predict.call_args[1]
    assert call_kwargs["inputs"]["n"] == 3
    assert call_kwargs["inputs"]["temperature"] == 0.7
