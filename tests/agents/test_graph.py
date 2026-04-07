"""Tests for agents/graph.py — LangGraph StateGraph assembly."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from trading_databricks.agents.config import MASConfig
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
def test_mechanism_validator_node_accepted(
    mock_spark, mock_load_whitelist, mock_get_client
):
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
    state: MASState = {
        "headline": "SEC approves ETF",
        "body": body,
        "extraction": extraction,
    }

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


SAMPLE_ARTICLES = [
    {
        "headline": "SEC approves first spot Bitcoin ETF applications",
        "body": (
            "The SEC approves spot Bitcoin ETF applications from major asset"
            " managers, enabling regulated institutional access."
        ),
        "published_at": "2026-01-15T10:00:00Z",
        "expected_treatment_l1": "Regulation",
        "expected_mechanism": "institutional_access",
        "expected_direction": "bullish",
    },
    {
        "headline": "Federal Reserve raises interest rates by 75 basis points",
        "body": (
            "The Federal Reserve raised interest rates by 75 basis points to"
            " combat inflation, tightening monetary policy."
        ),
        "published_at": "2026-01-16T14:00:00Z",
        "expected_treatment_l1": "Macro",
        "expected_mechanism": "liquidity_contraction",
        "expected_direction": "bearish",
    },
    {
        "headline": "Major cryptocurrency exchange confirms $480M hack",
        "body": (
            "A major cryptocurrency exchange confirmed a $480M hack and"
            " suspended withdrawals, destroying user confidence."
        ),
        "published_at": "2026-01-17T08:00:00Z",
        "expected_treatment_l1": "Security",
        "expected_mechanism": "trust_erosion",
        "expected_direction": "bearish",
    },
    {
        "headline": "Mt. Gox trustee confirms July BTC distribution",
        "body": (
            "Mt. Gox trustee confirms July 2026 BTC distribution to creditors,"
            " adding sell pressure to Bitcoin markets."
        ),
        "published_at": "2026-01-18T09:00:00Z",
        "expected_treatment_l1": "Supply",
        "expected_mechanism": "sell_pressure",
        "expected_direction": "bearish",
    },
    {
        "headline": "BlackRock announces Bitcoin treasury allocation",
        "body": (
            "BlackRock announced a Bitcoin treasury allocation strategy,"
            " increasing institutional demand for BTC."
        ),
        "published_at": "2026-01-19T11:00:00Z",
        "expected_treatment_l1": "Adoption",
        "expected_mechanism": "demand_increase",
        "expected_direction": "bullish",
    },
    {
        "headline": "Ethereum network completes major protocol upgrade",
        "body": (
            "The Ethereum network completed a major protocol upgrade improving"
            " scalability, increasing adoption potential."
        ),
        "published_at": "2026-01-20T15:00:00Z",
        "expected_treatment_l1": "Technology",
        "expected_mechanism": "scalability_improvement",
        "expected_direction": "bullish",
    },
    {
        "headline": "China bans crypto mining operations",
        "body": (
            "China announced a ban on all cryptocurrency mining operations,"
            " restricting crypto activity in the region."
        ),
        "published_at": "2026-01-21T06:00:00Z",
        "expected_treatment_l1": "Regulation",
        "expected_mechanism": "regulatory_restriction",
        "expected_direction": "bearish",
    },
    {
        "headline": "Bitcoin ETF sees record $2B inflow",
        "body": (
            "Bitcoin ETF saw a record $2B single-day inflow, signaling strong"
            " institutional demand."
        ),
        "published_at": "2026-01-22T12:00:00Z",
        "expected_treatment_l1": "Adoption",
        "expected_mechanism": "demand_increase",
        "expected_direction": "bullish",
    },
    {
        "headline": "Stablecoin issuer Tether freezes $100M in wallets",
        "body": (
            "Tether froze $100M in wallets linked to sanctioned entities,"
            " raising concerns about crypto censorship."
        ),
        "published_at": "2026-01-23T07:00:00Z",
        "expected_treatment_l1": "Regulation",
        "expected_mechanism": "regulatory_restriction",
        "expected_direction": "bearish",
    },
    {
        "headline": "Coinbase obtains MiCA license for European operations",
        "body": (
            "Coinbase obtained full MiCA regulatory license enabling compliant"
            " crypto trading across the EU."
        ),
        "published_at": "2026-01-24T10:00:00Z",
        "expected_treatment_l1": "Regulation",
        "expected_mechanism": "regulatory_clarity",
        "expected_direction": "bullish",
    },
]

_VALID_WHITELIST = {
    ("Regulation", "institutional_access"): ["bullish"],
    ("Regulation", "regulatory_clarity"): ["bullish"],
    ("Regulation", "regulatory_restriction"): ["bearish"],
    ("Macro", "liquidity_contraction"): ["bearish"],
    ("Security", "trust_erosion"): ["bearish"],
    ("Supply", "sell_pressure"): ["bearish"],
    ("Adoption", "demand_increase"): ["bullish"],
    ("Technology", "scalability_improvement"): ["bullish"],
}


def _make_llm_response(article: dict) -> dict:
    """Return a mock LLM response dict for a sample article."""
    body_snippet = article["body"][:80].lower()
    l1 = article["expected_treatment_l1"]
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "treatment_class": f"{l1}.subcategory",
                            "trigger_event": article["headline"][:100],
                            "mechanism": article["expected_mechanism"],
                            "expected_direction": article["expected_direction"],
                            "confidence": 0.9,
                            "asset": "BTC",
                            "evidence_spans": [body_snippet[:50]],
                            "reasoning": "Extracted from article.",
                        }
                    )
                }
            }
        ]
    }


@patch("trading_databricks.agents.graph.mlflow.deployments.get_deploy_client")
@patch("trading_databricks.agents.graph.load_whitelist")
@patch("trading_databricks.agents.graph.get_spark_session")
@patch(
    "trading_databricks.agents.nodes.causality_extractor.mlflow.deployments.get_deploy_client"
)
@patch("trading_databricks.agents.nodes.causality_extractor.get_spark_session")
@patch("trading_databricks.agents.nodes.causality_extractor.load_taxonomy")
@patch("trading_databricks.agents.nodes.causality_extractor.load_whitelist")
@patch("trading_databricks.agents.nodes.few_shot_retriever.get_accepted_chain_count")
@patch("trading_databricks.agents.nodes.few_shot_retriever.get_spark_session")
def test_end_to_end_fast_path(
    mock_fsr_spark: MagicMock,
    mock_fsr_count: MagicMock,
    mock_ext_whitelist: MagicMock,
    mock_ext_taxonomy: MagicMock,
    mock_ext_spark: MagicMock,
    mock_ext_client: MagicMock,
    mock_val_spark: MagicMock,
    mock_val_whitelist: MagicMock,
    mock_val_client: MagicMock,
) -> None:
    """Full graph: single article through the complete 5-node pipeline."""
    from trading_databricks.agents.graph import build_mas_graph

    article = SAMPLE_ARTICLES[0]

    # few_shot_retriever: cold start (count < 50) → returns STATIC_FEW_SHOT_EXAMPLES
    mock_fsr_count.return_value = 0

    # causality_extractor mocks
    mock_ext_taxonomy.return_value = {"Regulation": ["etf_ruling", "subcategory"]}
    mock_ext_whitelist.return_value = _VALID_WHITELIST
    mock_llm_client = MagicMock()
    mock_llm_client.predict.return_value = _make_llm_response(article)
    mock_ext_client.return_value = mock_llm_client

    # mechanism_validator mocks (embed_fn via graph.py's mlflow client)
    mock_val_whitelist.return_value = _VALID_WHITELIST
    mock_embed_client = MagicMock()
    mock_embed_client.predict.return_value = {"data": [{"embedding": [0.1] * 384}]}
    mock_val_client.return_value = mock_embed_client

    graph = build_mas_graph(config=MASConfig())
    state = {
        "headline": article["headline"],
        "body": article["body"],
        "published_at": article["published_at"],
    }
    result = graph.invoke(state)

    assert "validation_status" in result
    assert result["validation_status"] in (
        "accepted",
        "rejected_mechanism",
        "rejected_direction",
        "rejected_evidence",
    )
    assert "extraction_path" in result
    assert result["extraction_path"] in ("fast", "standard", "careful")


@patch("trading_databricks.agents.graph.mlflow.deployments.get_deploy_client")
@patch("trading_databricks.agents.graph.load_whitelist")
@patch("trading_databricks.agents.graph.get_spark_session")
@patch(
    "trading_databricks.agents.nodes.causality_extractor.mlflow.deployments.get_deploy_client"
)
@patch("trading_databricks.agents.nodes.causality_extractor.get_spark_session")
@patch("trading_databricks.agents.nodes.causality_extractor.load_taxonomy")
@patch("trading_databricks.agents.nodes.causality_extractor.load_whitelist")
@patch("trading_databricks.agents.nodes.few_shot_retriever.get_accepted_chain_count")
@patch("trading_databricks.agents.nodes.few_shot_retriever.get_spark_session")
def test_end_to_end_10_articles(
    mock_fsr_spark: MagicMock,
    mock_fsr_count: MagicMock,
    mock_ext_whitelist: MagicMock,
    mock_ext_taxonomy: MagicMock,
    mock_ext_spark: MagicMock,
    mock_ext_client: MagicMock,
    mock_val_spark: MagicMock,
    mock_val_whitelist: MagicMock,
    mock_val_client: MagicMock,
) -> None:
    """Full graph: all 10 articles produce a final state with validation_status."""
    from trading_databricks.agents.graph import build_mas_graph

    mock_fsr_count.return_value = 0
    mock_ext_taxonomy.return_value = {
        "Regulation": ["etf_ruling", "subcategory"],
        "Macro": ["subcategory"],
        "Security": ["subcategory"],
        "Supply": ["subcategory"],
        "Adoption": ["subcategory"],
        "Technology": ["subcategory"],
    }
    mock_ext_whitelist.return_value = _VALID_WHITELIST
    mock_val_whitelist.return_value = _VALID_WHITELIST

    mock_embed_client = MagicMock()
    mock_embed_client.predict.return_value = {"data": [{"embedding": [0.1] * 384}]}
    mock_val_client.return_value = mock_embed_client

    graph = build_mas_graph(config=MASConfig())

    for article in SAMPLE_ARTICLES:
        mock_llm_client = MagicMock()
        mock_llm_client.predict.return_value = _make_llm_response(article)
        mock_ext_client.return_value = mock_llm_client

        state = {
            "headline": article["headline"],
            "body": article["body"],
            "published_at": article["published_at"],
        }
        result = graph.invoke(state)

        assert "validation_status" in result, (
            f"Missing validation_status for: {article['headline']}"
        )
        assert result["validation_status"] in (
            "accepted",
            "rejected_mechanism",
            "rejected_direction",
            "rejected_evidence",
        ), f"Invalid status '{result['validation_status']}' for: {article['headline']}"
        assert "extraction" in result, f"Missing extraction for: {article['headline']}"
