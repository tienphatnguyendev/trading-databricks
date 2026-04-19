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
    assert route_article(state) == "evidence_extractor"


def test_route_article_standard_path():
    from trading_databricks.agents.graph import route_article

    state: MASState = {"headline": "test", "extraction_path": "standard"}
    assert route_article(state) == "evidence_extractor"


def test_route_article_careful_path():
    from trading_databricks.agents.graph import route_article

    state: MASState = {"headline": "test", "extraction_path": "careful"}
    assert route_article(state) == "evidence_extractor"


def test_route_article_no_path():
    """Missing extraction_path also routes to causality_extractor."""
    from trading_databricks.agents.graph import route_article

    state: MASState = {"headline": "test"}
    assert route_article(state) == "evidence_extractor"


def test_mechanism_validator_node_no_extraction():
    """Returns rejected_evidence when state has no extraction."""
    from trading_databricks.agents.graph import mechanism_validator_node

    state: MASState = {"headline": "test", "body": "some body text"}
    result = mechanism_validator_node(state, MASConfig())

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
    from trading_databricks.agents.graph import mechanism_validator_node

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

    result = mechanism_validator_node(state, MASConfig())

    assert result["validation_status"] == "accepted"


@patch("trading_databricks.agents.graph.load_whitelist", return_value={})
@patch("trading_databricks.agents.graph.load_taxonomy", return_value={})
@patch("trading_databricks.agents.graph.get_spark_session")
@patch("trading_databricks.agents.nodes.few_shot_retriever.get_spark_session")
@patch("trading_databricks.agents.nodes.few_shot_retriever.get_accepted_chain_count")
def test_build_mas_graph_compiles(mock_count, mock_fsr_spark, mock_spark, mock_tax, mock_wl):
    """build_mas_graph() returns a CompiledStateGraph without raising."""
    from langgraph.graph.state import CompiledStateGraph

    from trading_databricks.agents.graph import build_mas_graph

    graph = build_mas_graph(config=MASConfig())
    assert isinstance(graph, CompiledStateGraph)


SAMPLE_ARTICLES: list[dict[str, str]] = [
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

_VALID_WHITELIST: dict[tuple[str, str], list[str]] = {
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


@patch("trading_databricks.agents.graph.load_whitelist")
@patch("trading_databricks.agents.graph.load_taxonomy")
@patch("trading_databricks.agents.graph.get_spark_session")
@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose", return_value=[])
@patch("trading_databricks.agents.nodes.rag_retriever._join_market_outcomes", return_value=[])
@patch("trading_databricks.agents.nodes.judge_panel._invoke_judge", return_value={"passed": True, "score": 0.9, "reason": ""})
@patch("trading_databricks.agents.nodes.few_shot_retriever.get_accepted_chain_count")
@patch("trading_databricks.agents.nodes.few_shot_retriever.get_spark_session")
@patch("mlflow.deployments.get_deploy_client")
def test_end_to_end_fast_path(
    mock_deploy_client: MagicMock,
    mock_fsr_spark: MagicMock,
    mock_fsr_count: MagicMock,
    mock_judge: MagicMock,
    mock_join: MagicMock,
    mock_firehose: MagicMock,
    mock_val_spark: MagicMock,
    mock_val_taxonomy: MagicMock,
    mock_val_whitelist: MagicMock,
) -> None:
    """Full graph: single article through the complete 5-node pipeline."""
    from trading_databricks.agents.graph import build_mas_graph

    article = SAMPLE_ARTICLES[0]

    # few_shot_retriever: cold start (count < 50) → returns STATIC_FEW_SHOT_EXAMPLES
    mock_fsr_count.return_value = 0

    # Pre-loaded taxonomy/whitelist at graph construction time
    mock_val_taxonomy.return_value = {"Regulation": ["etf_ruling", "subcategory"]}
    mock_val_whitelist.return_value = _VALID_WHITELIST

    # Single unified client: dispatch on endpoint to avoid double-patch collision.
    # Both causality_extractor and graph.py call mlflow.deployments.get_deploy_client,
    # which resolves to the same underlying function. Patching it once here ensures
    # both callers see the same mock; the side_effect distinguishes LLM vs embed calls.
    llm_resp = _make_llm_response(article)
    embed_resp = {"data": [{"embedding": [0.1] * 384}]}

    config = MASConfig()

    def _predict_dispatch(*, endpoint: str, **_kwargs: dict) -> dict:
        if endpoint == config.llm_endpoint:
            return llm_resp
        return embed_resp

    unified_client = MagicMock()
    unified_client.predict.side_effect = _predict_dispatch
    mock_deploy_client.return_value = unified_client

    graph = build_mas_graph(config=config)
    state = {
        "headline": article["headline"],
        "body": article["body"],
        "published_at": article["published_at"],
    }
    result = graph.invoke(state)

    assert "validation_status" in result
    assert result["validation_status"] == "accepted"
    assert "extraction_path" in result
    assert result["extraction_path"] in ("fast", "standard", "careful")


@patch("trading_databricks.agents.graph.load_whitelist")
@patch("trading_databricks.agents.graph.load_taxonomy")
@patch("trading_databricks.agents.graph.get_spark_session")
@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose", return_value=[])
@patch("trading_databricks.agents.nodes.rag_retriever._join_market_outcomes", return_value=[])
@patch("trading_databricks.agents.nodes.judge_panel._invoke_judge", return_value={"passed": True, "score": "up", "reason": ""})
@patch("trading_databricks.agents.nodes.few_shot_retriever.get_accepted_chain_count")
@patch("trading_databricks.agents.nodes.few_shot_retriever.get_spark_session")
@patch("mlflow.deployments.get_deploy_client")
def test_end_to_end_10_articles(
    mock_deploy_client: MagicMock,
    mock_fsr_spark: MagicMock,
    mock_fsr_count: MagicMock,
    mock_judge: MagicMock,
    mock_join: MagicMock,
    mock_firehose: MagicMock,
    mock_val_spark: MagicMock,
    mock_val_taxonomy: MagicMock,
    mock_val_whitelist: MagicMock,
) -> None:
    """Full graph: all 10 articles produce a final state with validation_status."""
    from trading_databricks.agents.graph import build_mas_graph

    mock_fsr_count.return_value = 0
    mock_val_taxonomy.return_value = {
        "Regulation": ["etf_ruling", "subcategory"],
        "Macro": ["subcategory"],
        "Security": ["subcategory"],
        "Supply": ["subcategory"],
        "Adoption": ["subcategory"],
        "Technology": ["subcategory"],
    }
    mock_val_whitelist.return_value = _VALID_WHITELIST

    embed_resp = {"data": [{"embedding": [0.1] * 384}]}
    config = MASConfig()

    graph = build_mas_graph(config=config)

    for article in SAMPLE_ARTICLES:
        llm_resp = _make_llm_response(article)

        def _predict_dispatch(
            *, endpoint: str, _llm: dict = llm_resp, **_kwargs: dict
        ) -> dict:
            if endpoint == config.llm_endpoint:
                return _llm
            return embed_resp

        unified_client = MagicMock()
        unified_client.predict.side_effect = _predict_dispatch
        # Reassign before each invoke — each graph.invoke() calls get_deploy_client()
        # fresh at node execution time, so updating return_value here is picked up next.
        mock_deploy_client.return_value = unified_client

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
        assert "extraction_path" in result, (
            f"Missing extraction_path for: {article['headline']}"
        )

@patch("trading_databricks.agents.graph.load_whitelist", return_value={})
@patch("trading_databricks.agents.graph.load_taxonomy", return_value={})
@patch("trading_databricks.agents.graph.get_spark_session")
def test_build_mas_graph_includes_subpipeline_nodes(mock_spark, mock_tax, mock_wl):
    from trading_databricks.agents.config import MASConfig
    from trading_databricks.agents.graph import build_mas_graph

    graph = build_mas_graph(config=MASConfig())
    node_names = set(graph.get_graph().nodes.keys())

    expected = {
        "context_retriever",
        "few_shot_retriever",
        "uncertainty_router",
        "evidence_extractor",
        "rag_retriever",
        "hypothesis_generator",
        "judge_panel",
        "reflector",
        "mechanism_validator",
    }
    assert expected.issubset(node_names), f"missing nodes: {expected - node_names}"
    # The monolithic causality_extractor should no longer be a graph node
    assert "causality_extractor" not in node_names
