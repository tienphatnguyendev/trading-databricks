"""LangGraph StateGraph assembly for the MAS pipeline.

Wires all 5 nodes into a CompiledStateGraph using closure-bound config.
All transform logic lives in nodes/. This file owns topology only.
"""
from __future__ import annotations

from typing import Literal

import mlflow.deployments

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.nodes.few_shot_retriever import get_spark_session
from trading_databricks.agents.nodes.mechanism_validator import validate_extraction
from trading_databricks.agents.state import MASState
from trading_databricks.agents.taxonomy import load_whitelist


def route_article(state: MASState) -> Literal["causality_extractor"]:
    """Route from uncertainty_router to the next node.

    V1: all extraction paths (fast/standard/careful) proceed to causality_extractor.
    The extraction_path field is already set in state; causality_extractor reads it
    internally to select temperature/n_choices. The conditional edge is kept explicit
    here for future extensibility (e.g., a trivial path short-circuiting to END).
    """
    return "causality_extractor"


def _mechanism_validator_node(state: MASState, config: MASConfig) -> dict:
    """Bridge validate_extraction() to the LangGraph (state) -> dict interface.

    validate_extraction() has a 5-arg signature incompatible with LangGraph nodes.
    This private helper closes over config and injects whitelist + embed_fn at call time.

    Returns dict with keys: validation_status, rejected_mechanism_raw, evidence_relevance_score.
    """
    extraction = state.get("extraction")
    if not extraction:
        return {
            "validation_status": "rejected_evidence",
            "rejected_mechanism_raw": None,
            "evidence_relevance_score": None,
        }

    spark = get_spark_session()
    whitelist = load_whitelist(spark, catalog=config.catalog, schema=config.schema)

    client = mlflow.deployments.get_deploy_client("databricks")

    def embed_fn(text: str) -> list[float]:
        resp = client.predict(
            endpoint=config.validator.embedding_model,
            inputs={"input": [text]},
        )
        return resp["data"][0]["embedding"]

    result = validate_extraction(
        extraction,
        state.get("body", ""),
        whitelist,
        config=config.validator,
        embed_fn=embed_fn,
    )
    return dict(result)


def build_mas_graph(*, config: MASConfig):
    """Compile the 5-node MAS StateGraph.

    Node execution order:
        context_retriever → few_shot_retriever → uncertainty_router
        → [conditional: route_article] → causality_extractor
        → mechanism_validator → END

    All nodes receive MASConfig via lambda closure — zero changes to node files.

    Args:
        config: MASConfig containing nested configs for all nodes.

    Returns:
        CompiledStateGraph ready for .invoke() calls and MLflow logging.
    """
    from langgraph.graph import StateGraph

    from trading_databricks.agents.nodes import (
        causality_extractor,
        context_retriever,
        few_shot_retriever,
        uncertainty_router,
    )

    graph = StateGraph(MASState)

    graph.add_node("context_retriever",   lambda s: context_retriever(s, config=config))
    graph.add_node("few_shot_retriever",  lambda s: few_shot_retriever(s, config=config))
    graph.add_node("uncertainty_router",  lambda s: uncertainty_router(s, config=config))
    graph.add_node("causality_extractor", lambda s: causality_extractor(s, config=config))
    graph.add_node("mechanism_validator", lambda s: _mechanism_validator_node(s, config))

    graph.set_entry_point("context_retriever")
    graph.add_edge("context_retriever",   "few_shot_retriever")
    graph.add_edge("few_shot_retriever",  "uncertainty_router")
    graph.add_conditional_edges(
        "uncertainty_router",
        route_article,
        {"causality_extractor": "causality_extractor"},
    )
    graph.add_edge("causality_extractor", "mechanism_validator")
    graph.set_finish_point("mechanism_validator")

    return graph.compile()
