"""LangGraph StateGraph assembly for the MAS pipeline.

Outer topology:
    context_retriever → few_shot_retriever → uncertainty_router
    → [conditional: route_article] → evidence_extractor → ...

Inner causality sub-pipeline (replaces the old monolithic causality_extractor):
    evidence_extractor
    → rag_retriever
    → hypothesis_generator
    → judge_panel
    → reflector
        → (retry) hypothesis_generator
        → (proceed) mechanism_validator → END

All transform logic lives in nodes/. This file owns topology only.
"""
from __future__ import annotations

import os
from typing import Literal

import mlflow.deployments

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.nodes import (
    REFLECTOR_ROUTE_RETRY,
    REFLECTOR_ROUTE_VALIDATOR,
)
from trading_databricks.agents.nodes.few_shot_retriever import get_spark_session
from trading_databricks.agents.nodes.mechanism_validator import validate_extraction
from trading_databricks.agents.state import MASState
from trading_databricks.agents.taxonomy import load_whitelist


def route_article(state: MASState) -> Literal["evidence_extractor"]:
    """Route from uncertainty_router into the causality sub-pipeline.

    All extraction paths (fast/standard/careful) enter at evidence_extractor.
    Per-path behavior (e.g., temperature, prompt depth) is now owned by
    individual sub-pipeline nodes via state.extraction_path.
    """
    return "evidence_extractor"


def route_after_reflector(state: MASState) -> str:
    """Read the `_route` key set by the reflector node and dispatch."""
    return state.get("_route", REFLECTOR_ROUTE_VALIDATOR)


def mechanism_validator_node(state: MASState, config: MASConfig) -> dict:
    """Bridge validate_extraction() to the LangGraph (state) -> dict interface."""
    extraction = state.get("extraction")
    if not extraction:
        return {
            "validation_status": "rejected_evidence",
            "rejected_mechanism_raw": None,
            "evidence_relevance_score": None,
        }

    spark = get_spark_session()
    whitelist = load_whitelist(spark, catalog=config.catalog, schema=config.schema)

    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE", os.environ.get("DATABRICKS_PROFILE", ""))
    target_uri = f"databricks://{profile}" if profile else "databricks"
    client = mlflow.deployments.get_deploy_client(target_uri)

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
    """Compile the MAS StateGraph with the 5-node causality sub-pipeline."""
    from langgraph.graph import StateGraph

    from trading_databricks.agents.nodes import (
        context_retriever,
        evidence_extractor,
        few_shot_retriever,
        hypothesis_generator,
        judge_panel,
        rag_retriever,
        reflector,
        uncertainty_router,
    )

    graph = StateGraph(MASState)

    # Outer nodes (unchanged)
    graph.add_node("context_retriever",   lambda s: context_retriever(s, config=config))
    graph.add_node("few_shot_retriever",  lambda s: few_shot_retriever(s, config=config))
    graph.add_node("uncertainty_router",  lambda s: uncertainty_router(s, config=config))

    # Sub-pipeline nodes (new)
    graph.add_node("evidence_extractor",   lambda s: evidence_extractor(s, config=config))
    graph.add_node("rag_retriever",        lambda s: rag_retriever(s, config=config))
    graph.add_node("hypothesis_generator", lambda s: hypothesis_generator(s, config=config))
    graph.add_node("judge_panel",          lambda s: judge_panel(s, config=config))
    graph.add_node("reflector",            lambda s: reflector(s, config=config))

    # Validator (unchanged)
    graph.add_node("mechanism_validator",  lambda s: mechanism_validator_node(s, config))

    # Outer edges
    graph.set_entry_point("context_retriever")
    graph.add_edge("context_retriever",   "few_shot_retriever")
    graph.add_edge("few_shot_retriever",  "uncertainty_router")
    graph.add_conditional_edges(
        "uncertainty_router",
        route_article,
        {"evidence_extractor": "evidence_extractor"},
    )

    # Sub-pipeline edges
    graph.add_edge("evidence_extractor",   "rag_retriever")
    graph.add_edge("rag_retriever",        "hypothesis_generator")
    graph.add_edge("hypothesis_generator", "judge_panel")
    graph.add_edge("judge_panel",          "reflector")
    graph.add_conditional_edges(
        "reflector",
        route_after_reflector,
        {
            REFLECTOR_ROUTE_RETRY:     "hypothesis_generator",
            REFLECTOR_ROUTE_VALIDATOR: "mechanism_validator",
        },
    )

    graph.set_finish_point("mechanism_validator")

    return graph.compile()
