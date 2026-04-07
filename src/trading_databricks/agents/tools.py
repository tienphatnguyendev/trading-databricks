"""Unity Catalog function tools and retrieval logic for gen-AI agents.

These tools are registered as UC functions and called by Databricks Agent
Framework agents to query the gold layer.
"""
from __future__ import annotations

import json

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.nodes.few_shot_retriever import (
    STATIC_FEW_SHOT_EXAMPLES,
    get_accepted_chain_count,
    get_spark_session,
    merge_examples,
    query_vector_search,
)


def retrieve_few_shot_chains(
    headline: str,
    *,
    config: MASConfig | None = None,
) -> list[dict]:
    """Retrieve few-shot causal chain examples for a news headline.

    This is a standalone tool wrapper around the few-shot retriever node,
    providing a simpler interface for testing and non-LangGraph usage.

    Args:
        headline: Article headline for semantic search
        config: Optional MASConfig (uses default if not provided)

    Returns:
        List of 0-5 few-shot examples (dict format matching VS results)

    Cold-start behavior:
        Returns static examples only if accepted chain count < 50

    Warm-start behavior:
        Queries Vector Search, merges with static examples

    Example:
        >>> examples = retrieve_few_shot_chains("Fed raises interest rates")
        >>> len(examples)
        5
        >>> examples[0]["treatment_class"]
        'Macro.rate_decision'
    """
    if config is None:
        config = MASConfig()

    # Query accepted chain count (cached)
    spark = get_spark_session()
    accepted_count = get_accepted_chain_count(
        spark,
        catalog=config.catalog,
        schema=config.schema,
    )

    # Cold-start guard
    if accepted_count < config.router.min_index_size:
        return STATIC_FEW_SHOT_EXAMPLES

    # Warm path: query VS and merge
    dynamic_results = query_vector_search(
        headline,
        index_name=config.vs_index_name,
        num_results=5,
    )

    return merge_examples(
        dynamic_results,
        STATIC_FEW_SHOT_EXAMPLES,
        max_total=5,
    )


def fetch_mechanism_whitelist(*, config: MASConfig | None = None) -> str:
    """Read the mechanism whitelist from gold.mechanism_whitelist as a JSON array.

    This is a UC function tool registered in Mosaic AI Agent Framework.
    Returns the full whitelist so the LLM can reference valid (treatment, mechanism)
    pairs and their allowed directions.

    Args:
        config: MASConfig — catalog/schema determine the table path.
                Uses default MASConfig() if not provided.

    Returns:
        JSON array string: [{treatment_l1, mechanism, valid_directions, description}, ...]

    Example:
        >>> whitelist_json = fetch_mechanism_whitelist()
        >>> import json; rows = json.loads(whitelist_json)
        >>> rows[0]["treatment_l1"]
        'Regulation'
    """
    if config is None:
        config = MASConfig()
    spark = get_spark_session()
    rows = spark.table(f"{config.catalog}.{config.schema}.mechanism_whitelist").collect()
    return json.dumps([row.asDict() for row in rows])


def fetch_market_context(published_at: str) -> dict:
    """Fetch market context features for a given publication time.

    # TODO: implement post TRD-5

    Args:
        published_at: Timestamp string of the article publication

    Returns:
        Dict containing market_vol_at_publish, market_trend_at_publish,
        and dollar_bar_close. V1 returns None for all fields.
    """
    return {
        "market_vol_at_publish": None,
        "market_trend_at_publish": None,
        "dollar_bar_close": None,
    }
