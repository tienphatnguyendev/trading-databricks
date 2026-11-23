"""Few-shot retriever node for MAS agent pipeline.

Queries Vector Search index for similar accepted causal chains.
Falls back to static examples during cold-start or VS failures.
"""
from __future__ import annotations

import logging
import time
from typing import TypedDict

from trading_databricks.agents.config import MASConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type hints
# ---------------------------------------------------------------------------
from trading_databricks.agents.state import MASState


# ---------------------------------------------------------------------------
# Static few-shot examples — 5 examples covering 5 L1 categories
# Format mirrors VS query result schema exactly
# ---------------------------------------------------------------------------
STATIC_FEW_SHOT_EXAMPLES: list[dict] = [
    {
        "story_id": "static-example-reg-etf",
        "treatment_class": "Regulation.etf_ruling",
        "mechanism": "institutional_access",
        "expected_direction": "bullish",
        "confidence": 0.95,
        "trigger_event": "SEC approves first spot Bitcoin ETF applications from BlackRock and Fidelity",
        "causal_chain_json": '{"trigger_event": "SEC approves first spot Bitcoin ETF applications from BlackRock and Fidelity", "mechanism": "institutional_access", "asset": "BTC", "expected_direction": "bullish", "confidence": 0.95, "evidence_spans": ["SEC approves spot Bitcoin ETF", "opens regulated access for institutions"], "reasoning": "ETF approval removes regulatory barriers, enabling traditional finance institutions to allocate capital to BTC through regulated vehicles."}',
    },
    {
        "story_id": "static-example-macro-rate",
        "treatment_class": "Macro.rate_decision",
        "mechanism": "liquidity_contraction",
        "expected_direction": "bearish",
        "confidence": 0.85,
        "trigger_event": "Federal Reserve raises interest rates by 75 basis points to combat inflation",
        "causal_chain_json": '{"trigger_event": "Federal Reserve raises interest rates by 75 basis points to combat inflation", "mechanism": "liquidity_contraction", "asset": "BOTH", "expected_direction": "bearish", "confidence": 0.85, "evidence_spans": ["Fed raises rates by 75 basis points", "tightening monetary policy"], "reasoning": "Higher interest rates increase borrowing costs and drain liquidity from risk assets, creating bearish pressure on crypto markets."}',
    },
    {
        "story_id": "static-example-sec-hack",
        "treatment_class": "Security.exchange_hack",
        "mechanism": "trust_erosion",
        "expected_direction": "bearish",
        "confidence": 0.90,
        "trigger_event": "Major cryptocurrency exchange confirms $480M hack, suspends withdrawals",
        "causal_chain_json": '{"trigger_event": "Major cryptocurrency exchange confirms $480M hack, suspends withdrawals", "mechanism": "trust_erosion", "asset": "BOTH", "expected_direction": "bearish", "confidence": 0.90, "evidence_spans": ["exchange confirms hack", "suspends withdrawals", "$480M stolen"], "reasoning": "Exchange security breach destroys user confidence in centralized custody, triggering panic withdrawals and sell-offs across the market."}',
    },
    {
        "story_id": "static-example-supply-unlock",
        "treatment_class": "Supply.unlock_distribution",
        "mechanism": "sell_pressure",
        "expected_direction": "bearish",
        "confidence": 0.80,
        "trigger_event": "Mt. Gox trustee transfers 10,000 BTC to exchanges ahead of creditor distributions",
        "causal_chain_json": '{"trigger_event": "Mt. Gox trustee transfers 10,000 BTC to exchanges ahead of creditor distributions", "mechanism": "sell_pressure", "asset": "BTC", "expected_direction": "bearish", "confidence": 0.80, "evidence_spans": ["Mt. Gox transfers 10,000 BTC to exchanges", "creditor distributions"], "reasoning": "Large-scale distribution of long-dormant BTC creates immediate sell pressure as creditors liquidate recovered assets."}',
    },
    {
        "story_id": "static-example-adoption-inst",
        "treatment_class": "Adoption.institutional_buy",
        "mechanism": "institutional_inflow",
        "expected_direction": "bullish",
        "confidence": 0.88,
        "trigger_event": "BlackRock announces 5% Bitcoin allocation across flagship funds",
        "causal_chain_json": '{"trigger_event": "BlackRock announces 5% Bitcoin allocation across flagship funds", "mechanism": "institutional_inflow", "asset": "BTC", "expected_direction": "bullish", "confidence": 0.88, "evidence_spans": ["BlackRock announces 5% Bitcoin allocation", "flagship funds"], "reasoning": "Major asset manager allocation signals institutional validation and creates sustained buying pressure from large capital pools."}',
    },
]


# Module-level cache: {catalog.schema: (count, timestamp)}
_accepted_count_cache: dict[str, tuple[int, float]] = {}


def get_accepted_chain_count(
    spark,
    *,
    catalog: str,
    schema: str,
    cache_ttl_seconds: int = 300,
) -> int:
    """Query count of accepted causal chains, cached for 5 minutes.

    Returns 0 when:
    - spark is None (Model Serving, no Spark available)
    - Table does not exist yet (cold-start)
    - Any query error
    """
    if spark is None:
        return 0

    cache_key = f"{catalog}.{schema}"
    now = time.time()

    # Check cache
    if cache_key in _accepted_count_cache:
        count, timestamp = _accepted_count_cache[cache_key]
        if now - timestamp < cache_ttl_seconds:
            logger.debug(f"Cache hit for {cache_key}: count={count}")
            return count

    # Query source table
    try:
        logger.debug(f"Cache miss for {cache_key}, querying DB")
        query = f"""
            SELECT COUNT(*)
            FROM `{catalog}`.`{schema}`.news_causal_features
            WHERE validation_status = 'accepted'
        """
        count = spark.sql(query).collect()[0][0]
    except Exception as e:
        # TABLE_NOT_FOUND during cold-start, or other query errors
        logger.warning("get_accepted_chain_count failed (cold-start?): %s", e)
        return 0

    # Update cache
    _accepted_count_cache[cache_key] = (count, now)
    logger.info(f"Accepted chain count: {count} (cached for {cache_ttl_seconds}s)")

    return count


# Placeholder for MCP tool - will be imported properly when MCP is available
# For testing, this gets monkeypatched
def query_vs_index(*args, **kwargs):
    """Placeholder for Databricks MCP query_vs_index tool."""
    raise NotImplementedError("query_vs_index must be provided by Databricks MCP")


def query_vector_search(
    headline: str,
    *,
    index_name: str,
    num_results: int = 5,
) -> list[dict]:
    """Query VS index for similar accepted causal chains.

    Returns empty list on VS error (fail gracefully).
    """
    if not headline or not headline.strip():
        logger.warning("Empty headline provided, returning empty results")
        return []

    try:
        logger.debug(f"Querying VS index: {index_name} for headline: {headline[:100]}")

        # Module-level reference; monkeypatch replaces it in tests
        query_fn = query_vs_index

        results = query_fn(
            index_name=index_name,
            columns=[
                "story_id",
                "treatment_class",
                "mechanism",
                "expected_direction",
                "confidence",
                "trigger_event",
                "causal_chain_json",
            ],
            query_text=headline,
            num_results=num_results,
            filters_json='{"validation_status": ["accepted"]}',
        )

        logger.info(f"VS query returned {len(results)} results")
        return results

    except Exception as e:
        logger.warning("VS query failed: %s, returning empty results", e)
        return []


def merge_examples(
    dynamic: list[dict],
    static: list[dict],
    *,
    max_total: int = 5,
) -> list[dict]:
    """Merge dynamic + static examples, prioritizing dynamic."""
    result = dynamic[:max_total]
    remaining = max_total - len(result)
    if remaining > 0:
        result.extend(static[:remaining])

    logger.debug(
        f"Merged {len(dynamic)} dynamic + {remaining} static = {len(result)} total"
    )
    return result


# ---------------------------------------------------------------------------
# Spark session helper
# ---------------------------------------------------------------------------
def get_spark_session():
    """Get Spark session (placeholder for Databricks Connect).

    In production, this will be replaced with actual Databricks Connect
    session via environment setup. For testing, this gets monkeypatched.

    Raises:
        NotImplementedError: When called without Databricks Connect setup.
    """
    raise NotImplementedError("get_spark_session must be provided by Databricks Connect")


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------
def few_shot_retriever(state: MASState, config: MASConfig) -> MASState:
    """Retrieve few-shot examples for the Causality Extractor.

    Gracefully handles no-Spark environments (Model Serving containers)
    by falling back to static examples (cold-start mode).
    """
    headline = state["headline"]

    # Step 1: Query accepted chain count (cached 5 minutes)
    # Graceful fallback: if Spark is unavailable, treat as cold-start.
    try:
        spark = get_spark_session()
    except NotImplementedError:
        logger.info("No Spark session (serving mode) — using static examples")
        return {
            **state,
            "few_shot_examples": STATIC_FEW_SHOT_EXAMPLES,
            "retrieval_story_ids": [],
        }

    accepted_count = get_accepted_chain_count(
        spark,
        catalog=config.catalog,
        schema=config.schema,
    )

    # Step 2: Cold-start guard
    if accepted_count < config.router.min_index_size:
        logger.info(
            f"Cold start: {accepted_count} < {config.router.min_index_size}, "
            f"using static examples only"
        )
        return {
            **state,
            "few_shot_examples": STATIC_FEW_SHOT_EXAMPLES,
            "retrieval_story_ids": [],
        }

    # Step 3: Query VS (warm path)
    logger.info(f"Warm start: {accepted_count} chains available, querying VS")
    dynamic_results = query_vector_search(
        headline,
        index_name=config.vs_index_name,
        num_results=5,
    )

    # Step 4: Merge dynamic + static
    merged_examples = merge_examples(
        dynamic_results,
        STATIC_FEW_SHOT_EXAMPLES,
        max_total=5,
    )

    # Step 5: Update state
    logger.info(
        f"Retrieved {len(dynamic_results)} dynamic + "
        f"{len(merged_examples) - len(dynamic_results)} static = "
        f"{len(merged_examples)} total examples"
    )

    return {
        **state,
        "few_shot_examples": merged_examples,
        "retrieval_story_ids": [ex["story_id"] for ex in dynamic_results],
    }
