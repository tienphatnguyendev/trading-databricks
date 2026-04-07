"""Few-shot retriever node for MAS agent pipeline.

Queries Vector Search index for similar accepted causal chains.
Falls back to static examples during cold-start or VS failures.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


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

    Args:
        spark: SparkSession instance
        catalog: Databricks catalog name
        schema: Schema name (typically 'gold')
        cache_ttl_seconds: Cache TTL in seconds (default: 300 = 5min)

    Returns:
        Count of rows where validation_status='accepted'

    Example:
        >>> count = get_accepted_chain_count(
        ...     spark,
        ...     catalog="trading-signal-gen-dev",
        ...     schema="gold"
        ... )
        >>> print(f"Accepted chains: {count}")
    """
    cache_key = f"{catalog}.{schema}"
    now = time.time()

    # Check cache
    if cache_key in _accepted_count_cache:
        count, timestamp = _accepted_count_cache[cache_key]
        if now - timestamp < cache_ttl_seconds:
            logger.debug(f"Cache hit for {cache_key}: count={count}")
            return count

    # Query source table
    logger.debug(f"Cache miss for {cache_key}, querying DB")
    query = f"""
        SELECT COUNT(*)
        FROM `{catalog}`.`{schema}`.news_causal_features
        WHERE validation_status = 'accepted'
    """
    count = spark.sql(query).collect()[0][0]

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

    Args:
        headline: Article headline for semantic search
        index_name: Full VS index name (catalog.schema.index)
        num_results: Number of results to return (default: 5)

    Returns:
        List of dicts with VS result format:
        - story_id: str
        - treatment_class: str
        - mechanism: str
        - expected_direction: str
        - confidence: float
        - trigger_event: str
        - causal_chain_json: str

        Returns empty list on VS error (fail gracefully).

    Example:
        >>> results = query_vector_search(
        ...     "Fed raises interest rates",
        ...     index_name="trading-signal-gen-dev.gold.causal_chains_idx"
        ... )
        >>> len(results)
        5
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
    """Merge dynamic + static examples, prioritizing dynamic.

    Strategy: Take up to max_total dynamic examples first, then fill
    remaining slots with static examples.

    Args:
        dynamic: VS query results (0-N examples)
        static: Hardcoded fallback examples (5 examples)
        max_total: Maximum total examples to return (default: 5)

    Returns:
        List of up to max_total examples, dynamic-first

    Example:
        >>> dynamic = [{"story_id": "vs-1"}, {"story_id": "vs-2"}]
        >>> static = [{"story_id": "static-1"}, ...]
        >>> merged = merge_examples(dynamic, static, max_total=5)
        >>> len(merged)
        5
        >>> merged[0]["story_id"]
        'vs-1'
        >>> merged[2]["story_id"]
        'static-1'
    """
    # Take dynamic first
    result = dynamic[:max_total]

    # Fill remaining slots with static
    remaining = max_total - len(result)
    if remaining > 0:
        result.extend(static[:remaining])

    logger.debug(
        f"Merged {len(dynamic)} dynamic + {remaining} static = {len(result)} total"
    )

    return result


def few_shot_retriever():
    """Placeholder for few-shot retriever node implementation."""
    raise NotImplementedError("few_shot_retriever node not yet implemented")
