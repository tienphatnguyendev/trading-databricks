"""Gold layer transform for MAS news causal features.

Implements the news_causal_features materialized view logic:
  1. Gatekeeper filters ~95% non-crypto articles
     - spacy mode: pandas UDF with EntityRuler NER (notebook/eval)
     - sql mode: RLIKE with equivalent term patterns (SDP pipeline)
  2. ai_query() calls the mas-news-causal Model Serving endpoint
  3. STRUCT response expanded into CAUSAL_FEATURES_SCHEMA fields

This module is imported by agents/pipeline.py — no logic lives there.
"""

from __future__ import annotations

from typing import Literal

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from trading_databricks.common.uc_identifiers import SILVER_SCHEMA, quoted_table_fqn


# ---------------------------------------------------------------------------
# SQL-based gatekeeper: RLIKE pattern mirroring gatekeeper.py EntityRuler
# ---------------------------------------------------------------------------
_CRYPTO_SINGLE_TERMS = [
    # CRYPTO_ASSET
    "bitcoin", "btc", "ethereum", "eth", "ether", "crypto",
    "cryptocurrency", "altcoin", "doge", "xrp", "solana",
    # CRYPTO_EXCHANGE
    "binance", "coinbase", "kraken", "bybit", "okx", "ftx",
    "gemini", "bitstamp",
    # CRYPTO_STABLECOIN
    "usdc", "usdt", "tether", "stablecoin",
    # CRYPTO_PROTOCOL
    "uniswap", "aave", "makerdao", "chainlink", "arbitrum",
    "comp", "steth",
    # CRYPTO_CONCEPT
    "blockchain", "defi", "nft", "halving", "staking",
    "tokenomics", "l2", "decentralized", "on-chain", "whitepaper",
]

_CRYPTO_MULTI_TERMS = [
    "dai stablecoin", "compound finance", "lido finance",
    "layer 2", "smart contract", "proof of work", "proof of stake",
    "bitcoin mining", "crypto mining", "op token",
]

# Build RLIKE: (?i) for case-insensitive, \b for word boundaries
_single = "|".join(rf"\b{t}\b" for t in _CRYPTO_SINGLE_TERMS)
_WS_PAT = r"\s+"
_multi = "|".join(rf"\b{t.replace(' ', _WS_PAT)}\b" for t in _CRYPTO_MULTI_TERMS)
CRYPTO_RLIKE_PATTERN = f"(?i)({_single}|{_multi})"

# ---------------------------------------------------------------------------
# ai_query return type: STRUCT matching MASPyfuncModel output signature
# With failOnError => false, ai_query wraps in STRUCT<result: ..., errorMessage: STRING>
# ---------------------------------------------------------------------------
_AI_QUERY_RETURN_TYPE = (
    "STRUCT<"
    "  treatment_class: STRING,"
    "  trigger_event: STRING,"
    "  mechanism: STRING,"
    "  asset: STRING,"
    "  expected_direction: STRING,"
    "  confidence: DOUBLE,"
    "  causal_chain_json: STRING,"
    "  evidence_spans: STRING,"
    "  validation_status: STRING,"
    "  rejected_mechanism_raw: STRING,"
    "  retrieval_story_ids: STRING,"
    "  extraction_path: STRING,"
    "  evidence_relevance_score: DOUBLE"
    ">"
)


def _apply_sql_gatekeeper(df: DataFrame) -> DataFrame:
    """Filter to crypto-relevant articles via SQL RLIKE (no spacy)."""
    return df.filter(
        F.col("headline").rlike(CRYPTO_RLIKE_PATTERN)
        | F.col("body").rlike(CRYPTO_RLIKE_PATTERN)
    )


def _apply_spacy_gatekeeper(
    df: DataFrame, config: "GatekeeperConfig",
) -> DataFrame:
    """Filter to crypto-relevant articles via spacy EntityRuler UDF."""
    import pandas as pd
    from pyspark.sql.types import BooleanType
    from trading_databricks.agents.gatekeeper import is_crypto_relevant

    @F.pandas_udf(BooleanType())
    def gatekeeper_udf(headlines: pd.Series, bodies: pd.Series) -> pd.Series:
        results = []
        for headline, body in zip(headlines, bodies, strict=False):
            is_impactful, _ = is_crypto_relevant(
                str(headline) if headline is not None else "",
                str(body) if body is not None else "",
                config=config,
            )
            results.append(bool(is_impactful))
        return pd.Series(results, dtype=object)

    return df.filter(gatekeeper_udf(F.col("headline"), F.col("body")))


def news_causal_features_transform(
    spark: SparkSession,
    *,
    catalog: str,
    gatekeeper_config: "GatekeeperConfig | None" = None,
    gatekeeper_mode: Literal["spacy", "sql"] = "spacy",
) -> DataFrame:
    """Build the news_causal_features DataFrame plan.

    Reads silver.news_cleaned, filters via gatekeeper, enriches via
    ai_query('mas-news-causal', ...), and expands the STRUCT response into
    the 23-field CAUSAL_FEATURES_SCHEMA output.

    Args:
        spark: Active SparkSession (Databricks Connect or SDP runtime).
        catalog: Unity Catalog catalog name.
        gatekeeper_config: GatekeeperConfig for spacy mode. Ignored in sql mode.
        gatekeeper_mode: 'spacy' (default) for EntityRuler UDF, 'sql' for RLIKE.
            Use 'sql' in SDP pipeline contexts where spacy isn't installed.

    Returns:
        DataFrame with schema matching CAUSAL_FEATURES_SCHEMA (23 fields).
        Not yet materialized — caller triggers execution.
    """
    cleaned = spark.table(quoted_table_fqn(catalog, SILVER_SCHEMA, "news_cleaned"))

    # Stage 1+2 gatekeeper: keep only crypto-impactful articles (~5%)
    if gatekeeper_mode == "sql":
        filtered = _apply_sql_gatekeeper(cleaned)
    else:
        if gatekeeper_config is None:
            from trading_databricks.agents.config import GatekeeperConfig
            gatekeeper_config = GatekeeperConfig()
        filtered = _apply_spacy_gatekeeper(cleaned, gatekeeper_config)

    # Limit to recent articles to keep ai_query volume manageable on
    # serverless SDP (avoid cluster timeout on large backlogs).
    filtered = filtered.filter(
        F.col("published_at") >= F.current_timestamp() - F.expr("INTERVAL 7 DAYS")
    )

    # With failOnError => false, ai_query returns:
    #   STRUCT<result: STRUCT<...model output...>, errorMessage: STRING>
    # Access model fields via cr.result.<field>
    _r = F.col("cr.result")

    return (
        filtered
        # Call MAS endpoint — failOnError => false means individual failures
        # return null result + errorMessage instead of crashing the MV.
        .withColumn(
            "cr",
            F.expr(
                f"ai_query("
                f"  endpoint => 'mas-news-causal',"
                f"  request => named_struct("
                f"    'headline', headline,"
                f"    'body', body,"
                f"    'published_at', CAST(published_at AS STRING)"
                f"  ),"
                f"  returnType => '{_AI_QUERY_RETURN_TYPE}',"
                f"  failOnError => false"
                f")"
            ),
        )
        # Expand STRUCT into individual output columns
        .select(
            F.col("story_id"),
            F.col("published_at"),
            F.lit(True).alias("is_impactful"),
            F.lit(1.0).cast("double").alias("gatekeeper_confidence"),
            F.lit("v1.0").alias("agent_model_version"),
            F.current_timestamp().alias("processed_at"),
            _r.treatment_class.alias("treatment_class"),
            _r.trigger_event.alias("trigger_event"),
            _r.mechanism.alias("mechanism"),
            _r.asset.alias("asset"),
            _r.expected_direction.alias("expected_direction"),
            _r.confidence.alias("confidence"),
            _r.causal_chain_json.alias("causal_chain_json"),
            F.from_json(_r.evidence_spans, "array<string>").alias("evidence_spans"),
            _r.validation_status.alias("validation_status"),
            _r.rejected_mechanism_raw.alias("rejected_mechanism_raw"),
            F.from_json(
                _r.retrieval_story_ids, "array<string>"
            ).alias("retrieval_story_ids"),
            F.lit(None).cast("double").alias("market_vol_at_publish"),
            F.lit(None).cast("string").alias("market_trend_at_publish"),
            F.lit(None).cast("double").alias("dollar_bar_close"),
            _r.extraction_path.alias("extraction_path"),
            _r.evidence_relevance_score.alias("evidence_relevance_score"),
            F.lit(None).cast("string").alias("confounding_cluster_id"),
        )
    )
