"""Gold layer transform for MAS news causal features.

Implements the news_causal_features materialized view logic:
  1. Gatekeeper pandas UDF filters ~95% non-crypto articles
  2. ai_query() calls the mas-news-causal Model Serving endpoint
  3. JSON response expanded into CAUSAL_FEATURES_SCHEMA fields

This module is imported by agents/pipeline.py — no logic lives there.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType

from trading_databricks.agents.config import GatekeeperConfig
from trading_databricks.agents.gatekeeper import is_crypto_relevant


def _make_gatekeeper_udf(config: GatekeeperConfig) -> Callable:
    """Return a pandas UDF that filters articles by crypto-relevance.

    The UDF calls is_crypto_relevant() (which caches build_nlp() per process
    via @functools.cache) for each row. Calling build_nlp() inside the UDF
    body per spec §5.4 — cache is per-executor-process, not serialized.

    Args:
        config: GatekeeperConfig — text_window used for body truncation.

    Returns:
        A pandas_udf(BooleanType()) callable taking (headlines, bodies) Series.
    """

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

    return gatekeeper_udf


def news_causal_features_transform(
    spark: SparkSession,
    *,
    catalog: str,
    gatekeeper_config: GatekeeperConfig,
) -> DataFrame:
    """Build the news_causal_features DataFrame plan.

    Reads silver.news_cleaned, filters via gatekeeper, enriches via
    ai_query('mas-news-causal', ...), and expands the JSON response into
    the 23-field CAUSAL_FEATURES_SCHEMA output.

    Args:
        spark: Active SparkSession (Databricks Connect or SDP runtime).
        catalog: Unity Catalog catalog name (e.g., "trading-signal-gen-dev").
        gatekeeper_config: GatekeeperConfig for NER-based filtering.

    Returns:
        DataFrame with schema matching CAUSAL_FEATURES_SCHEMA (23 fields).
        Not yet materialized — caller triggers execution.
    """
    gatekeeper_udf = _make_gatekeeper_udf(gatekeeper_config)

    _cr = F.col("causal_result")

    return (
        spark.table(f"`{catalog}`.silver.news_cleaned")
        # Stage 1+2 gatekeeper: keep only crypto-impactful articles (~5%)
        .filter(gatekeeper_udf(F.col("headline"), F.col("body")))
        # Call MAS endpoint — returns JSON string with all causal fields
        .withColumn(
            "causal_result",
            F.expr(
                "ai_query('mas-news-causal',"
                " to_json(struct(headline, body, published_at)))"
            ),
        )
        # Expand causal_result JSON into individual output columns
        .select(
            F.col("story_id"),
            F.col("published_at"),
            F.lit(True).alias("is_impactful"),
            F.lit(1.0).cast("double").alias("gatekeeper_confidence"),
            F.lit("v1.0").alias("agent_model_version"),
            F.current_timestamp().alias("processed_at"),
            F.get_json_object(_cr, "$.treatment_class").alias("treatment_class"),
            F.get_json_object(_cr, "$.trigger_event").alias("trigger_event"),
            F.get_json_object(_cr, "$.mechanism").alias("mechanism"),
            F.get_json_object(_cr, "$.asset").alias("asset"),
            F.get_json_object(_cr, "$.expected_direction").alias("expected_direction"),
            F.get_json_object(_cr, "$.confidence").cast("double").alias("confidence"),
            F.get_json_object(_cr, "$.causal_chain_json").alias("causal_chain_json"),
            F.from_json(
                F.get_json_object(_cr, "$.evidence_spans"),
                "array<string>",
            ).alias("evidence_spans"),
            F.get_json_object(_cr, "$.validation_status").alias("validation_status"),
            F.get_json_object(_cr, "$.rejected_mechanism_raw").alias(
                "rejected_mechanism_raw"
            ),
            F.from_json(
                F.get_json_object(_cr, "$.retrieval_story_ids"),
                "array<string>",
            ).alias("retrieval_story_ids"),
            F.lit(None).cast("double").alias("market_vol_at_publish"),
            F.lit(None).cast("string").alias("market_trend_at_publish"),
            F.lit(None).cast("decimal(18,8)").alias("dollar_bar_close"),
            F.get_json_object(_cr, "$.extraction_path").alias("extraction_path"),
            F.get_json_object(_cr, "$.evidence_relevance_score")
            .cast("double")
            .alias("evidence_relevance_score"),
            F.lit(None).cast("string").alias("confounding_cluster_id"),
        )
    )
