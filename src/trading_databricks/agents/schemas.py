"""PySpark StructType schemas for all MAS Delta tables.

Module-level constants — no Spark session required for definition.
Import these wherever you need schema enforcement or createDataFrame calls.
"""
from __future__ import annotations

from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DateType,
    DecimalType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Primary output table — one row per news article processed by the MAS.
# Fields 1-6: always populated.
# Fields 7-17: NULL when is_impactful=False.
# Fields 18-20: NULL until TRD-5 market context integration.
# Fields 21-23: V3 additions (nullable for backward compat with V1/V2 rows).
# ---------------------------------------------------------------------------
CAUSAL_FEATURES_SCHEMA = StructType(
    [
        # --- always populated ---
        StructField("story_id", StringType(), nullable=False),
        StructField("published_at", TimestampType(), nullable=False),
        StructField("is_impactful", BooleanType(), nullable=False),
        StructField("gatekeeper_confidence", DoubleType(), nullable=True),
        StructField("agent_model_version", StringType(), nullable=False),
        StructField("processed_at", TimestampType(), nullable=False),
        # --- NULL when is_impactful=False ---
        StructField("treatment_class", StringType(), nullable=True),   # "L1.l2_subcategory"
        StructField("trigger_event", StringType(), nullable=True),     # max 200 chars
        StructField("mechanism", StringType(), nullable=True),
        StructField("asset", StringType(), nullable=True),             # BTC|ETH|BOTH
        StructField("expected_direction", StringType(), nullable=True),  # bullish|bearish|volatile
        StructField("confidence", DoubleType(), nullable=True),        # [0.0, 1.0]
        StructField("causal_chain_json", StringType(), nullable=True),
        StructField("evidence_spans", ArrayType(StringType()), nullable=True),
        StructField("validation_status", StringType(), nullable=True),  # accepted|rejected_*
        StructField("rejected_mechanism_raw", StringType(), nullable=True),
        StructField("retrieval_story_ids", ArrayType(StringType()), nullable=True),
        # --- NULL until TRD-5 market context ---
        StructField("market_vol_at_publish", DoubleType(), nullable=True),
        StructField("market_trend_at_publish", StringType(), nullable=True),
        StructField("dollar_bar_close", DoubleType(), nullable=True),  # Changed from DecimalType(18, 8) for VS compatibility
        # --- V3 additions ---
        StructField("extraction_path", StringType(), nullable=True),          # fast|standard|careful
        StructField("evidence_relevance_score", DoubleType(), nullable=True),  # [0,1] cosine
        StructField("confounding_cluster_id", StringType(), nullable=True),    # UUID if confounded
    ]
)

# ---------------------------------------------------------------------------
# Reference table: 34 rows, 8 L1 categories, seeded by seed_taxonomy().
# ---------------------------------------------------------------------------
TREATMENT_TAXONOMY_SCHEMA = StructType(
    [
        StructField("l1_category", StringType(), nullable=False),
        StructField("l2_subcategory", StringType(), nullable=False),
        StructField("description", StringType(), nullable=False),
        StructField("example", StringType(), nullable=False),
    ]
)

# ---------------------------------------------------------------------------
# Reference table: 40 rows (5 per L1), seeded by seed_taxonomy().
# ---------------------------------------------------------------------------
MECHANISM_WHITELIST_SCHEMA = StructType(
    [
        StructField("treatment_l1", StringType(), nullable=False),
        StructField("mechanism", StringType(), nullable=False),
        StructField("valid_directions", ArrayType(StringType()), nullable=False),
        StructField("description", StringType(), nullable=False),
    ]
)

# ---------------------------------------------------------------------------
# Audit table written by the DAGness audit job (TRD-27).
# One row per audit run.
# ---------------------------------------------------------------------------
DAGNESS_AUDIT_SCHEMA = StructType(
    [
        StructField("audit_date", DateType(), nullable=False),
        StructField("total_accepted_chains", IntegerType(), nullable=False),
        StructField("num_nodes", IntegerType(), nullable=False),
        StructField("num_edges", IntegerType(), nullable=False),
        StructField("dagness_score", DoubleType(), nullable=False),   # 0.0 = perfect DAG
        StructField("is_dag", BooleanType(), nullable=False),
        StructField("top_cycle_contributors", StringType(), nullable=True),  # JSON array
        StructField("flagged_story_ids", ArrayType(StringType()), nullable=True),
    ]
)

# ---------------------------------------------------------------------------
# Audit table written by the confounding detection job (TRD-31).
# One row per confounding cluster detected.
# ---------------------------------------------------------------------------
CONFOUNDING_AUDIT_SCHEMA = StructType(
    [
        StructField("audit_date", DateType(), nullable=False),
        StructField("cluster_start", TimestampType(), nullable=False),
        StructField("cluster_end", TimestampType(), nullable=False),
        StructField("asset", StringType(), nullable=False),
        StructField("expected_direction", StringType(), nullable=False),
        StructField("distinct_treatments", IntegerType(), nullable=False),
        StructField("article_count", IntegerType(), nullable=False),
        StructField("treatment_classes", ArrayType(StringType()), nullable=False),
        StructField("story_ids", ArrayType(StringType()), nullable=False),
        StructField("confounding_flag", StringType(), nullable=False),
    ]
)
