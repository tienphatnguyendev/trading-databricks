"""SDP pipeline definition for the news domain.

This file contains ONLY @dp.table / @dp.materialized_view decorators and
dp.create_auto_cdc_flow calls.  All transformation logic lives in the
sibling modules (bronze.py, silver.py, gold.py) and is locally testable.

Pipeline:
  bronze_news_raw      — streaming table via cloudFiles Auto Loader
  silver_news_cleaned  — SCD Type 1 AUTO CDC on story_id, sequenced by scraped_at
  gold_news_features   — materialized view (full recompute)
  _news_latest         — temporary view: last 24h for gen-AI agent queries

DABs pipeline resource: resources/news_pipeline.pipeline.yml
"""
from __future__ import annotations

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.functions import col

from trading_databricks.news.bronze import enrich_bronze
from trading_databricks.news.expectations import (
    BRONZE_VALID_BATCH_ID,
    BRONZE_VALID_SCRAPED_AT,
    BRONZE_VALID_STORY_ID,
    SILVER_VALID_HEADLINE,
    SILVER_VALID_STORY_ID,
)
from trading_databricks.news.schemas import NEWS_RAW_SCHEMA_HINTS

# ── Pipeline configuration (injected by DABs at runtime) ─────────────────────

bronze_schema = spark.conf.get("bronze_schema")  # noqa: F821  (spark injected by SDP)
silver_schema = spark.conf.get("silver_schema")
gold_schema = spark.conf.get("gold_schema")
schema_location_base = spark.conf.get("schema_location_base")
source_bucket = spark.conf.get("source_bucket")

# ── Bronze ────────────────────────────────────────────────────────────────────


@dp.table(
    name=f"{bronze_schema}.news_raw",
    comment="Raw news records ingested from S3 via Auto Loader (append-only).",
    cluster_by=["scraped_at"],
)
@dp.expect_or_drop("valid_story_id", BRONZE_VALID_STORY_ID)
@dp.expect_or_drop("valid_scraped_at", BRONZE_VALID_SCRAPED_AT)
@dp.expect_or_drop("valid_batch_id", BRONZE_VALID_BATCH_ID)
def bronze_news_raw():
    return enrich_bronze(
        spark.readStream.format("cloudFiles")  # noqa: F821
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaHints", NEWS_RAW_SCHEMA_HINTS)
        .option("cloudFiles.schemaLocation", f"{schema_location_base}/bronze_news_raw")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("rescuedDataColumn", "_rescued_data")
        .load(f"s3://{source_bucket}/news/tradingview/year=*/month=*/day=*/")
    )


# ── Silver — AUTO CDC (SCD Type 1) ────────────────────────────────────────────

dp.create_streaming_table(
    name=f"{silver_schema}.news_cleaned",
    comment="Deduplicated, cast, and normalised news records (SCD Type 1 on story_id).",
    cluster_by=["published_at"],
    expect_or_drop={
        "valid_story_id": SILVER_VALID_STORY_ID,
        "valid_headline": SILVER_VALID_HEADLINE,
    },
)

dp.create_auto_cdc_flow(
    target=f"{silver_schema}.news_cleaned",
    source=f"{bronze_schema}.news_raw",
    keys=["story_id"],
    sequence_by=col("scraped_at"),
    stored_as_scd_type="1",
    apply_as_deletes=None,
)

# ── Gold ──────────────────────────────────────────────────────────────────────


@dp.materialized_view(
    name=f"{gold_schema}.news_features",
    comment="Sentiment, named entities, and classification flags for each news article.",
    cluster_by=["published_at"],
)
def gold_news_features():
    # TODO: invoke news.gold.news_features() once NLP stack is selected (TRD-4)
    raise NotImplementedError("Gold news features not yet implemented — see TRD-4")


# ── Temporary view for gen-AI agents ─────────────────────────────────────────


@dp.temporary_view(name="_news_latest")
def _news_latest():
    """Last 24 hours of cleaned news — pipeline-internal, not persisted to catalog."""
    return spark.read.table(f"{silver_schema}.news_cleaned").filter(  # noqa: F821
        F.col("published_at") >= F.date_sub(F.current_date(), 1)
    )
