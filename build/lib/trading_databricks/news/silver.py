"""Silver layer transforms for the news domain.

Pure PySpark functions — no SDP decorators, no SparkSession creation.
Deduplication is handled declaratively by dp.create_auto_cdc_flow (SCD Type 1)
in pipeline.py; the functions here prepare the data before CDC is applied.

Source: {catalog}.bronze.news_raw
Target: {catalog}.silver.news_cleaned (AUTO CDC, SCD Type 1 on story_id)
"""
from __future__ import annotations

import re

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ── Individual transforms ─────────────────────────────────────────────────────

def cast_timestamps(df: DataFrame) -> DataFrame:
    """Cast published_at and scraped_at STRING columns to TIMESTAMP.

    Nulls in published_at are preserved (field is nullable in source contract).
    """
    return df.withColumn(
        "published_at", F.to_timestamp("published_at")
    ).withColumn("scraped_at", F.to_timestamp("scraped_at"))


def normalize_text(df: DataFrame) -> DataFrame:
    """Trim and collapse internal whitespace in headline and body.

    Null values are preserved unchanged.
    """
    for col in ("headline", "body"):
        df = df.withColumn(
            col,
            F.when(
                F.col(col).isNotNull(),
                F.regexp_replace(F.trim(F.col(col)), r"\s+", " "),
            ).otherwise(F.col(col)),
        )
    return df


def flatten_provider(df: DataFrame) -> DataFrame:
    """Parse the provider JSON string into typed columns.

    Extracts id, name, and logoId from the provider JSON object.
    The original provider column is dropped after flattening.
    """
    parsed = F.from_json(
        F.col("provider"),
        "id STRING, name STRING, logoId STRING",
    )
    return (
        df.withColumn("_provider", parsed)
        .withColumn("provider_id", F.col("_provider.id"))
        .withColumn("provider_name", F.col("_provider.name"))
        .withColumn("provider_logo_id", F.col("_provider.logoId"))
        .drop("_provider", "provider")
    )


def add_ingested_at(df: DataFrame) -> DataFrame:
    """Add _ingested_at metadata column (current processing timestamp)."""
    return df.withColumn("_ingested_at", F.current_timestamp())


# ── Pipeline chain ────────────────────────────────────────────────────────────

def clean_news(df: DataFrame) -> DataFrame:
    """Full silver transform chain for news.

    Order matters:
    1. cast_timestamps — before normalize so TIMESTAMP nulls propagate cleanly
    2. normalize_text — after cast so we operate on typed data
    3. flatten_provider — independent, can run after text normalization
    4. add_ingested_at — always last so it captures processing time
    """
    return (
        df.transform(cast_timestamps)
        .transform(normalize_text)
        .transform(flatten_provider)
        .transform(add_ingested_at)
    )
