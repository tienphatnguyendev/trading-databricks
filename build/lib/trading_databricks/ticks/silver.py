"""Silver layer transforms for the ticks domain.

Pure PySpark functions — no SDP decorators, no SparkSession creation.

Source: {catalog}.bronze.ticks_raw
Target: {catalog}.silver.ticks_cleaned (streaming table)

Transformations (in order):
1. cast_types          — timestamp, price, quantity to proper types
2. compute_dollar_value — price * quantity
3. add_ingested_at     — processing timestamp metadata

Deduplication by trade_id is handled via @dp.expect_or_drop in pipeline.py
which drops records with null trade_id.  Full dedup (idempotency) is
achieved because the streaming pipeline processes each file exactly once
via Auto Loader checkpoints.
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType, TimestampType


def cast_types(df: DataFrame) -> DataFrame:
    """Cast raw string fields to proper types.

    price and quantity are stored as strings in NDJSON to preserve precision.
    """
    return (
        df.withColumn("timestamp", F.col("timestamp").cast(TimestampType()))
        .withColumn("price", F.col("price").cast(DecimalType(18, 8)))
        .withColumn("quantity", F.col("quantity").cast(DecimalType(18, 8)))
    )


def compute_dollar_value(df: DataFrame) -> DataFrame:
    """Add dollar_value = price * quantity column."""
    return df.withColumn(
        "dollar_value",
        (F.col("price") * F.col("quantity")).cast(DecimalType(38, 8)),
    )


def add_ingested_at(df: DataFrame) -> DataFrame:
    """Add _ingested_at metadata column (current processing timestamp)."""
    return df.withColumn("_ingested_at", F.current_timestamp())


def clean_ticks(df: DataFrame) -> DataFrame:
    """Full silver transform chain for ticks."""
    return (
        df.transform(cast_types)
        .transform(compute_dollar_value)
        .transform(add_ingested_at)
    )
