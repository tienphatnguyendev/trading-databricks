"""Bronze layer transforms for the ticks domain.

Pure PySpark functions — no SDP decorators, no SparkSession creation.

Source: S3 NDJSON files written by tick-scraper-aws via Auto Loader.
Target: {catalog}.bronze.ticks_raw (streaming table, append-only).

STATUS: awaiting tick-scraper-aws schema confirmation.
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def enrich_bronze(df: DataFrame) -> DataFrame:
    """Add ingestion metadata columns to the raw Auto Loader DataFrame.

    Args:
        df: Raw DataFrame produced by cloudFiles Auto Loader.

    Returns:
        DataFrame with _ingested_at and _source_file columns appended.
    """
    return df.withColumn("_ingested_at", F.current_timestamp()).withColumn(
        "_source_file", F.col("_metadata.file_path")
    )
