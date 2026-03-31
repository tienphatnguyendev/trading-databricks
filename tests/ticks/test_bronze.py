"""Tests for trading_databricks.ticks.bronze pure transforms."""
from __future__ import annotations

from pyspark.sql import Row


def test_enrich_bronze_adds_ingested_at(spark):
    from trading_databricks.ticks.bronze import enrich_bronze

    df = spark.createDataFrame(
        [Row(trade_id="t1", price="50000", _metadata=Row(file_path="s3://test/f.ndjson"))]
    )
    result = enrich_bronze(df)
    assert "_ingested_at" in result.columns


def test_enrich_bronze_adds_source_file(spark):
    from trading_databricks.ticks.bronze import enrich_bronze

    df = spark.createDataFrame(
        [Row(trade_id="t1", price="50000", _metadata=Row(file_path="s3://bucket/f.ndjson"))]
    )
    result = enrich_bronze(df)
    assert result.collect()[0]["_source_file"] == "s3://bucket/f.ndjson"
