"""Tests for trading_databricks.news.bronze pure transforms."""
from __future__ import annotations

import pytest
from chispa import assert_df_equality
from pyspark.sql import Row


def test_enrich_bronze_adds_ingested_at(spark):
    from trading_databricks.news.bronze import enrich_bronze

    df = spark.createDataFrame(
        [Row(story_id="s1", headline="Test", _metadata=Row(file_path="s3://test/file.ndjson"))]
    )
    result = enrich_bronze(df)
    assert "_ingested_at" in result.columns


def test_enrich_bronze_adds_source_file(spark):
    from trading_databricks.news.bronze import enrich_bronze

    df = spark.createDataFrame(
        [Row(story_id="s1", headline="Test", _metadata=Row(file_path="s3://test/file.ndjson"))]
    )
    result = enrich_bronze(df)
    assert "_source_file" in result.columns
    assert result.collect()[0]["_source_file"] == "s3://test/file.ndjson"


def test_enrich_bronze_preserves_existing_columns(spark):
    from trading_databricks.news.bronze import enrich_bronze

    df = spark.createDataFrame(
        [Row(story_id="s1", headline="Headline", _metadata=Row(file_path="s3://x/y"))]
    )
    result = enrich_bronze(df)
    assert "story_id" in result.columns
    assert "headline" in result.columns
