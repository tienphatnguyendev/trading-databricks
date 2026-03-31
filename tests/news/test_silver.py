"""Tests for trading_databricks.news.silver pure transforms."""
from __future__ import annotations

import pytest
from pyspark.sql import Row
from pyspark.sql.types import StringType, StructField, StructType, TimestampType


def test_cast_timestamps_converts_scraped_at(spark):
    from trading_databricks.news.silver import cast_timestamps

    df = spark.createDataFrame(
        [Row(scraped_at="2026-03-31T06:00:00Z", published_at="2026-03-31T05:00:00Z")]
    )
    result = cast_timestamps(df)
    assert result.schema["scraped_at"].dataType == TimestampType()


def test_cast_timestamps_null_published_at_stays_null(spark):
    from trading_databricks.news.silver import cast_timestamps

    schema = StructType([
        StructField("scraped_at", StringType(), nullable=False),
        StructField("published_at", StringType(), nullable=True),
    ])
    df = spark.createDataFrame([("2026-03-31T06:00:00Z", None)], schema=schema)
    result = cast_timestamps(df)
    row = result.collect()[0]
    assert row["published_at"] is None


def test_normalize_text_trims_whitespace(spark):
    from trading_databricks.news.silver import normalize_text

    df = spark.createDataFrame([Row(headline="  Bitcoin surges  ", body="  Great article  ")])
    result = normalize_text(df)
    row = result.collect()[0]
    assert row["headline"] == "Bitcoin surges"
    assert row["body"] == "Great article"


def test_normalize_text_collapses_internal_whitespace(spark):
    from trading_databricks.news.silver import normalize_text

    schema = StructType([
        StructField("headline", StringType(), nullable=True),
        StructField("body", StringType(), nullable=True),
    ])
    df = spark.createDataFrame([("Bitcoin  surges  again", None)], schema=schema)
    result = normalize_text(df)
    row = result.collect()[0]
    assert row["headline"] == "Bitcoin surges again"


def test_normalize_text_null_body_stays_null(spark):
    from trading_databricks.news.silver import normalize_text

    schema = StructType([
        StructField("headline", StringType(), nullable=True),
        StructField("body", StringType(), nullable=True),
    ])
    df = spark.createDataFrame([("Test", None)], schema=schema)
    result = normalize_text(df)
    assert result.collect()[0]["body"] is None


def test_flatten_provider_extracts_fields(spark):
    from trading_databricks.news.silver import flatten_provider

    df = spark.createDataFrame(
        [Row(provider='{"id": "tv", "name": "TradingView", "logoId": "logo123"}')]
    )
    result = flatten_provider(df)
    row = result.collect()[0]
    assert row["provider_id"] == "tv"
    assert row["provider_name"] == "TradingView"
    assert row["provider_logo_id"] == "logo123"
    assert "provider" not in result.columns


def test_add_ingested_at_adds_column(spark):
    from trading_databricks.news.silver import add_ingested_at

    df = spark.createDataFrame([Row(story_id="s1")])
    result = add_ingested_at(df)
    assert "_ingested_at" in result.columns


def test_clean_news_full_pipeline(spark):
    from trading_databricks.news.silver import clean_news

    schema = StructType([
        StructField("story_id", StringType(), nullable=False),
        StructField("headline", StringType(), nullable=True),
        StructField("body", StringType(), nullable=True),
        StructField("published_at", StringType(), nullable=True),
        StructField("scraped_at", StringType(), nullable=False),
        StructField("provider", StringType(), nullable=True),
    ])
    df = spark.createDataFrame(
        [("s1", "  Bitcoin  surges  ", None, "2026-03-31T05:00:00Z",
          "2026-03-31T06:00:00Z", '{"id": "tv", "name": "TradingView", "logoId": "lg1"}')],
        schema=schema,
    )
    result = clean_news(df)
    row = result.collect()[0]
    assert row["headline"] == "Bitcoin surges"
    assert row["body"] is None
    assert row["provider_id"] == "tv"
    assert "_ingested_at" in result.columns
    assert "provider" not in result.columns
