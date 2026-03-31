"""Tests for trading_databricks.ticks.silver pure transforms."""
from __future__ import annotations

from decimal import Decimal

from pyspark.sql import Row
from pyspark.sql.types import DecimalType, TimestampType


def test_cast_types_converts_timestamp(spark):
    from trading_databricks.ticks.silver import cast_types

    df = spark.createDataFrame(
        [Row(timestamp="2026-03-31T06:00:00.000000Z", price="50000.12345678", quantity="0.01234567")]
    )
    result = cast_types(df)
    assert result.schema["timestamp"].dataType == TimestampType()


def test_cast_types_converts_price_to_decimal(spark):
    from trading_databricks.ticks.silver import cast_types

    df = spark.createDataFrame(
        [Row(timestamp="2026-03-31T06:00:00Z", price="50000.12345678", quantity="0.01234567")]
    )
    result = cast_types(df)
    assert isinstance(result.schema["price"].dataType, DecimalType)


def test_compute_dollar_value(spark):
    from trading_databricks.ticks.silver import cast_types, compute_dollar_value

    df = spark.createDataFrame(
        [Row(timestamp="2026-03-31T06:00:00Z", price="50000.0", quantity="2.0")]
    )
    result = cast_types(df).transform(compute_dollar_value)
    row = result.collect()[0]
    assert float(row["dollar_value"]) == pytest.approx(100000.0, rel=1e-6)


def test_add_ingested_at_adds_column(spark):
    from trading_databricks.ticks.silver import add_ingested_at

    df = spark.createDataFrame([Row(trade_id="t1")])
    result = add_ingested_at(df)
    assert "_ingested_at" in result.columns


def test_clean_ticks_full_pipeline(spark):
    from trading_databricks.ticks.silver import clean_ticks

    df = spark.createDataFrame(
        [Row(timestamp="2026-03-31T06:00:00Z", price="50000.0", quantity="2.0", trade_id="t1")]
    )
    result = clean_ticks(df)
    cols = result.columns
    assert "dollar_value" in cols
    assert "_ingested_at" in cols


import pytest
