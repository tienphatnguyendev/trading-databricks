"""Tests for ticks domain schema definitions."""
from __future__ import annotations

from trading_databricks.ticks.schemas import BAR_SCHEMA, TICKS_CLEANED_SCHEMA, TICKS_RAW_SCHEMA


def test_ticks_raw_schema_has_required_fields():
    field_names = {f.name for f in TICKS_RAW_SCHEMA.fields}
    assert {"trade_id", "timestamp", "price", "quantity", "side"}.issubset(field_names)


def test_bar_schema_has_ohlcv_fields():
    field_names = {f.name for f in BAR_SCHEMA.fields}
    assert {
        "symbol",
        "exchange",
        "bar_index",
        "bar_start_time",
        "bar_end_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dollar_volume",
        "tick_count",
    }.issubset(field_names)


def test_ticks_cleaned_schema_has_dollar_value():
    field_names = {f.name for f in TICKS_CLEANED_SCHEMA.fields}
    assert "dollar_value" in field_names
