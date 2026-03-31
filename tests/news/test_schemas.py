"""Tests for news domain schema definitions."""
from __future__ import annotations

from pyspark.sql.types import StringType

from trading_databricks.news.schemas import NEWS_RAW_SCHEMA, NEWS_RAW_SCHEMA_HINTS


def test_news_raw_schema_has_required_fields():
    field_names = {f.name for f in NEWS_RAW_SCHEMA.fields}
    required = {"story_id", "headline", "scraped_at", "batch_id", "source"}
    assert required.issubset(field_names)


def test_story_id_is_not_nullable():
    field = next(f for f in NEWS_RAW_SCHEMA.fields if f.name == "story_id")
    assert field.nullable is False


def test_schema_hints_is_non_empty_string():
    assert isinstance(NEWS_RAW_SCHEMA_HINTS, str)
    assert len(NEWS_RAW_SCHEMA_HINTS) > 0
    assert "story_id" in NEWS_RAW_SCHEMA_HINTS
