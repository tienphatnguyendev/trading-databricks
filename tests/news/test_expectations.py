"""Tests for news domain SDP expectation rule strings."""
from __future__ import annotations

from trading_databricks.news.expectations import (
    BRONZE_VALID_BATCH_ID,
    BRONZE_VALID_SCRAPED_AT,
    BRONZE_VALID_STORY_ID,
    SILVER_VALID_HEADLINE,
    SILVER_VALID_STORY_ID,
)


def test_bronze_valid_story_id_is_not_null():
    assert "story_id" in BRONZE_VALID_STORY_ID
    assert "IS NOT NULL" in BRONZE_VALID_STORY_ID


def test_bronze_valid_scraped_at_is_not_null():
    assert "scraped_at" in BRONZE_VALID_SCRAPED_AT
    assert "IS NOT NULL" in BRONZE_VALID_SCRAPED_AT


def test_bronze_valid_batch_id_is_not_null():
    assert "batch_id" in BRONZE_VALID_BATCH_ID
    assert "IS NOT NULL" in BRONZE_VALID_BATCH_ID


def test_silver_valid_story_id_non_empty():
    assert "story_id" in SILVER_VALID_STORY_ID


def test_silver_valid_headline_not_null():
    assert "headline" in SILVER_VALID_HEADLINE
