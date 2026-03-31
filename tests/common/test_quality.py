"""Tests for trading_databricks.common.quality expectation helpers."""
from __future__ import annotations

from trading_databricks.common.quality import non_empty_string, not_null, positive


def test_not_null_expression():
    assert not_null("story_id") == "story_id IS NOT NULL"


def test_positive_expression():
    assert positive("price") == "price > 0"


def test_non_empty_string_expression():
    expr = non_empty_string("headline")
    assert "IS NOT NULL" in expr
    assert "LENGTH" in expr
    assert "TRIM" in expr
    assert "headline" in expr
