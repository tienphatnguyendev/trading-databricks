"""SDP data quality expectation rules for the news domain.

Each constant is a plain SQL expression string consumed by
@dp.expect_or_drop or @dp.expect_or_fail decorators in pipeline.py.
Centralising them here allows unit tests to validate expressions
independently of the SDP runtime.
"""
from __future__ import annotations

from trading_databricks.common.quality import not_null, non_empty_string

# ── Bronze expectations ───────────────────────────────────────────────────────

BRONZE_VALID_STORY_ID = not_null("story_id")
BRONZE_VALID_SCRAPED_AT = not_null("scraped_at")
BRONZE_VALID_BATCH_ID = not_null("batch_id")

# ── Silver expectations ───────────────────────────────────────────────────────

SILVER_VALID_STORY_ID = non_empty_string("story_id")
SILVER_VALID_HEADLINE = not_null("headline")
