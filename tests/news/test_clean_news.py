"""Tests for silver news transformation.

TODO: implement tests after src/news/silver/clean_news.py is implemented.

Planned tests:
- test_deduplicate_keeps_latest_scraped_at
- test_cast_timestamps_converts_published_at
- test_cast_timestamps_null_published_at_stays_null
- test_normalize_text_trims_whitespace
- test_normalize_text_collapses_internal_whitespace
- test_normalize_text_null_body_stays_null
- test_add_ingested_at_adds_column
- test_clean_news_full_pipeline
"""
from __future__ import annotations
