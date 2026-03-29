"""Silver transformation: bronze.news_raw → silver.news_cleaned.

Transformations:
1. Dedup by story_id (keep latest scraped_at)
2. Cast published_at and scraped_at to TIMESTAMP
3. Normalize whitespace in headline and body
4. Flatten provider JSON into typed columns
5. Add _ingested_at metadata column

Write mode: MERGE on story_id (Delta upsert).

TODO: implement clean_news(), run_silver_transform()
"""
from __future__ import annotations
