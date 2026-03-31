"""StructType definitions for the news domain.

These schemas reflect the NDJSON contract produced by news-scraper-aws.
They are used as schemaHints in Auto Loader and for test assertions.
"""
from __future__ import annotations

from pyspark.sql.types import StringType, StructField, StructType

# Minimum fields present in every news NDJSON record written by news-scraper-aws.
# Auto Loader infers additional columns; _rescued_data and _metadata are added
# by the cloudFiles reader automatically.
NEWS_RAW_SCHEMA = StructType(
    [
        StructField("story_id", StringType(), nullable=False),
        StructField("headline", StringType(), nullable=True),
        StructField("url", StringType(), nullable=True),
        StructField("published_at", StringType(), nullable=True),  # cast in silver
        StructField("source", StringType(), nullable=True),
        StructField("source_feed", StringType(), nullable=True),
        StructField("body", StringType(), nullable=True),
        StructField("ast_description_raw", StringType(), nullable=True),  # JSON string
        StructField("provider", StringType(), nullable=True),  # JSON string, flattened in silver
        StructField("scraped_at", StringType(), nullable=False),  # cast in silver
        StructField("fetch_method", StringType(), nullable=True),
        StructField("batch_id", StringType(), nullable=False),
    ]
)

# schemaHints string for cloudFiles Auto Loader (subset of critical typed fields)
NEWS_RAW_SCHEMA_HINTS = (
    "story_id STRING, headline STRING, url STRING, "
    "published_at STRING, source STRING, source_feed STRING, "
    "body STRING, ast_description_raw STRING, provider STRING, "
    "scraped_at STRING, fetch_method STRING, batch_id STRING"
)
