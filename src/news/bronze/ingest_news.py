"""Bronze ingestion: S3 NDJSON → bronze.news_raw via Auto Loader.

Reads news NDJSON files from S3 written by news-scraper-aws.
Uses trigger(availableNow=True) — processes all new files then terminates.
Scheduled every 15 minutes to match the EventBridge scrape cadence.

Source path: s3://{bucket}/bronze/tradingview/
Target table: {catalog}.bronze.news_raw

TODO: implement run_bronze_ingestion()
"""
from __future__ import annotations
