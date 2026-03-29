"""Bronze ingestion: S3 NDJSON → bronze.ticks_raw via Auto Loader.

Reads BTC-USDT tick NDJSON files written by tick-scraper-aws (not yet built).

Expected tick record schema (minimum):
    timestamp:  exchange timestamp, ISO 8601, microsecond precision
    price:      trade price as decimal string
    quantity:   trade quantity as decimal string
    side:       "buy" or "sell"
    trade_id:   exchange-assigned trade ID
    symbol:     e.g. "BTCUSDT"
    exchange:   e.g. "binance"
    batch_id:   batch identifier from tick-scraper-aws

Source path: s3://{bucket}/bronze/ticks/{symbol}/
Target table: {catalog}.bronze.ticks_raw

STATUS: awaiting tick-scraper-aws schema definition.
TODO: implement run_bronze_ingestion() after tick-scraper-aws is built
"""
from __future__ import annotations
