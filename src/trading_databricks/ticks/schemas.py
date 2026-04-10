"""StructType definitions for the ticks domain.

Schema reflects the minimum NDJSON contract expected from tick-scraper-aws.
STATUS: awaiting tick-scraper-aws schema confirmation.
"""
from __future__ import annotations

from pyspark.sql.types import (
    DecimalType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# Minimum fields expected in every tick NDJSON record.
TICKS_RAW_SCHEMA = StructType(
    [
        StructField("timestamp", StringType(), nullable=False),   # cast in silver
        StructField("price", StringType(), nullable=False),       # cast to DECIMAL in silver
        StructField("quantity", StringType(), nullable=False),    # cast to DECIMAL in silver
        StructField("side", StringType(), nullable=True),         # "buy" | "sell"
        StructField("trade_id", StringType(), nullable=False),
        StructField("symbol", StringType(), nullable=False),
        StructField("exchange", StringType(), nullable=False),
        StructField("batch_id", StringType(), nullable=False),
    ]
)

TICKS_RAW_SCHEMA_HINTS = (
    "timestamp STRING, price STRING, quantity STRING, "
    "side STRING, trade_id STRING, symbol STRING, "
    "exchange STRING, batch_id STRING"
)

# Silver cleaned schema (after type casting and dollar_value computation)
TICKS_CLEANED_SCHEMA = StructType(
    [
        StructField("timestamp", TimestampType(), nullable=False),
        StructField("price", DecimalType(18, 8), nullable=False),
        StructField("quantity", DecimalType(18, 8), nullable=False),
        StructField("dollar_value", DecimalType(38, 8), nullable=False),
        StructField("side", StringType(), nullable=True),
        StructField("trade_id", StringType(), nullable=False),
        StructField("symbol", StringType(), nullable=False),
        StructField("exchange", StringType(), nullable=False),
        StructField("batch_id", StringType(), nullable=False),
        StructField("_ingested_at", TimestampType(), nullable=False),
    ]
)

# Common output schema for all bar types
BAR_SCHEMA = StructType(
    [
        StructField("bar_index", LongType(), nullable=False),
        StructField("bar_start_time", TimestampType(), nullable=False),
        StructField("bar_end_time", TimestampType(), nullable=False),
        StructField("open", DecimalType(18, 8), nullable=False),
        StructField("high", DecimalType(18, 8), nullable=False),
        StructField("low", DecimalType(18, 8), nullable=False),
        StructField("close", DecimalType(18, 8), nullable=False),
        StructField("volume", DecimalType(18, 8), nullable=False),
        StructField("dollar_volume", DecimalType(38, 8), nullable=False),
        StructField("tick_count", LongType(), nullable=False),
    ]
)
