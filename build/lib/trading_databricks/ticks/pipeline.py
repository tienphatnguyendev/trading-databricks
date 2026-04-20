"""SDP pipeline definition for the ticks domain.

This file contains ONLY @dp.table / @dp.materialized_view decorators.
All transformation logic lives in the sibling modules and is locally testable.

Pipeline:
  bronze_ticks_raw        — streaming table via cloudFiles Auto Loader
  silver_ticks_cleaned    — streaming table with quality expectations
  gold_dollar_bars        — materialized view (full recompute, batch)
  gold_volume_bars        — materialized view
  gold_tick_imbalance_bars   — materialized view
  gold_dollar_imbalance_bars — materialized view

Gold bars use @dp.materialized_view (full recompute) because Lopez de Prado
cumulative bar algorithms require ordered stateful processing — inherently batch.

DABs pipeline resource: resources/tick_pipeline.pipeline.yml

STATUS: schedule commented out — awaiting tick-scraper-aws schema confirmation.
"""
from __future__ import annotations

from pyspark import pipelines as dp

from trading_databricks.common.config import (
    DollarBarConfig,
    DollarImbalanceBarConfig,
    TickImbalanceBarConfig,
    VolumeBarConfig,
)
from trading_databricks.ticks.bronze import enrich_bronze
from trading_databricks.ticks.expectations import (
    BRONZE_VALID_PRICE,
    BRONZE_VALID_QUANTITY,
    BRONZE_VALID_TIMESTAMP,
    BRONZE_VALID_TRADE_ID,
    SILVER_POSITIVE_PRICE,
    SILVER_POSITIVE_QUANTITY,
    SILVER_VALID_TIMESTAMP,
    SILVER_VALID_TRADE_ID,
)
from trading_databricks.ticks.gold import (
    build_dollar_bars,
    build_dollar_imbalance_bars,
    build_tick_imbalance_bars,
    build_volume_bars,
)
from trading_databricks.ticks.schemas import TICKS_RAW_SCHEMA_HINTS
from trading_databricks.ticks.silver import clean_ticks

# ── Pipeline configuration ────────────────────────────────────────────────────

bronze_schema = spark.conf.get("bronze_schema")  # noqa: F821
silver_schema = spark.conf.get("silver_schema")
gold_schema = spark.conf.get("gold_schema")
schema_location_base = spark.conf.get("schema_location_base")
source_bucket = spark.conf.get("source_bucket")

# Bar thresholds — TODO: tune after first gold tables are populated (TRD-5)
_dollar_bar_config = DollarBarConfig(threshold=1_000_000.0)
_volume_bar_config = VolumeBarConfig(threshold=10.0)
_tick_imbalance_config = TickImbalanceBarConfig()
_dollar_imbalance_config = DollarImbalanceBarConfig()

# ── Bronze ────────────────────────────────────────────────────────────────────


@dp.table(
    name=f"{bronze_schema}.ticks_raw",
    comment="Raw BTC-USDT tick records ingested from S3 via Auto Loader (append-only).",
    cluster_by=["timestamp"],
)
@dp.expect_or_drop("valid_trade_id", BRONZE_VALID_TRADE_ID)
@dp.expect_or_drop("valid_timestamp", BRONZE_VALID_TIMESTAMP)
@dp.expect_or_drop("valid_price", BRONZE_VALID_PRICE)
@dp.expect_or_drop("valid_quantity", BRONZE_VALID_QUANTITY)
def bronze_ticks_raw():
    return enrich_bronze(
        spark.readStream.format("cloudFiles")  # noqa: F821
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaHints", TICKS_RAW_SCHEMA_HINTS)
        .option("cloudFiles.schemaLocation", f"{schema_location_base}/bronze_ticks_raw")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("rescuedDataColumn", "_rescued_data")
        .load(
            f"s3://{source_bucket}/ticks/binance/symbol=*/year=*/month=*/day=*/"
        )
    )


# ── Silver ────────────────────────────────────────────────────────────────────


@dp.table(
    name=f"{silver_schema}.ticks_cleaned",
    comment="Type-cast, validated BTC-USDT ticks with dollar_value column.",
    cluster_by=["timestamp"],
)
@dp.expect_or_drop("valid_trade_id", SILVER_VALID_TRADE_ID)
@dp.expect_or_drop("valid_timestamp", SILVER_VALID_TIMESTAMP)
@dp.expect_or_drop("positive_price", SILVER_POSITIVE_PRICE)
@dp.expect_or_drop("positive_quantity", SILVER_POSITIVE_QUANTITY)
def silver_ticks_cleaned():
    return clean_ticks(
        spark.readStream.table(f"{bronze_schema}.ticks_raw")  # noqa: F821
    )


# ── Gold — bar materialized views (full recompute) ────────────────────────────


@dp.materialized_view(
    name=f"{gold_schema}.dollar_bars",
    comment="Dollar bars sampled every N dollars of dollar_value (Lopez de Prado).",
    cluster_by=["bar_start_time"],
)
def gold_dollar_bars():
    ticks = spark.read.table(f"{silver_schema}.ticks_cleaned")  # noqa: F821
    return build_dollar_bars(ticks, config=_dollar_bar_config)


@dp.materialized_view(
    name=f"{gold_schema}.volume_bars",
    comment="Volume bars sampled every N units of quantity (Lopez de Prado).",
    cluster_by=["bar_start_time"],
)
def gold_volume_bars():
    ticks = spark.read.table(f"{silver_schema}.ticks_cleaned")  # noqa: F821
    return build_volume_bars(ticks, config=_volume_bar_config)


@dp.materialized_view(
    name=f"{gold_schema}.tick_imbalance_bars",
    comment="Tick imbalance bars with adaptive threshold (Lopez de Prado).",
    cluster_by=["bar_start_time"],
)
def gold_tick_imbalance_bars():
    ticks = spark.read.table(f"{silver_schema}.ticks_cleaned")  # noqa: F821
    return build_tick_imbalance_bars(ticks, config=_tick_imbalance_config)


@dp.materialized_view(
    name=f"{gold_schema}.dollar_imbalance_bars",
    comment="Dollar imbalance bars with adaptive threshold (Lopez de Prado).",
    cluster_by=["bar_start_time"],
)
def gold_dollar_imbalance_bars():
    ticks = spark.read.table(f"{silver_schema}.ticks_cleaned")  # noqa: F821
    return build_dollar_imbalance_bars(ticks, config=_dollar_imbalance_config)
