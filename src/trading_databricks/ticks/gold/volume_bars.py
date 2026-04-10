"""Volume bar construction (Lopez de Prado, AFML Chapter 2).

Samples a new bar every time the cumulative quantity traded crosses a
fixed threshold N.

TODO: implement build_volume_bars() after tick-scraper-aws schema is confirmed.
"""
from __future__ import annotations

from trading_databricks.common.config import VolumeBarConfig


def build_volume_bars(ticks_df, *, config: VolumeBarConfig):
    """Build volume bars from a cleaned ticks DataFrame.

    Args:
        ticks_df: Spark DataFrame from silver.ticks_cleaned, ordered by timestamp.
        config:   VolumeBarConfig specifying the volume threshold per bar.

    Returns:
        Spark DataFrame conforming to BAR_SCHEMA with one row per bar.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    # 1. Define window for cumulative volume calculation.
    # Partition by symbol/exchange and order by timestamp/trade_id for determinism.
    window_spec = Window.partitionBy("symbol", "exchange").orderBy("timestamp", "trade_id")

    # 2. Calculate cumulative quantity and assign bar indices.
    # The bar_index is determined by the cumulative volume before the current tick.
    # This ensures each tick is assigned to the bar where it began.
    df_with_index = (
        ticks_df
        .withColumn("cum_vol", F.sum("quantity").over(window_spec))
        .withColumn(
            "bar_index",
            F.floor((F.col("cum_vol") - F.col("quantity")) / config.threshold).cast("long")
        )
    )

    # 3. Aggregate ticks into bars.
    # We use min_by/max_by for Open and Close to handle ticks at the same millisecond.
    bars_df = (
        df_with_index
        .groupBy("symbol", "exchange", "bar_index")
        .agg(
            F.min("timestamp").alias("bar_start_time"),
            F.max("timestamp").alias("bar_end_time"),
            F.expr("min_by(price, struct(timestamp, trade_id))").alias("open"),
            F.max("price").alias("high"),
            F.min("price").alias("low"),
            F.expr("max_by(price, struct(timestamp, trade_id))").alias("close"),
            F.sum("quantity").alias("volume"),
            F.sum("dollar_value").alias("dollar_volume"),
            F.count("*").alias("tick_count")
        )
    )

    return bars_df
