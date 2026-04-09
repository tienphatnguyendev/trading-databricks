import logging

import polars as pl
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def load_dollar_bars(
    spark: SparkSession,
    symbol: str = "BTCUSDT",
    catalog: str = "trading-signal-gen-dev",
    limit: int = 100_000,
) -> pl.DataFrame:
    """Safely loads dollar bars from gold into Polars for a specific symbol."""
    table_name = f"{catalog}.gold.dollar_bars"
    logger.info(f"Loading up to {limit} bars from {table_name} for {symbol}")

    df = (
        spark.table(table_name)
        .filter(f"symbol = '{symbol}'")
        .orderBy("bar_index")
        .limit(limit)
    )
    return pl.from_arrow(df.toArrow())


def load_raw_ticks(
    spark: SparkSession,
    symbol: str = "BTCUSDT",
    catalog: str = "trading-signal-gen-dev",
    limit: int = 500_000,
) -> pl.DataFrame:
    """Safely loads raw ticks for dollar bar construction QA."""
    table_name = f"{catalog}.bronze.raw_aggtrades_samples"
    logger.info(f"Loading up to {limit} ticks from {table_name} for {symbol}")

    df = (
        spark.table(table_name)
        .filter(f"symbol = '{symbol}'")
        .orderBy("timestamp")
        .limit(limit)
    )
    return pl.from_arrow(df.toArrow())


def load_training_data(
    spark: SparkSession,
    symbol: str = "BTCUSDT",
    catalog: str = "smart-trading-dev",
    limit: int = 100_000,
) -> pl.DataFrame:
    """Loads ml_training_data for baseline comparisons."""
    table_name = f"{catalog}.others.ml_training_data"
    logger.info(f"Loading up to {limit} training samples from {table_name} for {symbol}")

    df = (
        spark.table(table_name)
        .filter(f"symbol = '{symbol}'")
        .orderBy("event_ts")
        .limit(limit)
    )
    return pl.from_arrow(df.toArrow())
