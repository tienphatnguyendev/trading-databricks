"""Tests for dollar bar construction.

Bar construction functions are stubs (NotImplementedError) until
tick-scraper-aws schema is confirmed in TRD-5.
"""
from __future__ import annotations

import pytest
from pyspark.sql import Row

from trading_databricks.common.config import DollarBarConfig


def test_build_dollar_bars_raises_not_implemented(spark):
    from trading_databricks.ticks.gold.dollar_bars import build_dollar_bars

    df = spark.createDataFrame([Row(bar_index=1)])
    with pytest.raises(NotImplementedError):
        build_dollar_bars(df, config=DollarBarConfig(threshold=1_000_000.0))
