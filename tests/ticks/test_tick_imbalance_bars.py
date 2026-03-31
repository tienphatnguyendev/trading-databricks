"""Tests for tick imbalance bar construction (stub verification)."""
from __future__ import annotations

import pytest
from pyspark.sql import Row

from trading_databricks.common.config import TickImbalanceBarConfig


def test_build_tick_imbalance_bars_raises_not_implemented(spark):
    from trading_databricks.ticks.gold.tick_imbalance_bars import build_tick_imbalance_bars

    df = spark.createDataFrame([Row(bar_index=1)])
    with pytest.raises(NotImplementedError):
        build_tick_imbalance_bars(df, config=TickImbalanceBarConfig())
