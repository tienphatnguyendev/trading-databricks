"""Tests for dollar imbalance bar construction (stub verification)."""
from __future__ import annotations

import pytest
from pyspark.sql import Row

from trading_databricks.common.config import DollarImbalanceBarConfig


def test_build_dollar_imbalance_bars_raises_not_implemented(spark):
    from trading_databricks.ticks.gold.dollar_imbalance_bars import build_dollar_imbalance_bars

    df = spark.createDataFrame([Row(bar_index=1)])
    with pytest.raises(NotImplementedError):
        build_dollar_imbalance_bars(df, config=DollarImbalanceBarConfig())
