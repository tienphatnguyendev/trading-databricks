"""Tests for volume bar construction (stub verification)."""
from __future__ import annotations

import pytest
from pyspark.sql import Row

from trading_databricks.common.config import VolumeBarConfig


def test_build_volume_bars_raises_not_implemented(spark):
    from trading_databricks.ticks.gold.volume_bars import build_volume_bars

    df = spark.createDataFrame([Row(bar_index=1)])
    with pytest.raises(NotImplementedError):
        build_volume_bars(df, config=VolumeBarConfig(threshold=10.0))
