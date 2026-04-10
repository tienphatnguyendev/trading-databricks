"""Tests for volume bar construction (stub verification)."""
from __future__ import annotations

import pytest
from pyspark.sql import Row

from trading_databricks.common.config import VolumeBarConfig


def test_build_volume_bars_functional(spark):
    from datetime import datetime, timedelta
    from decimal import Decimal
    from trading_databricks.ticks.gold.volume_bars import build_volume_bars

    # 1. Create mock data (2 bars for BTC-USDT, 1 bar for ETH-USDT)
    base_time = datetime(2025, 1, 1, 12, 0, 0)
    data = [
        # BTC-USDT Bar 0 (Total Vol: 6)
        Row(
            symbol="BTC-USDT", exchange="BINANCE", trade_id="1",
            timestamp=base_time, price=Decimal("50000.0"),
            quantity=Decimal("4.0"), dollar_value=Decimal("200000.0")
        ),
        Row(
            symbol="BTC-USDT", exchange="BINANCE", trade_id="2",
            timestamp=base_time + timedelta(seconds=1), price=Decimal("50100.0"),
            quantity=Decimal("2.0"), dollar_value=Decimal("100200.0")
        ),
        # BTC-USDT Bar 1 (Total Vol: 5)
        Row(
            symbol="BTC-USDT", exchange="BINANCE", trade_id="3",
            timestamp=base_time + timedelta(seconds=2), price=Decimal("50050.0"),
            quantity=Decimal("5.0"), dollar_value=Decimal("250250.0")
        ),
        # ETH-USDT Bar 0 (Total Vol: 8)
        Row(
            symbol="ETH-USDT", exchange="BINANCE", trade_id="4",
            timestamp=base_time, price=Decimal("3000.0"),
            quantity=Decimal("8.0"), dollar_value=Decimal("24000.0")
        ),
    ]
    ticks_df = spark.createDataFrame(data)

    # 2. Build volume bars with threshold 5.0.
    # BTC ticks: qty [4.0, 2.0, 5.0].
    # Tick 1: qty 4.0. Cum volume before tick: 0.0. Bar index: 0.
    # Tick 2: qty 2.0. Cum volume before tick: 4.0. Bar index: 0.
    # Tick 3: qty 5.0. Cum volume before tick: 6.0. Bar index: 1.
    # Result: BTC has 2 bars. Bar 0 (vol 6.0, ticks 1-2), Bar 1 (vol 5.0, tick 3).
    # ETH ticks: qty [8.0].
    # Tick 4: qty 8.0. Cum volume before tick: 0.0. Bar index: 0.
    # Result: ETH has 1 bar. Bar 0 (vol 8.0, tick 4).
    config = VolumeBarConfig(threshold=5.0)
    bars_df = build_volume_bars(ticks_df, config=config)
    results = bars_df.collect()

    # 3. Assertions
    assert len(results) == 3

    # BTC-USDT Bar 0
    btc_bar0 = next(r for r in results if r.symbol == "BTC-USDT" and r.bar_index == 0)
    assert btc_bar0.volume == Decimal("6.0")
    assert btc_bar0.tick_count == 2
    assert btc_bar0.open == Decimal("50000.0")
    assert btc_bar0.close == Decimal("50100.0")
    assert btc_bar0.high == Decimal("50100.0")
    assert btc_bar0.low == Decimal("50000.0")

    # BTC-USDT Bar 1
    btc_bar1 = next(r for r in results if r.symbol == "BTC-USDT" and r.bar_index == 1)
    assert btc_bar1.volume == Decimal("5.0")
    assert btc_bar1.tick_count == 1
    assert btc_bar1.open == Decimal("50050.0")
    assert btc_bar1.close == Decimal("50050.0")

    # ETH-USDT Bar 0
    eth_bar0 = next(r for r in results if r.symbol == "ETH-USDT" and r.bar_index == 0)
    assert eth_bar0.volume == Decimal("8.0")
    assert eth_bar0.tick_count == 1
