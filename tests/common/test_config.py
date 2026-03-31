"""Tests for trading_databricks.common.config dataclasses."""
from __future__ import annotations

import pytest

from trading_databricks.common.config import (
    DollarBarConfig,
    DollarImbalanceBarConfig,
    TickImbalanceBarConfig,
    VolumeBarConfig,
)


def test_dollar_bar_config_immutable():
    cfg = DollarBarConfig(threshold=1_000_000.0)
    with pytest.raises(Exception):
        cfg.threshold = 999.0  # type: ignore[misc]


def test_volume_bar_config_stores_threshold():
    cfg = VolumeBarConfig(threshold=10.0)
    assert cfg.threshold == 10.0


def test_tick_imbalance_config_defaults():
    cfg = TickImbalanceBarConfig()
    assert cfg.initial_expected_bar_size == 100
    assert cfg.ewma_span == 10


def test_dollar_imbalance_config_defaults():
    cfg = DollarImbalanceBarConfig()
    assert cfg.initial_expected_bar_size == 100
    assert cfg.ewma_span == 10
