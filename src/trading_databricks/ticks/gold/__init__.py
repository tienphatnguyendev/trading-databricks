"""Gold bar construction sub-package for the ticks domain.

Re-exports the four build_* entry points so callers can import from
trading_databricks.ticks.gold directly.
"""
from __future__ import annotations

from trading_databricks.ticks.gold.dollar_bars import build_dollar_bars
from trading_databricks.ticks.gold.dollar_imbalance_bars import build_dollar_imbalance_bars
from trading_databricks.ticks.gold.tick_imbalance_bars import build_tick_imbalance_bars
from trading_databricks.ticks.gold.volume_bars import build_volume_bars

__all__ = [
    "build_dollar_bars",
    "build_volume_bars",
    "build_tick_imbalance_bars",
    "build_dollar_imbalance_bars",
]
