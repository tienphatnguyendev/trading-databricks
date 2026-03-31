"""Dollar bar construction (Lopez de Prado, AFML Chapter 2).

Samples a new bar every time the cumulative dollar_value (price × quantity)
traded crosses a fixed threshold N.

Pure function — operates on a pandas DataFrame (via .toPandas()) for
stateful cumulative logic, then returns a Spark DataFrame.

TODO: implement build_dollar_bars() after tick-scraper-aws schema is confirmed.
"""
from __future__ import annotations

from trading_databricks.common.config import DollarBarConfig


def build_dollar_bars(ticks_df, *, config: DollarBarConfig):
    """Build dollar bars from a cleaned ticks DataFrame.

    Args:
        ticks_df: Spark DataFrame from silver.ticks_cleaned, ordered by timestamp.
        config:   DollarBarConfig specifying the dollar threshold per bar.

    Returns:
        Spark DataFrame conforming to BAR_SCHEMA with one row per bar.

    TODO: implement — sample every config.threshold dollars of dollar_value.
    """
    raise NotImplementedError("Dollar bar construction not yet implemented — see TRD-5")
