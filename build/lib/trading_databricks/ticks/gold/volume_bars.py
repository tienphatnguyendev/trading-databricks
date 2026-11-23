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

    TODO: implement — sample every config.threshold units of quantity.
    """
    raise NotImplementedError("Volume bar construction not yet implemented — see TRD-5")
