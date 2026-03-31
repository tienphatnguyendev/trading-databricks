"""Tick imbalance bar construction (Lopez de Prado, AFML Chapter 2).

Samples a new bar when the cumulative tick-direction imbalance exceeds an
adaptive threshold, estimated as:
    E[T] * |2 * P(b_t = 1) - 1|
where E[T] is the EWMA of previous bar lengths and P(b_t = 1) is the EWMA
of the fraction of up-ticks.

TODO: implement build_tick_imbalance_bars() after tick-scraper-aws is confirmed.
"""
from __future__ import annotations

from trading_databricks.common.config import TickImbalanceBarConfig


def build_tick_imbalance_bars(ticks_df, *, config: TickImbalanceBarConfig):
    """Build tick imbalance bars from a cleaned ticks DataFrame.

    Args:
        ticks_df: Spark DataFrame from silver.ticks_cleaned, ordered by timestamp.
        config:   TickImbalanceBarConfig with initial_expected_bar_size and ewma_span.

    Returns:
        Spark DataFrame conforming to BAR_SCHEMA with one row per bar.

    TODO: implement — adaptive threshold on cumulative tick direction imbalance.
    """
    raise NotImplementedError(
        "Tick imbalance bar construction not yet implemented — see TRD-5"
    )
