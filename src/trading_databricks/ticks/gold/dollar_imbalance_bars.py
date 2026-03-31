"""Dollar imbalance bar construction (Lopez de Prado, AFML Chapter 2).

Like tick imbalance bars but the imbalance is measured in dollars rather
than raw tick direction:
    b_t * dollar_value_t
where b_t ∈ {-1, +1} is the tick direction (signed by side).

TODO: implement build_dollar_imbalance_bars() after tick-scraper-aws is confirmed.
"""
from __future__ import annotations

from trading_databricks.common.config import DollarImbalanceBarConfig


def build_dollar_imbalance_bars(ticks_df, *, config: DollarImbalanceBarConfig):
    """Build dollar imbalance bars from a cleaned ticks DataFrame.

    Args:
        ticks_df: Spark DataFrame from silver.ticks_cleaned, ordered by timestamp.
        config:   DollarImbalanceBarConfig with initial_expected_bar_size and ewma_span.

    Returns:
        Spark DataFrame conforming to BAR_SCHEMA with one row per bar.

    TODO: implement — adaptive threshold on cumulative dollar imbalance.
    """
    raise NotImplementedError(
        "Dollar imbalance bar construction not yet implemented — see TRD-5"
    )
