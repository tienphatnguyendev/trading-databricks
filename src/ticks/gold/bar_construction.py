"""Gold bar construction: silver.ticks_cleaned → gold.*_bars.

Implements four alternative bar types (Lopez de Prado, AFML Chapter 2):

  Dollar bars:           sample every N dollars of dollar_value
  Volume bars:           sample every N units of quantity
  Tick imbalance bars:   adaptive threshold on cumulative tick direction imbalance
  Dollar imbalance bars: adaptive threshold on cumulative dollar imbalance

Each bar table output schema:
    bar_index       LONG       sequential bar number
    bar_start_time  TIMESTAMP  timestamp of first tick in bar
    bar_end_time    TIMESTAMP  timestamp of last tick in bar
    open            DOUBLE     first price in bar
    high            DOUBLE     max price in bar
    low             DOUBLE     min price in bar
    close           DOUBLE     last price in bar
    volume          DOUBLE     sum of quantity in bar
    dollar_volume   DOUBLE     sum of dollar_value in bar
    tick_count      LONG       number of ticks in bar

Target tables:
    {catalog}.gold.dollar_bars
    {catalog}.gold.volume_bars
    {catalog}.gold.tick_imbalance_bars
    {catalog}.gold.dollar_imbalance_bars

TODO: implement build_dollar_bars(), build_volume_bars(),
      build_tick_imbalance_bars(), build_dollar_imbalance_bars(),
      run_gold_bars()
"""
from __future__ import annotations
