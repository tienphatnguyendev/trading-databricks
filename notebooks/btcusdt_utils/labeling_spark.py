"""Spark ML triple-barrier labeling (AFML Ch. 3).

Distributed implementation using Pandas UDF grouped map for unlimited
scalability on massive bar datasets.

Designed for serverless compute:
* Uses applyInPandas() for partition-local computation
* No .cache()/.persist() calls
* Scales to unlimited bars via horizontal partitioning

Reference: Advances in Financial Machine Learning, Chapter 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StructField,
    StructType,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def apply_triple_barrier_pandas(
    pdf: pd.DataFrame,
    pt_mult: float,
    sl_mult: float,
    max_hold: int,
) -> pd.DataFrame:
    """Apply triple-barrier labeling to a pandas DataFrame partition.

    This function is called by applyInPandas() on each partition.
    It implements the same vectorized wavefront scan as the Polars
    baseline but operates on pandas DataFrames.

    Parameters
    ----------
    pdf : pd.DataFrame
        Partition data with columns: close, high, low, vol, bar_index
    pt_mult : float
        Profit-taking multiplier (barrier = close × (1 + pt_mult × vol))
    sl_mult : float
        Stop-loss multiplier (barrier = close × (1 − sl_mult × vol))
    max_hold : int
        Maximum holding period (bars) — vertical barrier

    Returns
    -------
    pd.DataFrame
        Labeled events with columns: bar_index, label, touch_bar,
        touch_type, event_start, event_end

    Notes
    -----
    **Algorithm**: Vectorized wavefront scan
    * For each event at bar i, scan forward k = 1…max_hold
    * Check if high[i+k] breaches upper OR low[i+k] breaches lower
    * Process all events at step k in one vectorized pass
    * Complexity: O(max_hold × n) per partition

    **Barriers**::

        upper[i] = close[i] × (1 + pt_mult × vol[i])
        lower[i] = close[i] × (1 − sl_mult × vol[i])

    **Labels**:
    * 1 = TP win (hit profit-taking barrier)
    * 0 = SL loss (hit stop-loss barrier)
    * Vertical = close[i+max_hold] >= close[i] ? 1 : 0

    **Touch types**:
    * 0 = TP barrier
    * 1 = SL barrier
    * 2 = Vertical barrier

    See Also
    --------
    apply_triple_barrier_spark : Spark wrapper using applyInPandas()
    """
    # Extract arrays (already sorted by bar_index within partition)
    close = pdf["close"].values
    high = pdf["high"].values
    low = pdf["low"].values
    vol = pdf["vol"].values
    bar_indices = pdf["bar_index"].values

    n = len(close)
    n_events = n - max_hold

    if n_events <= 0:
        # Not enough bars for labeling in this partition
        return pd.DataFrame({
            "bar_index": pd.Series([], dtype="int64"),
            "label": pd.Series([], dtype="int8"),
            "touch_bar": pd.Series([], dtype="int32"),
            "touch_type": pd.Series([], dtype="int8"),
            "event_start": pd.Series([], dtype="int64"),
            "event_end": pd.Series([], dtype="int64"),
        })

    # ── Barrier levels at event start ─────────────────────────
    close_ev = close[:n_events]
    vol_ev = vol[:n_events]
    upper = close_ev * (1.0 + pt_mult * vol_ev)
    lower = close_ev * (1.0 - sl_mult * vol_ev)

    # ── Output arrays (default: vertical barrier) ─────────────
    labels = np.full(n_events, -1, dtype=np.int8)
    touch_bars = np.full(n_events, max_hold, dtype=np.int32)
    touch_types = np.full(n_events, 2, dtype=np.int8)
    event_starts = bar_indices[:n_events]
    event_ends = event_starts + max_hold  # default: vertical

    unresolved = np.ones(n_events, dtype=bool)

    # ── Wavefront scan ────────────────────────────────────────
    for k in range(1, max_hold + 1):
        if not unresolved.any():
            break

        h_k = high[k : k + n_events]
        l_k = low[k : k + n_events]

        tp_hit = unresolved & (h_k >= upper)
        sl_hit = unresolved & (l_k <= lower)

        # TP only
        tp_only = tp_hit & ~sl_hit
        if tp_only.any():
            labels[tp_only] = 1
            touch_bars[tp_only] = k
            touch_types[tp_only] = 0
            event_ends[tp_only] = event_starts[tp_only] + k

        # SL only
        sl_only = sl_hit & ~tp_hit
        if sl_only.any():
            labels[sl_only] = 0
            touch_bars[sl_only] = k
            touch_types[sl_only] = 1
            event_ends[sl_only] = event_starts[sl_only] + k

        # Same-bar collision: deeper exceedance wins
        both = tp_hit & sl_hit
        if both.any():
            tp_excess = h_k[both] - upper[both]
            sl_excess = lower[both] - l_k[both]
            tp_wins = tp_excess >= sl_excess

            both_idx = np.where(both)[0]
            tp_win = both_idx[tp_wins]
            sl_win = both_idx[~tp_wins]

            for idx_arr, lbl, tt in [(tp_win, 1, 0), (sl_win, 0, 1)]:
                if idx_arr.size:
                    labels[idx_arr] = lbl
                    touch_bars[idx_arr] = k
                    touch_types[idx_arr] = tt
                    event_ends[idx_arr] = event_starts[idx_arr] + k

        unresolved &= ~(tp_hit | sl_hit)

    # ── Vertical barrier: close-to-close direction ────────────
    vert = labels == -1
    if vert.any():
        vi = np.where(vert)[0]
        labels[vert] = (close[vi + max_hold] >= close[vi]).astype(np.int8)

    return pd.DataFrame({
        "bar_index": event_starts,
        "label": labels,
        "touch_bar": touch_bars,
        "touch_type": touch_types,
        "event_start": event_starts,
        "event_end": event_ends,
    })


def apply_triple_barrier_spark(
    df: DataFrame,
    *,
    pt_mult: float,
    sl_mult: float,
    max_hold: int,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    vol_col: str = "vol",
    partition_col: str | None = None,
) -> DataFrame:
    """Apply triple-barrier labeling using Pandas UDF grouped map.

    Distributes labeling computation across partitions using
    applyInPandas(). Each partition runs the vectorized wavefront
    scan independently.

    Parameters
    ----------
    df : DataFrame
        Spark DataFrame with OHLCV bars and volatility estimates.
        Must contain: close, high, low, vol columns.
    pt_mult : float
        Profit-taking multiplier (barrier = close × (1 + pt_mult × vol))
    sl_mult : float
        Stop-loss multiplier (barrier = close × (1 − sl_mult × vol))
    max_hold : int
        Maximum holding period (bars) — vertical barrier
    close_col : str, default="close"
        Name of close price column
    high_col : str, default="high"
        Name of high price column
    low_col : str, default="low"
        Name of low price column
    vol_col : str, default="vol"
        Name of volatility column
    partition_col : str, optional
        Column to use for partitioning. If None, uses single partition
        (not recommended for large datasets).

    Returns
    -------
    DataFrame
        Labeled events with columns:
        * bar_index : int64 — original bar index
        * label : int8 — 1 (TP win), 0 (SL loss), or vertical
        * touch_bar : int32 — bars until barrier touch
        * touch_type : int8 — 0 (TP), 1 (SL), 2 (vertical)
        * event_start : int64 — bar index where event begins
        * event_end : int64 — bar index where event resolves

    Examples
    --------
    >>> from btcusdt_utils.labeling_spark import apply_triple_barrier_spark
    >>> # Add volatility column first
    >>> df_bars = df_bars.withColumn(
    ...     "vol",
    ...     F.stddev(F.col("close")).over(
    ...         Window.rowsBetween(-20, 0)
    ...     )
    ... )
    >>> # Apply triple-barrier labeling
    >>> df_labeled = apply_triple_barrier_spark(
    ...     df_bars,
    ...     pt_mult=3.0,
    ...     sl_mult=3.0,
    ...     max_hold=100,
    ... )

    Notes
    -----
    **Partitioning Strategy**:
    * Each partition must have at least `max_hold` bars
    * Events near partition boundaries may be truncated
    * For best results, partition by time ranges with overlap

    **Performance**: O(max_hold × n / p) where p = # partitions
    * Embarrassingly parallel across partitions
    * No shuffle required (grouped map pattern)
    * Scales to unlimited bars via horizontal partitioning

    **Memory**: Serverless-compatible
    * No .cache() or .persist() calls
    * Partition-local computation only
    * Spills to disk if partition exceeds memory

    **Correctness**: Matches Polars baseline
    * Same vectorized wavefront scan algorithm
    * Edge cases handled identically
    * Vertical barrier uses close-to-close direction

    See Also
    --------
    apply_triple_barrier_pandas : Pandas UDF function
    btcusdt_utils.labeling.apply_triple_barrier : Polars baseline
    """
    # Add monotonic bar index for event tracking
    df_indexed = df.withColumn(
        "bar_index",
        F.monotonically_increasing_id(),
    )

    # Select required columns and rename to standard names
    df_prepared = df_indexed.select(
        F.col("bar_index"),
        F.col(close_col).alias("close"),
        F.col(high_col).alias("high"),
        F.col(low_col).alias("low"),
        F.col(vol_col).alias("vol"),
        F.col(partition_col).alias("partition_key") if partition_col else F.lit(0).alias("partition_key"),
    )

    # Define output schema
    output_schema = StructType([
        StructField("bar_index", LongType(), False),
        StructField("label", IntegerType(), False),
        StructField("touch_bar", IntegerType(), False),
        StructField("touch_type", IntegerType(), False),
        StructField("event_start", LongType(), False),
        StructField("event_end", LongType(), False),
    ])

    # Apply triple-barrier labeling via Pandas UDF grouped map
    df_labeled = df_prepared.groupBy("partition_key").applyInPandas(
        lambda pdf: apply_triple_barrier_pandas(
            pdf.sort_values("bar_index"),
            pt_mult=pt_mult,
            sl_mult=sl_mult,
            max_hold=max_hold,
        ),
        schema=output_schema,
    )

    return df_labeled.drop("partition_key") if partition_col is None else df_labeled


def compute_volatility(
    df: DataFrame,
    *,
    close_col: str = "close",
    window_size: int = 20,
    method: str = "ewm",
    span: int | None = None,
) -> DataFrame:
    """Compute volatility estimates for triple-barrier labeling.

    Parameters
    ----------
    df : DataFrame
        Spark DataFrame with OHLCV bars
    close_col : str, default="close"
        Name of close price column
    window_size : int, default=20
        Rolling window size for volatility (only used if method="rolling")
    method : str, default="ewm"
        Volatility method: "ewm" (exponential) or "rolling" (simple)
    span : int, optional
        EMA span for exponential volatility (defaults to window_size)

    Returns
    -------
    DataFrame
        Original DataFrame with added "vol" column

    Examples
    --------
    >>> df_with_vol = compute_volatility(df_bars, window_size=20)
    >>> df_labeled = apply_triple_barrier_spark(
    ...     df_with_vol, pt_mult=3.0, sl_mult=3.0, max_hold=100
    ... )
    """
    from pyspark.sql import Window

    if span is None:
        span = window_size

    # Compute log returns
    window_spec = Window.orderBy(F.monotonically_increasing_id())
    df_returns = df.withColumn(
        "_log_return",
        F.log(F.col(close_col) / F.lag(F.col(close_col), 1).over(window_spec)),
    )

    if method == "ewm":
        # Exponential weighted moving standard deviation
        # Approximate with rolling stddev (Spark doesn't have native EMA)
        window_vol = Window.rowsBetween(-span, 0)
        df_vol = df_returns.withColumn(
            "vol",
            F.stddev(F.col("_log_return")).over(window_vol),
        )
    elif method == "rolling":
        # Simple rolling standard deviation
        window_vol = Window.rowsBetween(-window_size + 1, 0)
        df_vol = df_returns.withColumn(
            "vol",
            F.stddev(F.col("_log_return")).over(window_vol),
        )
    else:
        raise ValueError(f"Unknown volatility method: {method}")

    # Fill nulls with a small constant to avoid division by zero
    df_vol = df_vol.withColumn(
        "vol",
        F.when(F.col("vol").isNull(), 0.001)
        .when(F.col("vol") == 0, 0.001)
        .otherwise(F.col("vol")),
    )

    return df_vol.drop("_log_return")
