"""End-to-end Spark ML feature engineering pipelines.

Provides convenience functions for chaining:
  1. Bar construction (time/threshold bars)
  2. Technical indicator computation
  3. Feature assembly (optional)
  4. Triple-barrier labeling (optional)

Pipelines are reusable for both training and inference and support
serverless compute (no .cache()/.persist() calls).

Examples
--------
>>> # Time bars + technical indicators
>>> from btcusdt_utils.features.pipeline_spark import build_feature_pipeline
>>> from btcusdt_utils.features._config import TechnicalConfig
>>> pipeline = build_feature_pipeline(
...     bar_type="time",
...     interval_minutes=5,
...     config=TechnicalConfig(),
... )
>>> df_features = pipeline.fit(df_ticks).transform(df_ticks)
>>>
>>> # Complete ML pipeline with labeling
>>> from btcusdt_utils.features.pipeline_spark import build_ml_pipeline
>>> df_ml_ready = build_ml_pipeline(
...     df_bars,  # Pre-built bars with OHLCV
...     config=TechnicalConfig(),
...     pt_mult=3.0,
...     sl_mult=3.0,
...     max_hold=100,
... )
>>> # df_ml_ready now has features + labels
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F
from pyspark.sql.window import Window

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

from btcusdt_utils.bars_spark import (
    ThresholdBarTransformer,
    TimeBarTransformer,
)
from btcusdt_utils.features._config import TechnicalConfig
from btcusdt_utils.features.technical_spark import (
    TechnicalIndicatorsTransformer,
)


def build_feature_pipeline(
    bar_type: Literal["time", "volume", "dollar", "tick"],
    *,
    interval_minutes: int | None = None,
    threshold: float | None = None,
    config: TechnicalConfig | None = None,
    time_col: str = "bucket",
    assemble_features: bool = False,
    feature_col: str = "features",
) -> Pipeline:
    """Build end-to-end feature engineering pipeline.

    Creates a Spark ML Pipeline that chains:
      1. Bar construction (time or threshold bars)
      2. Technical indicator computation
      3. Feature assembly (optional)

    Parameters
    ----------
    bar_type : {"time", "volume", "dollar", "tick"}
        Type of bars to construct.
    interval_minutes : int, optional
        For time bars: aggregation interval in minutes.
        Required if bar_type="time".
    threshold : float, optional
        For threshold bars: threshold value.
        Required if bar_type in {"volume", "dollar", "tick"}.
    config : TechnicalConfig, optional
        Technical indicator configuration. Defaults to TechnicalConfig().
    time_col : str, default="bucket"
        Name of datetime column for cyclical features.
    assemble_features : bool, default=False
        If True, adds VectorAssembler to combine all features into
        a single "features" vector column.
    feature_col : str, default="features"
        Name of the output vector column (only used if assemble_features=True).

    Returns
    -------
    Pipeline
        Spark ML Pipeline with stages:
          [BarTransformer, TechnicalIndicatorsTransformer, (VectorAssembler)]

    Raises
    ------
    ValueError
        If required parameters are missing for the specified bar_type.

    Examples
    --------
    >>> # Time bars pipeline
    >>> pipeline = build_feature_pipeline(
    ...     bar_type="time",
    ...     interval_minutes=5,
    ... )
    >>> df_features = pipeline.fit(df_ticks).transform(df_ticks)
    >>>
    >>> # Volume bars with feature assembly
    >>> pipeline = build_feature_pipeline(
    ...     bar_type="volume",
    ...     threshold=1000,
    ...     assemble_features=True,
    ... )
    >>> df_features = pipeline.fit(df_ticks).transform(df_ticks)
    >>> # Now df_features has a "features" vector column

    Notes
    -----
    **Performance**: Serverless-compatible
    * No .cache() or .persist() calls
    * All transformations are lazy (only materialized on action)
    * Scales to unlimited data via partitioning

    **Reusability**: Fit once, transform many
    * fit() computes any necessary statistics (if needed)
    * transform() applies the pipeline to new data
    * Same pipeline can be used for training and inference

    See Also
    --------
    btcusdt_utils.bars_spark : Bar construction transformers
    btcusdt_utils.features.technical_spark : Technical indicators
    build_ml_pipeline : Complete pipeline with labeling
    """
    if config is None:
        config = TechnicalConfig()

    stages = []

    # Stage 1: Bar Construction
    if bar_type == "time":
        if interval_minutes is None:
            raise ValueError("interval_minutes required for time bars")
        bar_transformer = TimeBarTransformer(
            intervalMinutes=interval_minutes,
            timestampCol="timestamp",
            priceCol="price",
            qtyCol="qty",
        )
    elif bar_type in ("volume", "dollar", "tick"):
        if threshold is None:
            raise ValueError(f"threshold required for {bar_type} bars")
        bar_transformer = ThresholdBarTransformer(
            threshold=threshold,
            barType=bar_type,
            timestampCol="timestamp",
            priceCol="price",
            qtyCol="qty",
        )
    else:
        raise ValueError(f"Invalid bar_type: {bar_type}. Must be one of: time, volume, dollar, tick")

    stages.append(bar_transformer)

    # Stage 2: Technical Indicators
    tech_transformer = TechnicalIndicatorsTransformer(
        config=config,
        timeCol=time_col,
    )
    stages.append(tech_transformer)

    # Stage 3: Feature Assembly (optional)
    if assemble_features:
        # Get all feature column names from config
        feature_cols = _get_feature_column_names(config, time_col)
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol=feature_col,
            handleInvalid="skip",  # Skip rows with null/NaN values
        )
        stages.append(assembler)

    return Pipeline(stages=stages)


def build_ml_pipeline(
    df: DataFrame,
    *,
    config: TechnicalConfig | None = None,
    pt_mult: float,
    sl_mult: float,
    max_hold: int,
    time_col: str = "bucket",
    vol_window: int = 20,
    vol_method: str = "ewm",
) -> DataFrame:
    """Build complete ML pipeline: features + triple-barrier labels.

    Convenience function that chains:
      1. Volatility computation
      2. Technical indicators
      3. Triple-barrier labeling

    This is NOT a Spark ML Pipeline object, but a functional pipeline
    that returns the fully processed DataFrame ready for ML training.

    Parameters
    ----------
    df : DataFrame
        Spark DataFrame with OHLCV bars. Must contain:
        open, high, low, close, volume columns.
    config : TechnicalConfig, optional
        Technical indicator configuration.
    pt_mult : float
        Profit-taking multiplier (barrier = close × (1 + pt_mult × vol))
    sl_mult : float
        Stop-loss multiplier (barrier = close × (1 − sl_mult × vol))
    max_hold : int
        Maximum holding period (bars) — vertical barrier
    time_col : str, default="bucket"
        Name of datetime column for cyclical features.
    vol_window : int, default=20
        Window size for volatility estimation.
    vol_method : str, default="ewm"
        Volatility method: "ewm" or "rolling"

    Returns
    -------
    DataFrame
        DataFrame with:
        * Original OHLCV columns
        * Technical indicators (100+ features)
        * Triple-barrier labels (label, touch_bar, touch_type, event_start, event_end)

    Examples
    --------
    >>> from btcusdt_utils.features.pipeline_spark import build_ml_pipeline
    >>> from btcusdt_utils.features._config import TechnicalConfig
    >>>
    >>> # Complete ML pipeline
    >>> df_ml = build_ml_pipeline(
    ...     df_bars,
    ...     config=TechnicalConfig(),
    ...     pt_mult=3.0,
    ...     sl_mult=3.0,
    ...     max_hold=100,
    ... )
    >>>
    >>> # Now df_ml has features + labels, ready for training
    >>> df_ml.select("close", "sma_dist_10", "rsi_14", "label").show(5)

    Notes
    -----
    **Workflow**:
    1. Compute volatility (EWMA of log returns)
    2. Compute technical indicators (SMA, EMA, RSI, MACD, etc.)
    3. Apply triple-barrier labeling (distributed via Pandas UDF)

    **Performance**: O(n / p) where p = # partitions
    * All operations are embarrassingly parallel
    * No shuffle required for labeling (grouped map)
    * Scales to unlimited bars via horizontal partitioning

    **Memory**: Serverless-compatible
    * No .cache() or .persist() calls
    * Partition-local computation only
    * Spills to disk if partition exceeds memory

    See Also
    --------
    compute_features_from_bars : Features only (no labeling)
    btcusdt_utils.labeling_spark.apply_triple_barrier_spark : Labeling only
    """
    from btcusdt_utils.labeling_spark import (
        apply_triple_barrier_spark,
        compute_volatility,
    )

    # Step 1: Compute volatility for barrier construction
    df_vol = compute_volatility(
        df,
        close_col="close",
        window_size=vol_window,
        method=vol_method,
    )

    # Step 2: Compute technical indicators
    df_features = compute_features_from_bars(
        df_vol,
        config=config,
        time_col=time_col,
    )

    # Step 3: Add bar_index for joining with labels
    # Note: Must be added BEFORE labeling so both DataFrames have it
    window_spec = Window.orderBy(time_col)
    df_features = df_features.withColumn(
        "bar_index",
        (F.row_number().over(window_spec) - 1).cast("long")  # 0-indexed
    )

    # Step 4: Apply triple-barrier labeling
    df_labels = apply_triple_barrier_spark(
        df_features,
        pt_mult=pt_mult,
        sl_mult=sl_mult,
        max_hold=max_hold,
        close_col="close",
        high_col="high",
        low_col="low",
        vol_col="vol",
    )

    # Step 5: Join labels back to features
    # Note: labeling returns only labeled events (subset of bars)
    df_ml = df_features.join(
        df_labels,
        on="bar_index",  # Both DataFrames now have bar_index
        how="inner",     # Only keep labeled bars
    )

    return df_ml


def _get_feature_column_names(
    config: TechnicalConfig,
    time_col: str,
) -> list[str]:
    """Get list of all feature column names from config.

    Parameters
    ----------
    config : TechnicalConfig
        Technical indicator configuration.
    time_col : str
        Datetime column name.

    Returns
    -------
    list[str]
        Alphabetically sorted list of all feature column names.

    Notes
    -----
    This function reconstructs the feature column names that will be
    generated by TechnicalIndicatorsTransformer based on the config.
    """
    feature_cols = []

    # OHLCV base columns
    feature_cols.extend(["open", "high", "low", "close", "volume"])

    # A. SMA distances
    feature_cols.extend(f"sma_dist_{w}" for w in config.sma_windows)

    # B. EMA distances
    feature_cols.extend(f"ema_dist_{s}" for s in config.ema_spans)

    # C. RSI
    feature_cols.extend(f"rsi_{p}" for p in config.rsi_periods)

    # D. MACD (line, signal, histogram)
    for fast, slow, sig in config.macd_configs:
        tag = f"{fast}_{slow}_{sig}"
        feature_cols.extend([
            f"macd_line_{tag}",
            f"macd_signal_{tag}",
            f"macd_hist_{tag}",
        ])

    # E. Bollinger Bands (pct, width, std)
    for period in config.bb_periods:
        feature_cols.extend([
            f"bb_pct_{period}",
            f"bb_width_{period}",
            f"bb_std_{period}",
        ])

    # F. ATR (absolute and percentage)
    for period in config.atr_periods:
        feature_cols.extend([
            f"atr_{period}",
            f"atr_pct_{period}",
        ])

    # G. Returns
    feature_cols.extend(f"return_{k}" for k in config.return_lookbacks)

    # H. Volume ratios
    feature_cols.extend(f"vol_ratio_{w}" for w in config.vol_ratio_windows)

    # I. Candle features
    if config.add_candle_features:
        feature_cols.extend([
            "body_size",
            "upper_shadow",
            "lower_shadow",
            "is_green",
            "range_pct",
        ])

    # J. Time cyclicals
    if config.add_time_features:
        feature_cols.extend([
            "hour_sin",
            "hour_cos",
            "minute_sin",
            "minute_cos",
        ])

    # K. SMA crossovers
    for fast, slow in config.sma_cross_pairs:
        feature_cols.append(f"sma_cross_{fast}_{slow}")

    return sorted(feature_cols)


def build_bars_only_pipeline(
    bar_type: Literal["time", "volume", "dollar", "tick"],
    *,
    interval_minutes: int | None = None,
    threshold: float | None = None,
) -> Pipeline:
    """Build pipeline for bar construction only (no technical indicators).

    Parameters
    ----------
    bar_type : {"time", "volume", "dollar", "tick"}
        Type of bars to construct.
    interval_minutes : int, optional
        For time bars: aggregation interval in minutes.
    threshold : float, optional
        For threshold bars: threshold value.

    Returns
    -------
    Pipeline
        Spark ML Pipeline with single bar construction stage.

    Examples
    --------
    >>> pipeline = build_bars_only_pipeline(
    ...     bar_type="time",
    ...     interval_minutes=5,
    ... )
    >>> df_bars = pipeline.fit(df_ticks).transform(df_ticks)
    """
    if bar_type == "time":
        if interval_minutes is None:
            raise ValueError("interval_minutes required for time bars")
        bar_transformer = TimeBarTransformer(
            intervalMinutes=interval_minutes,
            timestampCol="timestamp",
            priceCol="price",
            qtyCol="qty",
        )
    elif bar_type in ("volume", "dollar", "tick"):
        if threshold is None:
            raise ValueError(f"threshold required for {bar_type} bars")
        bar_transformer = ThresholdBarTransformer(
            threshold=threshold,
            barType=bar_type,
            timestampCol="timestamp",
            priceCol="price",
            qtyCol="qty",
        )
    else:
        raise ValueError(f"Invalid bar_type: {bar_type}")

    return Pipeline(stages=[bar_transformer])


def compute_features_from_bars(
    df: DataFrame,
    *,
    config: TechnicalConfig | None = None,
    time_col: str = "bucket",
) -> DataFrame:
    """Convenience function to compute features from pre-constructed bars.

    Use this when you already have OHLCV bars and just need to add
    technical indicators.

    Parameters
    ----------
    df : DataFrame
        DataFrame with OHLCV bars (must have: open, high, low, close, volume).
    config : TechnicalConfig, optional
        Technical indicator configuration.
    time_col : str, default="bucket"
        Name of datetime column.

    Returns
    -------
    DataFrame
        Original bars plus technical indicators.

    Examples
    --------
    >>> from btcusdt_utils.features.pipeline_spark import compute_features_from_bars
    >>> df_features = compute_features_from_bars(
    ...     df_bars,
    ...     config=TechnicalConfig(),
    ... )
    """
    transformer = TechnicalIndicatorsTransformer(
        config=config,
        timeCol=time_col,
    )
    return transformer.transform(df)
