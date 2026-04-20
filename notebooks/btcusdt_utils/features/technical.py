"""Configurable technical indicator computation (Polars lazy).

All indicators are **stationary-by-construction** (distances, ratios,
bounded oscillators) so no FFD post-processing is required.

Window sizes are driven by :class:`TechnicalConfig` — the same
Polars lazy pipeline serves both the compact 39-feature set and the
full 100+ feature sweep from Phase 4B.

Reference: Phase 4B §4 of BTCUSDT EDA notebook.
"""

from __future__ import annotations

from math import pi as PI

import polars as pl

from btcusdt_utils.features._config import TechnicalConfig

_REQUIRED = ("open", "high", "low", "close", "volume")


def compute_technical_indicators(
    df: pl.DataFrame,
    *,
    config: TechnicalConfig | None = None,
    time_col: str = "bucket",
) -> pl.DataFrame:
    """Compute technical indicators on OHLCV bar data.

    Parameters
    ----------
    df       : Must contain ``open``, ``high``, ``low``, ``close``,
               ``volume``.  Optional: ``trade_count``, datetime
               column named *time_col*.
    config   : Window configuration (keyword-only).  Defaults to
               ``TechnicalConfig()``.
    time_col : Name of the datetime column for cyclical features.

    Returns
    -------
    pl.DataFrame with original columns plus all computed features.
    """
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if config is None:
        config = TechnicalConfig()

    cols = set(df.columns)
    _c = pl.col("close")
    _h = pl.col("high")
    _l = pl.col("low")
    _o = pl.col("open")
    _v = pl.col("volume")

    lf = df.lazy()

    # ── A. SMA distance (stationary: close/sma − 1) ─────────
    lf = lf.with_columns(
        (_c / _c.rolling_mean(window_size=w) - 1).alias(f"sma_dist_{w}")
        for w in config.sma_windows
    )

    # ── B. EMA distance (stationary: close/ema − 1) ─────────
    lf = lf.with_columns(
        (_c / _c.ewm_mean(span=s, adjust=False) - 1).alias(f"ema_dist_{s}")
        for s in config.ema_spans
    )

    # ── C. RSI (bounded 0–100) ────────────────────────────
    _delta = _c.diff()
    for p in config.rsi_periods:
        gain = _delta.clip(lower_bound=0).rolling_mean(window_size=p)
        loss = (-_delta.clip(upper_bound=0)).rolling_mean(window_size=p)
        lf = lf.with_columns(
            (100.0 - 100.0 / (1.0 + gain / loss))
            .fill_nan(100.0)
            .alias(f"rsi_{p}"),
        )

    # ── D. MACD (fast, slow, signal) ───────────────────────
    for fast, slow, sig in config.macd_configs:
        tag = f"{fast}_{slow}_{sig}"
        lf = lf.with_columns(
            (_c.ewm_mean(span=fast, adjust=False)
             - _c.ewm_mean(span=slow, adjust=False))
            .alias(f"macd_line_{tag}"),
        ).with_columns(
            pl.col(f"macd_line_{tag}")
            .ewm_mean(span=sig, adjust=False)
            .alias(f"macd_signal_{tag}"),
        ).with_columns(
            (pl.col(f"macd_line_{tag}") - pl.col(f"macd_signal_{tag}"))
            .alias(f"macd_hist_{tag}"),
        )

    # ── E. Bollinger Bands (pct, width, std) ───────────────
    for p in config.bb_periods:
        sma_t = f"_bb_sma_{p}"
        std_t = f"_bb_std_{p}"
        lf = (
            lf.with_columns(
                _c.rolling_mean(window_size=p).alias(sma_t),
                _c.rolling_std(window_size=p).alias(std_t),
            ).with_columns(
                ((_c - (pl.col(sma_t) - 2 * pl.col(std_t)))
                 / (4 * pl.col(std_t))).alias(f"bb_pct_{p}"),
                (4 * pl.col(std_t) / pl.col(sma_t)).alias(f"bb_width_{p}"),
                pl.col(std_t).alias(f"bb_std_{p}"),
            ).drop(sma_t)
        )

    # ── F. ATR (true range / close ratio) ──────────────────
    lf = lf.with_columns(_c.shift(1).alias("_prev_close")).with_columns(
        pl.max_horizontal(
            _h - _l,
            (_h - pl.col("_prev_close")).abs(),
            (_l - pl.col("_prev_close")).abs(),
        ).alias("_tr"),
    )
    for p in config.atr_periods:
        lf = lf.with_columns(
            pl.col("_tr").rolling_mean(window_size=p).alias(f"atr_{p}"),
        ).with_columns(
            (pl.col(f"atr_{p}") / _c).alias(f"atr_pct_{p}"),
        )
    lf = lf.drop("_prev_close", "_tr")

    # ── G. Returns (close/close_lag − 1) ───────────────────
    lf = lf.with_columns(
        (_c / _c.shift(k) - 1).alias(f"return_{k}")
        for k in config.return_lookbacks
    )

    # ── H. Volume ratios (vol / rolling mean vol) ──────────
    lf = lf.with_columns(
        (_v / _v.rolling_mean(window_size=w)).alias(f"vol_ratio_{w}")
        for w in config.vol_ratio_windows
    )

    # ── I. Candle features ───────────────────────────────
    if config.add_candle_features:
        lf = lf.with_columns(
            ((_c - _o).abs() / _c).alias("body_size"),
            ((_h - pl.max_horizontal(_o, _c)) / _c).alias("upper_shadow"),
            ((pl.min_horizontal(_o, _c) - _l) / _c).alias("lower_shadow"),
            (_c >= _o).cast(pl.Int8).alias("is_green"),
            ((_h - _l) / _c).alias("range_pct"),
        )

    # ── J. Time cyclicals ────────────────────────────────
    if config.add_time_features and time_col in cols:
        _t = pl.col(time_col)
        lf = lf.with_columns(
            (2 * PI * _t.dt.hour().cast(pl.Float64) / 24).sin().alias("hour_sin"),
            (2 * PI * _t.dt.hour().cast(pl.Float64) / 24).cos().alias("hour_cos"),
            (2 * PI * _t.dt.minute().cast(pl.Float64) / 60).sin().alias("minute_sin"),
            (2 * PI * _t.dt.minute().cast(pl.Float64) / 60).cos().alias("minute_cos"),
        )

    # ── K. SMA crossovers (binary) ────────────────────────
    for fast, slow in config.sma_cross_pairs:
        lf = lf.with_columns(
            (_c.rolling_mean(window_size=fast)
             > _c.rolling_mean(window_size=slow))
            .cast(pl.Int8)
            .alias(f"sma_cross_{fast}_{slow}"),
        )

    return lf.collect()
