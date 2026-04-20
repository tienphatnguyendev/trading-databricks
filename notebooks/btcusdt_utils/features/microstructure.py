"""Microstructure feature engineering from raw tick data.

Rebuilds bar-level buy/sell features from tick-level
``is_buyer_maker`` field (Binance semantics: True = sell-initiated).
Polars-first for aggregation; NumPy for rolling estimators.

Reference: AFML Ch. 18–19; Phase 4A of BTCUSDT EDA notebook.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from btcusdt_utils.features._config import MicrostructureConfig


# ── NumPy rolling helpers (no Polars equivalent) ─────────────

def _rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    """Right-aligned rolling mean; first w−1 entries are NaN."""
    kernel = np.ones(w) / w
    conv = np.convolve(arr, kernel, mode="full")[:len(arr)]
    out = np.full(len(arr), np.nan)
    out[w - 1:] = conv[w - 1:]
    return out


def _rolling_cov_var(
    x: np.ndarray, y: np.ndarray, w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling Cov(x,y) and Var(y) over window *w*. NaN-safe."""
    n = len(x)
    cov_out = np.full(n, np.nan)
    var_out = np.full(n, np.nan)
    min_obs = max(5, w // 3)
    for i in range(w, n):
        xw = x[i - w + 1 : i + 1]
        yw = y[i - w + 1 : i + 1]
        mask = np.isfinite(xw) & np.isfinite(yw)
        if mask.sum() > min_obs:
            cm = np.cov(xw[mask], yw[mask])
            cov_out[i], var_out[i] = cm[0, 1], cm[1, 1]
    return cov_out, var_out


def _corwin_schultz_raw(
    high: np.ndarray, low: np.ndarray,
) -> np.ndarray:
    """Corwin-Schultz bid-ask spread estimator (bar-by-bar).

    When alpha < 0 (implying a negative spread, which is
    economically impossible), we clamp alpha to 0 and report
    a spread of 0.  This is standard in the literature and
    avoids discarding ~99 % of bars in tight-spread markets.
    """
    n = len(high)
    out = np.full(n, np.nan)
    sqrt_2 = np.sqrt(2.0)
    denom = 3.0 - 2.0 * sqrt_2
    for i in range(1, n):
        beta = 0.5 * (
            np.log(high[i - 1] / low[i - 1]) ** 2
            + np.log(high[i] / low[i]) ** 2
        )
        gamma = np.log(
            max(high[i - 1], high[i]) / min(low[i - 1], low[i]),
        ) ** 2
        alpha = (
            (np.sqrt(2 * beta) - np.sqrt(beta)) / denom
            - np.sqrt(gamma / denom)
        )
        # Clamp: negative alpha ⟹ spread < 0, impossible
        alpha = max(0.0, alpha)
        s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
        if np.isfinite(s):
            out[i] = s
    return out


# ── Polars aggregation helpers ─────────────────────────────

def _assign_ticks_to_bars(
    df_ticks: pl.DataFrame, df_bars: pl.DataFrame,
) -> pl.DataFrame:
    """Assign each tick to its volume bar via ``join_asof``."""
    bounds = (
        df_bars.select("bucket")
        .with_row_index("bar_idx")
        .sort("bucket")
    )
    return (
        df_ticks.sort("transact_timestamp")
        .join_asof(
            bounds,
            left_on="transact_timestamp",
            right_on="bucket",
            strategy="backward",
        )
    )


def _aggregate_buy_sell(df_barred: pl.DataFrame) -> pl.DataFrame:
    """Per-bar buy/sell aggregation from tick-level is_buyer_maker."""
    return (
        df_barred
        .filter(pl.col("bar_idx").is_not_null())
        .group_by("bar_idx")
        .agg(
            pl.col("qty_d").filter(~pl.col("is_buyer_maker")).sum().alias("buy_volume"),
            pl.col("qty_d").filter(pl.col("is_buyer_maker")).sum().alias("sell_volume"),
            pl.col("dollar_d").filter(~pl.col("is_buyer_maker")).sum().alias("buy_dollars"),
            pl.col("dollar_d").filter(pl.col("is_buyer_maker")).sum().alias("sell_dollars"),
            (~pl.col("is_buyer_maker")).sum().alias("buy_count"),
            pl.col("is_buyer_maker").sum().alias("sell_count"),
        )
        .sort("bar_idx")
    )


def _derive_per_bar_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """Compute OI, VPIN, buy_ratio etc. from buy/sell aggregates."""
    _bv = pl.col("buy_volume")
    _sv = pl.col("sell_volume")
    _tot = _bv + _sv
    _bd = pl.col("buy_dollars")
    _sd = pl.col("sell_dollars")
    _bc = pl.col("buy_count").cast(pl.Float64)
    _sc = pl.col("sell_count").cast(pl.Float64)
    return df.with_columns(
        ((_bv - _sv) / _tot).alias("order_imbalance"),
        ((_bv - _sv).abs() / _tot).alias("vpin_raw"),
        (_bv / _tot).alias("buy_ratio"),
        ((_bd - _sd) / (_bd + _sd)).alias("dollar_imbalance"),
        ((_bc - _sc) / (_bc + _sc)).alias("trade_imbalance"),
        (_bv - _sv).alias("signed_volume"),
    )


# ── Rolling micro feature computation ───────────────────────

def _compute_rolling_features(
    df_work: pl.DataFrame, config: MicrostructureConfig,
) -> dict[str, np.ndarray]:
    """Build all rolling micro features from the joined work frame."""
    n = df_work.shape[0]
    dp = df_work["delta_price"].to_numpy().astype(np.float64)
    log_ret = df_work["log_ret"].to_numpy().astype(np.float64)
    volume = df_work["volume"].to_numpy().astype(np.float64)
    high = df_work["high"].to_numpy().astype(np.float64)
    low = df_work["low"].to_numpy().astype(np.float64)
    signed_v = df_work["signed_volume"].fill_null(0.0).to_numpy().astype(np.float64)
    oi_arr = df_work["order_imbalance"].fill_null(0.0).to_numpy().astype(np.float64)
    vpin_arr = df_work["vpin_raw"].fill_null(0.0).to_numpy().astype(np.float64)
    dollar_oi = df_work["dollar_imbalance"].fill_null(0.0).to_numpy().astype(np.float64)
    trade_imb = df_work["trade_imbalance"].fill_null(0.0).to_numpy().astype(np.float64)

    arrays: dict[str, np.ndarray] = {}
    wins = config.rolling_windows

    # VPIN + OI rolling
    for w in wins:
        arrays[f"vpin_{w}"] = _rolling_mean(vpin_arr, w)
        arrays[f"oi_{w}"] = _rolling_mean(oi_arr, w)

    # Kyle's Lambda: Cov(Δp, signed_vol) / Var(signed_vol)
    for w in wins:
        cov_dp_sv, var_sv = _rolling_cov_var(dp, signed_v, w)
        arrays[f"kyle_lambda_{w}"] = np.where(
            (var_sv > 1e-20) & np.isfinite(cov_dp_sv),
            cov_dp_sv / var_sv, np.nan,
        )

    # Amihud illiquidity: |return| / volume
    amihud_raw = np.abs(log_ret) / np.where(volume > 0, volume, np.nan)
    arrays["amihud_raw"] = amihud_raw
    safe_amihud = np.where(np.isfinite(amihud_raw), amihud_raw, 0.0)
    for w in wins:
        arrays[f"amihud_{w}"] = _rolling_mean(safe_amihud, w)

    # Roll Model spread: 2·√(-Cov(Δp_t, Δp_{t-1}))
    # NOTE: np.maximum(0, ...) guards against np.where's eager
    # evaluation of np.sqrt on positive cov_pad elements.
    dp_lag = np.roll(dp, 1)
    dp_lag[0] = np.nan
    for w in config.roll_spread_windows:
        cov_dp, _ = _rolling_cov_var(dp[1:], dp_lag[1:], w)
        cov_pad = np.full(n, np.nan)
        cov_pad[1:] = cov_dp
        arrays[f"roll_spread_{w}"] = np.where(
            (cov_pad < 0) & np.isfinite(cov_pad),
            2.0 * np.sqrt(np.maximum(0.0, -cov_pad)),
            np.where(np.isfinite(cov_pad), 0.0, np.nan),
        )

    # Corwin-Schultz spread
    cs_raw = _corwin_schultz_raw(high, low)
    arrays["cs_spread_raw"] = cs_raw
    safe_cs = np.where(np.isfinite(cs_raw), cs_raw, 0.0)
    n_valid = int(np.sum(np.isfinite(cs_raw)))
    for w in config.cs_spread_windows:
        arrays[f"cs_spread_{w}"] = (
            _rolling_mean(safe_cs, w) if n_valid >= 10
            else np.full(n, np.nan)
        )

    # Dollar-weighted OI + Trade count imbalance
    for w in (20, 50):
        arrays[f"dollar_oi_{w}"] = _rolling_mean(dollar_oi, w)
        arrays[f"trade_imb_{w}"] = _rolling_mean(trade_imb, w)

    return arrays


# ── Public API ────────────────────────────────────────────────

def compute_microstructure_features(
    df_ticks: pl.DataFrame,
    df_bars: pl.DataFrame,
    *,
    config: MicrostructureConfig | None = None,
) -> pl.DataFrame:
    """Enrich bar DataFrame with microstructure features.

    Parameters
    ----------
    df_ticks : Tick data with ``transact_timestamp``, ``qty_d``,
               ``dollar_d``, ``is_buyer_maker``, ``price_d``.
    df_bars  : OHLCV bars with ``bucket``, ``open``, ``high``,
               ``low``, ``close``, ``volume``.
    config   : Microstructure configuration (keyword-only).

    Returns
    -------
    pl.DataFrame : bars enriched with ``log_ret``, ``delta_price``,
        per-bar buy/sell metrics, and all rolling microstructure
        features.
    """
    if config is None:
        config = MicrostructureConfig()

    # 1-2. Tick → bar assignment + aggregation (Polars)
    barred = _assign_ticks_to_bars(df_ticks, df_bars)
    agg = _aggregate_buy_sell(barred)
    micro = _derive_per_bar_metrics(agg)

    # 3. Join OHLCV + micro on bar_idx
    indexed = df_bars.with_row_index("bar_idx")
    work = (
        indexed.join(micro, on="bar_idx", how="left")
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1)).log().alias("log_ret"),
            (pl.col("close") - pl.col("close").shift(1)).alias("delta_price"),
        )
    )

    # 4. Rolling features (NumPy)
    rolling = _compute_rolling_features(work, config)
    cols = [pl.Series(name, arr) for name, arr in rolling.items()]
    return work.with_columns(cols)
