"""Feature engineering subpackage.

Submodules
----------
_config         : Frozen dataclass configs per domain
technical       : Polars lazy TA pipeline (configurable windows)
microstructure  : Tick-level buy/sell → bar-level rolling features
entropy         : Shannon entropy, Lempel-Ziv complexity
structural      : CUSUM, rolling ADF, martingale ratio

Legacy re-exports (Phase 3 backward compat)
-------------------------------------------
FEATURE_COLS, FEATURE_GROUPS, FEATURE_CATEGORIES, SKIP_STATIONARITY,
validate_features — from the original flat features.py (v1 API).
"""

from __future__ import annotations

import polars as pl

from btcusdt_utils.features._config import (
    TechnicalConfig,
    MicrostructureConfig,
    EntropyConfig,
    StructuralConfig,
)
from btcusdt_utils.features.technical import compute_technical_indicators
from btcusdt_utils.features.microstructure import (
    compute_microstructure_features,
)
from btcusdt_utils.features.entropy import compute_entropy_features
from btcusdt_utils.features.structural import compute_structural_features


# ── Legacy constants (Phase 3 backward compat) ─────────────────

FEATURE_COLS: list[str] = [
    # A. Moving Averages (8)
    "sma_5", "sma_10", "sma_20", "sma_50",
    "ema_5", "ema_10", "ema_20", "sma_cross_5_20",
    # B. RSI (1)
    "rsi_14",
    # C. MACD (3)
    "macd_line", "signal_line", "macd_hist",
    # D. Bollinger Bands (5)
    "bb_std", "bb_upper", "bb_lower", "bb_width", "bb_pct",
    # E. ATR (3)
    "tr", "atr_14", "atr_pct",
    # F. Returns (5)
    "return_1", "return_5", "return_10", "return_30", "log_return_1",
    # G. Volume (5)
    "vol_sma_10", "vol_sma_20", "vol_ratio", "buy_ratio", "trade_intensity",
    # H. VWAP (1)
    "vwap_distance",
    # I. Candle (4)
    "body_size", "upper_shadow", "lower_shadow", "is_green",
    # J. Time (4)
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "A. Moving Averages": [
        "sma_5", "sma_10", "sma_20", "sma_50",
        "ema_5", "ema_10", "ema_20", "sma_cross_5_20",
    ],
    "B. RSI": ["rsi_14"],
    "C. MACD": ["macd_line", "signal_line", "macd_hist"],
    "D. Bollinger Bands": [
        "bb_std", "bb_upper", "bb_lower", "bb_width", "bb_pct",
    ],
    "E. ATR": ["tr", "atr_14", "atr_pct"],
    "F. Returns": [
        "return_1", "return_5", "return_10", "return_30", "log_return_1",
    ],
    "G. Volume": [
        "vol_sma_10", "vol_sma_20", "vol_ratio",
        "buy_ratio", "trade_intensity",
    ],
    "H. VWAP": ["vwap_distance"],
    "I. Candle": ["body_size", "upper_shadow", "lower_shadow", "is_green"],
    "J. Time": ["hour_sin", "hour_cos", "minute_sin", "minute_cos"],
}

FEATURE_CATEGORIES: dict[str, tuple[str, str]] = {
    "sma_5": ("A. Moving Avg", "numeric float (price-scale)"),
    "sma_10": ("A. Moving Avg", "numeric float (price-scale)"),
    "sma_20": ("A. Moving Avg", "numeric float (price-scale)"),
    "sma_50": ("A. Moving Avg", "numeric float (price-scale)"),
    "ema_5": ("A. Moving Avg", "numeric float (price-scale)"),
    "ema_10": ("A. Moving Avg", "numeric float (price-scale)"),
    "ema_20": ("A. Moving Avg", "numeric float (price-scale)"),
    "sma_cross_5_20": ("A. Moving Avg", "binary integer {0, 1}"),
    "rsi_14": ("B. RSI", "numeric float (bounded 0–100)"),
    "macd_line": ("C. MACD", "numeric float"),
    "signal_line": ("C. MACD", "numeric float"),
    "macd_hist": ("C. MACD", "numeric float"),
    "bb_std": ("D. Bollinger", "numeric float (price-scale)"),
    "bb_upper": ("D. Bollinger", "numeric float (price-scale)"),
    "bb_lower": ("D. Bollinger", "numeric float (price-scale)"),
    "bb_width": ("D. Bollinger", "numeric float (ratio)"),
    "bb_pct": ("D. Bollinger", "numeric float (~bounded 0–1)"),
    "tr": ("E. ATR", "numeric float (price-scale)"),
    "atr_14": ("E. ATR", "numeric float (price-scale)"),
    "atr_pct": ("E. ATR", "numeric float (ratio, ≥0)"),
    "return_1": ("F. Returns", "numeric float (ratio)"),
    "return_5": ("F. Returns", "numeric float (ratio)"),
    "return_10": ("F. Returns", "numeric float (ratio)"),
    "return_30": ("F. Returns", "numeric float (ratio)"),
    "log_return_1": ("F. Returns", "numeric float"),
    "vol_sma_10": ("G. Volume", "numeric float"),
    "vol_sma_20": ("G. Volume", "numeric float"),
    "vol_ratio": ("G. Volume", "numeric float (ratio, ≥0)"),
    "buy_ratio": ("G. Volume", "numeric float (bounded 0–1)"),
    "trade_intensity": ("G. Volume", "numeric float (ratio, ≥0)"),
    "vwap_distance": ("H. VWAP", "numeric float (ratio)"),
    "body_size": ("I. Candle", "numeric float (ratio)"),
    "upper_shadow": ("I. Candle", "numeric float (ratio, ≥0)"),
    "lower_shadow": ("I. Candle", "numeric float (ratio, ≥0)"),
    "is_green": ("I. Candle", "binary integer {0, 1}"),
    "hour_sin": ("J. Time", "cyclical float (bounded −1, 1)"),
    "hour_cos": ("J. Time", "cyclical float (bounded −1, 1)"),
    "minute_sin": ("J. Time", "cyclical float (bounded −1, 1)"),
    "minute_cos": ("J. Time", "cyclical float (bounded −1, 1)"),
}

SKIP_STATIONARITY: frozenset[str] = frozenset({
    "is_green", "sma_cross_5_20",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
})


def validate_features(
    df: pl.DataFrame,
    feature_cols: list[str] | None = None,
    *,
    max_warmup: int = 50,
) -> dict:
    """Audit feature columns for nulls, NaNs, and infinities."""
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    nulls: dict[str, int] = {}
    nans: dict[str, int] = {}
    infs: dict[str, int] = {}
    issues: list[str] = []

    for col in feature_cols:
        nc = df[col].null_count()
        if nc > 0:
            nulls[col] = nc
            if nc > max_warmup + 5:
                issues.append(f"{col}: {nc} nulls (exceeds warm-up of {max_warmup})")
        if df[col].dtype.is_float():
            nan_count = df[col].is_nan().sum()
            inf_count = df[col].is_infinite().sum()
            if nan_count is not None and nan_count > 0:
                nans[col] = int(nan_count)
                issues.append(f"{col}: {nan_count} NaNs")
            if inf_count is not None and inf_count > 0:
                infs[col] = int(inf_count)
                issues.append(f"{col}: {inf_count} infinities")

    first_clean_idx: int | None = None
    if nulls:
        df_idx = df.with_row_index("_idx")
        clean_rows = df_idx.drop_nulls(subset=feature_cols)["_idx"]
        if clean_rows.len() > 0:
            first_clean_idx = int(clean_rows.min())

    return {
        "nulls": nulls,
        "nans": nans,
        "infs": infs,
        "issues": issues,
        "first_clean_idx": first_clean_idx,
        "n_features": len(feature_cols),
    }


__all__ = [
    # Configs
    "TechnicalConfig",
    "MicrostructureConfig",
    "EntropyConfig",
    "StructuralConfig",
    # Computation
    "compute_technical_indicators",
    "compute_microstructure_features",
    "compute_entropy_features",
    "compute_structural_features",
    # Legacy (Phase 3 compat)
    "FEATURE_COLS",
    "FEATURE_GROUPS",
    "FEATURE_CATEGORIES",
    "SKIP_STATIONARITY",
    "validate_features",
]
