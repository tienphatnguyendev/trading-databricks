"""Frozen dataclass configs for feature engineering modules.

Each config controls the hyper-parameters (window sizes, thresholds)
for one feature domain.  All use ``@dataclass(frozen=True)`` so they
are hashable and safe to pass across threads.

Defaults match the research-notebook values (Phase 4A/4B of
BTCUSDT EDA Parameter Optimization).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TechnicalConfig:
    """Window parameters for traditional TA indicators.

    All indicators are stationary-by-construction (distances, ratios)
    so no FFD post-processing is needed.
    """

    sma_windows: tuple[int, ...] = (3, 5, 10, 20, 50, 100, 200)
    ema_spans: tuple[int, ...] = (3, 5, 10, 20, 50, 100, 200)
    rsi_periods: tuple[int, ...] = (5, 7, 14, 21, 28, 50)
    macd_configs: tuple[tuple[int, int, int], ...] = (
        (8, 21, 5),
        (12, 26, 9),
        (5, 13, 5),
        (19, 39, 9),
    )
    bb_periods: tuple[int, ...] = (10, 20, 30, 50)
    atr_periods: tuple[int, ...] = (5, 7, 14, 21, 28)
    return_lookbacks: tuple[int, ...] = (1, 3, 5, 10, 20, 30, 60)
    vol_ratio_windows: tuple[int, ...] = (5, 10, 20, 50)
    sma_cross_pairs: tuple[tuple[int, int], ...] = (
        (3, 10),
        (5, 20),
        (5, 50),
        (10, 50),
        (20, 100),
        (50, 200),
    )
    add_time_features: bool = True
    add_candle_features: bool = True


@dataclass(frozen=True)
class MicrostructureConfig:
    """Window parameters for microstructure features.

    Controls rolling windows for VPIN, Kyle's Lambda, Amihud
    illiquidity, Roll spread, and Corwin-Schultz estimators.
    """

    rolling_windows: tuple[int, ...] = (10, 20, 50, 100)
    roll_spread_windows: tuple[int, ...] = (20, 50, 100)
    cs_spread_windows: tuple[int, ...] = (10, 20, 50)


@dataclass(frozen=True)
class EntropyConfig:
    """Window parameters for entropy features.

    Controls rolling windows for Shannon entropy and Lempel-Ziv
    complexity computed on binary return sequences.
    """

    shannon_windows: tuple[int, ...] = (10, 20, 50, 100)
    lz_windows: tuple[int, ...] = (20, 50, 100)


@dataclass(frozen=True)
class StructuralConfig:
    """Parameters for structural break features.

    Controls CUSUM reset intervals, rolling ADF test windows,
    martingale ratio windows, and volatility estimation.
    """

    cusum_resets: tuple[int | None, ...] = (None, 50, 100)
    adf_windows: tuple[int, ...] = (50, 100, 200)
    martingale_windows: tuple[int, ...] = (20, 50)
    vol_window: int = 20
