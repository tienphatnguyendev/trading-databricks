"""Hourly BTCUSDT realized-variance dataset and persistence baseline for December 2025."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

_DECEMBER_START = datetime(2025, 12, 1, tzinfo=UTC)
_DECEMBER_END = datetime(2026, 1, 1, tzinfo=UTC)
_VALIDATION_START = datetime(2025, 12, 18, tzinfo=UTC)
_TEST_START = datetime(2025, 12, 25, tzinfo=UTC)
_UTC_MICROSECONDS = pl.Datetime("us", "UTC")
_BAR_US = 5 * 60 * 1_000_000
_BARS_PER_HOUR = 12
_HOUR_US = _BAR_US * _BARS_PER_HOUR
_FORECAST_FLOOR = 1e-12


def _rolling_hourly_variance(squared_return: pl.Expr, hours: int) -> pl.Expr:
    window_size = hours * _BARS_PER_HOUR
    return squared_return.rolling_sum(window_size, min_samples=window_size) / hours


def build_hourly_dataset(bars: pl.DataFrame) -> pl.DataFrame:
    """Build fixed-clock origins with 24 hours of history and one hour of outcomes.

    Bars are for BTCUSDT and carry a close, UTC epoch microsecond endpoints, and
    a last trade timestamp in the half-open five-minute interval ending at
    each endpoint. Missing or invalid bars exclude any affected origin.
    """
    bar_end = pl.col("bar_end_us")
    last_trade = pl.col("last_trade_us")
    close = pl.col("close")
    bars = (
        bars.select(
            bar_end.cast(pl.Int64), last_trade.cast(pl.Int64), close.cast(pl.Float64)
        )
        .filter(
            bar_end.cast(_UTC_MICROSECONDS).is_between(
                _DECEMBER_START, _DECEMBER_END, closed="right"
            ),
            bar_end % _BAR_US == 0,
            last_trade >= bar_end - _BAR_US,
            last_trade < bar_end,
            close.is_finite(),
            close > 0.0,
        )
        .sort("bar_end_us")
    )
    if bars["bar_end_us"].n_unique() != bars.height:
        raise ValueError("Five-minute bars must have unique endpoints.")

    squared_return = pl.col("_squared_return")
    forecast_at = pl.col("forecast_at")
    return (
        bars.with_columns(
            # A missing bar invalidates the return to the next available close.
            _squared_return=pl.when(bar_end.diff() == _BAR_US).then(
                close.log().diff().pow(2)
            )
        )
        .with_columns(
            rv_1h=_rolling_hourly_variance(squared_return, 1),
            rv_6h_mean=_rolling_hourly_variance(squared_return, 6),
            rv_24h_mean=_rolling_hourly_variance(squared_return, 24),
        )
        # Shift at five-minute resolution so the target starts after the origin.
        .with_columns(
            forecast_at=bar_end.cast(_UTC_MICROSECONDS),
            actual_variance=pl.col("rv_1h").shift(-_BARS_PER_HOUR),
        )
        .filter(bar_end % _HOUR_US == 0)
        .drop_nulls(["rv_1h", "rv_6h_mean", "rv_24h_mean", "actual_variance"])
        .select(
            pl.lit("BTCUSDT").alias("symbol"),
            forecast_at,
            (forecast_at + timedelta(hours=1)).alias("label_end"),
            last_trade.cast(_UTC_MICROSECONDS).alias("max_feature_trade_at"),
            "rv_1h",
            "rv_6h_mean",
            "rv_24h_mean",
            "actual_variance",
            pl.when(forecast_at < _VALIDATION_START)
            .then(pl.lit("train"))
            .when(forecast_at < _TEST_START)
            .then(pl.lit("validation"))
            .otherwise(pl.lit("test"))
            .alias("split"),
        )
    )


def score_persistence(dataset: pl.DataFrame) -> pl.DataFrame:
    """Score persistence with QLIKE and volatility error in basis points."""
    forecast = pl.col("predicted_variance")
    actual = pl.col("actual_variance")
    return dataset.with_columns(
        predicted_variance=pl.col("rv_1h").clip(lower_bound=_FORECAST_FLOOR)
    ).with_columns(
        qlike=forecast.log() + actual / forecast,
        absolute_volatility_error_bps=(forecast.sqrt() - actual.sqrt()).abs() * 10_000,
    )
