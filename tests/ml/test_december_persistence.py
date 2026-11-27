from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from trading_databricks.ml.december_persistence import (
    build_hourly_dataset,
    score_persistence,
)

_DECEMBER_START = datetime(2025, 12, 1, tzinfo=UTC)
_FIVE_MINUTES = timedelta(minutes=5)
_ONE_HOUR = timedelta(hours=1)
_LOG_RETURN = 0.0001


def _epoch_us(timestamp: datetime) -> int:
    return (timestamp - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1)


@pytest.fixture(scope="module")
def complete_bars() -> pl.DataFrame:
    indices = range(1, 8_929)
    end_times = [_epoch_us(_DECEMBER_START + i * _FIVE_MINUTES) for i in indices]
    return pl.DataFrame(
        {
            "bar_end_us": end_times,
            "last_trade_us": [end - 1_000_000 for end in end_times],
            "close": [100.0 * math.exp(i * _LOG_RETURN) for i in indices],
        }
    )


def test_build_hourly_dataset_uses_complete_common_origins(
    complete_bars: pl.DataFrame,
) -> None:
    dataset = build_hourly_dataset(complete_bars.reverse())

    assert dataset.height == 719
    assert dict(dataset.group_by("split").len().iter_rows()) == {
        "train": 383,
        "validation": 168,
        "test": 168,
    }
    split_starts = dataset.group_by("split").agg(pl.col("forecast_at").min())
    assert dict(split_starts.iter_rows()) == {
        "train": datetime(2025, 12, 2, 1, tzinfo=UTC),
        "validation": datetime(2025, 12, 18, tzinfo=UTC),
        "test": datetime(2025, 12, 25, tzinfo=UTC),
    }
    assert dataset["symbol"].unique().to_list() == ["BTCUSDT"]
    for column in ("forecast_at", "label_end", "max_feature_trade_at"):
        assert dataset.schema[column] == pl.Datetime("us", "UTC")

    first = dataset.row(0, named=True)
    assert first["forecast_at"] == datetime(2025, 12, 2, 1, tzinfo=UTC)
    assert first["label_end"] == datetime(2025, 12, 2, 2, tzinfo=UTC)
    assert first["max_feature_trade_at"] == datetime(2025, 12, 2, 0, 59, 59, tzinfo=UTC)
    for column in ("rv_1h", "rv_6h_mean", "rv_24h_mean", "actual_variance"):
        assert first[column] == pytest.approx(0.00000012)
    assert dataset["label_end"][-1] == datetime(2026, 1, 1, tzinfo=UTC)


def test_future_price_changes_the_target_and_preserves_the_features(
    complete_bars: pl.DataFrame,
) -> None:
    origin = datetime(2025, 12, 13, 12, tzinfo=UTC)
    last_future_bar = _epoch_us(origin + _ONE_HOUR)
    changed = complete_bars.with_columns(
        pl.when(pl.col("bar_end_us") == last_future_bar)
        .then(pl.col("close") * 1.01)
        .otherwise(pl.col("close"))
        .alias("close")
    )

    original_hour = build_hourly_dataset(complete_bars).filter(
        pl.col("forecast_at") == origin
    )
    modified_hour = build_hourly_dataset(changed).filter(pl.col("forecast_at") == origin)

    assert modified_hour["actual_variance"][0] > original_hour["actual_variance"][0]
    assert_frame_equal(
        modified_hour.drop("actual_variance"),
        original_hour.drop("actual_variance"),
    )


def test_incomplete_bar_blocks_windows_that_span_it(
    complete_bars: pl.DataFrame,
) -> None:
    gap_hour = datetime(2025, 12, 4, tzinfo=UTC)
    gap_at = _epoch_us(gap_hour + _FIVE_MINUTES)
    gapped = complete_bars.filter(pl.col("bar_end_us") != gap_at)

    origins = set(build_hourly_dataset(gapped)["forecast_at"])

    assert gap_hour - _ONE_HOUR in origins
    assert gap_hour not in origins
    assert gap_hour + _ONE_HOUR not in origins
    assert gap_hour + timedelta(hours=24) not in origins
    assert gap_hour + timedelta(hours=25) in origins


def test_score_persistence_floors_the_forecast_and_keeps_zero_actual() -> None:
    dataset = pl.DataFrame({"rv_1h": [0.0, 4e-6], "actual_variance": [0.0, 9e-6]})

    scored = score_persistence(dataset)

    assert scored["predicted_variance"].to_list() == [1e-12, 4e-6]
    assert scored["actual_variance"].to_list() == [0.0, 9e-6]
    assert scored["qlike"].to_list() == pytest.approx(
        [math.log(1e-12), math.log(4e-6) + 2.25]
    )
    assert scored["absolute_volatility_error_bps"].to_list() == pytest.approx([0.01, 10])
