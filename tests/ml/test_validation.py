from datetime import datetime

import polars as pl
import pytest

from trading_databricks.ml.validation import (
    validate_bars,
    validate_events,
    validate_labels,
    validate_training_dataset,
)


def test_validate_bars() -> None:
    valid = pl.DataFrame(
        {
            "bar_index": [1, 2, 3],
            "close": [100.0, 101.0, 102.0],
            "bar_end_time": [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
            ],
            "symbol": ["BTCUSD", "BTCUSD", "BTCUSD"],
        }
    )
    validate_bars(valid)
    validate_bars(valid.clear())

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_bars(valid.drop("close"))

    with pytest.raises(ValueError, match="close prices must be > 0"):
        validate_bars(valid.with_columns(pl.Series("close", [100.0, 0.0, 102.0])))

    with pytest.raises(ValueError, match="bar_end_time contains nulls"):
        validate_bars(
            valid.with_columns(
                pl.Series(
                    "bar_end_time", [datetime(2023, 1, 1), None, datetime(2023, 1, 3)]
                )
            )
        )

    with pytest.raises(
        ValueError, match="bar_index must be strictly monotonically increasing"
    ):
        validate_bars(valid.with_columns(pl.Series("bar_index", [1, 3, 2])))

    with pytest.raises(
        ValueError, match="bar_index must be strictly monotonically increasing"
    ):
        validate_bars(valid.with_columns(pl.Series("bar_index", [1, 2, 2])))


def test_validate_events() -> None:
    valid = pl.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "symbol": ["BTCUSD", "BTCUSD"],
            "event_ts": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            "bar_index": [10, 20],
            "log_return": [0.01, -0.01],
            "cusum_side": ["up", "down"],
            "cusum_value": [0.05, -0.05],
            "threshold_used": [0.02, 0.02],
            "ewma_vol": [0.01, 0.01],
        }
    )
    validate_events(valid)
    validate_events(valid.clear())

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_events(valid.drop("event_id"))

    with pytest.raises(ValueError, match="ewma_vol must be > 0"):
        validate_events(valid.with_columns(pl.Series("ewma_vol", [0.01, 0.0])))

    with pytest.raises(ValueError, match="cusum_side must be 'up' or 'down'"):
        validate_events(valid.with_columns(pl.Series("cusum_side", ["up", "left"])))

    with pytest.raises(ValueError, match="threshold_used must be > 0 and non-null"):
        validate_events(valid.with_columns(pl.Series("threshold_used", [0.02, None])))

    with pytest.raises(ValueError, match="event_ts contains nulls"):
        validate_events(
            valid.with_columns(pl.Series("event_ts", [None, datetime(2023, 1, 2)]))
        )


def test_validate_labels() -> None:
    valid = pl.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "symbol": ["BTCUSD", "BTCUSD"],
            "event_ts": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            "label": [1, -1],
            "barrier_hit": ["upper", "lower"],
            "return_pct": [0.05, -0.05],
            "upper_level": [105.0, 105.0],
            "lower_level": [95.0, 95.0],
            "vertical_bar": [15, 25],
            "matured_at": [datetime(2023, 1, 1, 12), datetime(2023, 1, 2, 12)],
            "avg_uniqueness": [0.8, 1.0],
        }
    )
    validate_labels(valid)
    validate_labels(valid.clear())

    with pytest.raises(ValueError, match="label must be -1, 0, or 1"):
        validate_labels(valid.with_columns(pl.Series("label", [1, 2])))

    with pytest.raises(
        ValueError, match="barrier_hit must be 'upper', 'lower', or 'vertical'"
    ):
        validate_labels(valid.with_columns(pl.Series("barrier_hit", ["upper", "none"])))

    with pytest.raises(ValueError, match="avg_uniqueness must be > 0 and <= 1"):
        validate_labels(valid.with_columns(pl.Series("avg_uniqueness", [0.0, 1.0])))


def test_validate_training_dataset() -> None:
    valid = pl.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "label": [1, -1],
            "cpcv_group": [1, 2],
            "feature_1": [0.5, 0.6],
        }
    )
    validate_training_dataset(valid, n_groups=2)

    with pytest.raises(ValueError, match="Training dataset DataFrame is empty"):
        validate_training_dataset(valid.clear(), n_groups=2)

    with pytest.raises(ValueError, match="label column contains nulls"):
        validate_training_dataset(
            valid.with_columns(pl.Series("label", [1, None])), n_groups=2
        )

    with pytest.raises(ValueError, match="cpcv_group must be between 1 and n_groups"):
        validate_training_dataset(
            valid.with_columns(pl.Series("cpcv_group", [1, 3])), n_groups=2
        )

    with pytest.raises(ValueError, match="Column 'feature_1' is all nulls"):
        validate_training_dataset(
            valid.with_columns(pl.Series("feature_1", [None, None], dtype=pl.Float64)),
            n_groups=2,
        )
