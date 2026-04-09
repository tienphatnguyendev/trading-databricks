from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from trading_databricks.ml.features.assembly import assemble_training_dataset


def _events_lf(n: int = 6) -> pl.LazyFrame:
    base = datetime(2025, 1, 1)
    return pl.DataFrame(
        {
            "event_id": [f"evt-{i}" for i in range(n)],
            "symbol": ["BTCUSDT"] * n,
            "event_ts": [base + timedelta(days=i) for i in range(n)],
            "bar_index": list(range(0, n * 10, 10)),
        }
    ).lazy()


def _labels_lf(n: int = 6) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "event_id": [f"evt-{i}" for i in range(n)],
            "symbol": ["BTCUSDT"] * n,
            "label": [1, -1, 0, 1, -1, 1],
            "avg_uniqueness": [1.0, 0.8, 0.9, 1.0, 0.75, 0.85],
        }
    ).lazy()


def _feature_lf(n: int = 6, col_name: str = "feat_a") -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "event_id": [f"evt-{i}" for i in range(n)],
            "symbol": ["BTCUSDT"] * n,
            col_name: [float(i) for i in range(n)],
        }
    ).lazy()


def test_assemble_returns_eager_dataframe() -> None:
    result = assemble_training_dataset(
        _events_lf(), _labels_lf(), [_feature_lf(col_name="feat_a")]
    )
    assert isinstance(result, pl.DataFrame)


def test_assemble_cpcv_group_in_range_1_to_6() -> None:
    result = assemble_training_dataset(
        _events_lf(n=6), _labels_lf(n=6), [_feature_lf(n=6, col_name="feat_a")]
    )
    groups = result["cpcv_group"].to_list()
    assert all(1 <= g <= 6 for g in groups), f"Invalid cpcv_group values: {groups}"


def test_assemble_dataset_version_is_iso_string() -> None:
    result = assemble_training_dataset(
        _events_lf(), _labels_lf(), [_feature_lf(col_name="feat_a")]
    )
    version = result["dataset_version"][0]
    assert isinstance(version, str)
    # Should be ISO8601 parseable
    datetime.fromisoformat(version)


def test_assemble_all_feature_columns_present() -> None:
    result = assemble_training_dataset(
        _events_lf(),
        _labels_lf(),
        [_feature_lf(col_name="feat_a"), _feature_lf(col_name="feat_b")],
    )
    assert "feat_a" in result.columns
    assert "feat_b" in result.columns
    assert "label" in result.columns
    assert "avg_uniqueness" in result.columns
    assert "cpcv_group" in result.columns
    assert "dataset_version" in result.columns


def test_assemble_sorted_by_event_ts() -> None:
    """Assembly sorts by event_ts before assigning cpcv_group."""
    import random
    base = datetime(2025, 1, 1)
    indices = list(range(6))
    random.shuffle(indices)
    events = pl.DataFrame(
        {
            "event_id": [f"evt-{i}" for i in indices],
            "symbol": ["BTCUSDT"] * 6,
            "event_ts": [base + timedelta(days=i) for i in indices],
            "bar_index": [i * 10 for i in indices],
        }
    ).lazy()
    labels = _labels_lf()
    result = assemble_training_dataset(events, labels, [_feature_lf(col_name="feat_a")])
    # cpcv_group should be 1..6 for earliest..latest events
    ts_sorted = result.sort("event_ts")["cpcv_group"].to_list()
    assert ts_sorted == sorted(ts_sorted)
