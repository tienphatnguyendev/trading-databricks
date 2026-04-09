from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from trading_databricks.ml.training.sample_weights import compute_sample_weights


def test_returns_numpy_array_of_avg_uniqueness() -> None:
    """compute_sample_weights extracts avg_uniqueness as numpy float64 array."""
    labels = pl.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "avg_uniqueness": [1.0, 0.8, 0.6],
            "label": [1, -1, 0],
        }
    )
    weights = compute_sample_weights(labels)

    assert isinstance(weights, np.ndarray)
    assert weights.dtype == np.float64
    np.testing.assert_array_almost_equal(weights, [1.0, 0.8, 0.6])


def test_weights_length_matches_labels() -> None:
    """Output array length equals number of rows in labels."""
    labels = pl.DataFrame({"event_id": ["e1"], "avg_uniqueness": [0.75]})
    weights = compute_sample_weights(labels)
    assert len(weights) == 1


def test_raises_when_avg_uniqueness_column_missing() -> None:
    """Labels DataFrame without avg_uniqueness → ColumnNotFoundError."""
    labels = pl.DataFrame({"event_id": ["e1"], "label": [1]})
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        compute_sample_weights(labels)


def test_nan_weights_raise_value_error() -> None:
    labels = pl.DataFrame({"avg_uniqueness": [0.8, float("nan"), 0.6]})
    with pytest.raises(ValueError, match="NaN"):
        compute_sample_weights(labels)


def test_zero_weights_raise_value_error() -> None:
    labels = pl.DataFrame({"avg_uniqueness": [0.8, 0.0, 0.6]})
    with pytest.raises(ValueError, match="non-positive"):
        compute_sample_weights(labels)
