from __future__ import annotations

import numpy as np
import polars as pl


def compute_sample_weights(labels: pl.DataFrame) -> np.ndarray:
    """Extract avg_uniqueness from labeled events as a numpy float64 array.

    The avg_uniqueness column is computed by TRD-44's labeling module and
    stored in TRIPLE_BARRIER_LABELS_SCHEMA. This function exists for sklearn
    sample_weight parameter compatibility.

    Args:
        labels: DataFrame with at least an 'avg_uniqueness' column.

    Returns:
        1-D numpy float64 array of sample weights, one per event.

    Raises:
        polars.exceptions.ColumnNotFoundError: If 'avg_uniqueness' column is absent.
        ValueError: If weights contain NaN or non-positive values.
    """
    weights = labels.get_column("avg_uniqueness").to_numpy().astype(np.float64)

    n_nan = int(np.isnan(weights).sum())
    # NaN comparisons return False in numpy, so <= already excludes NaN
    n_nonpos = int((weights <= 0.0).sum())

    if n_nan > 0 or n_nonpos > 0:
        raise ValueError(
            f"Invalid sample weights: {n_nan} NaN, {n_nonpos} non-positive. "
            "Check upstream labeling for degenerate events."
        )

    return weights
