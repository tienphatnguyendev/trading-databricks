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
    """
    return labels.get_column("avg_uniqueness").to_numpy().astype(np.float64)
