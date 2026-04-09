from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

from trading_databricks.ml.config import BetSizingConfig

_BET_SIZING_COLUMNS = [
    "signal_id", "event_id", "symbol", "event_ts",
    "primary_side", "primary_confidence",
    "meta_label", "meta_confidence",
    "raw_bet_size", "discretized_weight",
    "primary_model_version", "meta_model_version",
]


def _empty_result() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "signal_id": pl.Utf8,
            "event_id": pl.Utf8,
            "symbol": pl.Utf8,
            "event_ts": pl.Datetime,
            "primary_side": pl.Int32,
            "primary_confidence": pl.Float64,
            "meta_label": pl.Int32,
            "meta_confidence": pl.Float64,
            "raw_bet_size": pl.Float64,
            "discretized_weight": pl.Float64,
            "primary_model_version": pl.Int32,
            "meta_model_version": pl.Int32,
        }
    )


def _discretize_expr(
    raw_col: pl.Expr, tiers: tuple[tuple[float, float], ...]
) -> pl.Expr:
    """Map raw bet size to discretized weight using config tiers.

    Tiers: ((upper_bound, weight), ...) sorted ascending by upper_bound.
    Example default: ((0.10, 0.00), (0.25, 0.15), (0.50, 0.35), (0.75, 0.60),
                      (1.01, 0.85))
    Last tier acts as default (anything >= previous threshold).
    """
    # Start from the highest tier as default
    expr = pl.lit(tiers[-1][1])
    # Iterate in reverse: if raw < upper_bound, use this weight
    for upper_bound, weight in reversed(tiers):
        expr = pl.when(raw_col < upper_bound).then(pl.lit(weight)).otherwise(expr)
    return expr


def compute_bet_sizes(
    predictions: pl.DataFrame,
    *,
    sigma_probability: float,
    config: BetSizingConfig | None = None,
) -> pl.DataFrame:
    """Compute raw and discretized bet sizes for predictions passing both thresholds.

    Steps:
    1. Validate sigma_probability > 0.
    2. Filter by meta_threshold (strictly >) and primary_threshold (strictly >).
    3. z_score = (meta_confidence - 0.5) / sigma_probability
    4. raw_bet_size = clip(2 * Phi(z_score) - 1, 0, 1)
    5. Discretize raw_bet_size into 5 tiers from config.

    Args:
        predictions: DataFrame with columns: signal_id, event_id, symbol, event_ts,
                     primary_side, primary_confidence, meta_label, meta_confidence,
                     primary_model_version, meta_model_version.
        sigma_probability: Standard deviation of meta-label probabilities from OOS
                           calibration. Loaded from MLflow artifact by TRD-49. Must > 0.
        config: BetSizingConfig with thresholds and discretization tiers.

    Returns:
        pl.DataFrame with raw_bet_size and discretized_weight columns added.
        Empty if no predictions pass both thresholds.
    """
    if sigma_probability < 1e-10:
        raise ValueError(
            f"sigma_probability={sigma_probability} is too small (< 1e-10). "
            "Zero sigma indicates degenerate calibration."
        )

    if config is None:
        config = BetSizingConfig()

    # Step 1a: Check for NaN or Null confidences and drop them with a warning
    nan_mask = (
        pl.col("meta_confidence").is_nan() | pl.col("meta_confidence").is_null() |
        pl.col("primary_confidence").is_nan() | pl.col("primary_confidence").is_null()
    )
    nan_count = predictions.select(nan_mask.sum()).item()
    if nan_count > 0:
        logger.warning(
            "Dropping %d predictions with NaN/Null confidence values (possible upstream model failure)",
            nan_count,
        )
        predictions = predictions.filter(~nan_mask)

    # Step 1b: Filter by both thresholds (strictly greater than)
    filtered = predictions.filter(
        (pl.col("meta_confidence") > config.meta_threshold)
        & (pl.col("primary_confidence") > config.primary_threshold)
    )

    if filtered.is_empty():
        return _empty_result()

    # Step 2: Compute z-scores and raw bet sizes via numpy/scipy
    z_scores = (filtered["meta_confidence"].to_numpy() - 0.5) / sigma_probability
    raw_sizes = np.clip(2.0 * norm.cdf(z_scores) - 1.0, 0.0, 1.0)

    # Step 3: Attach raw bet size and discretize
    result = filtered.with_columns(
        pl.Series("raw_bet_size", raw_sizes, dtype=pl.Float64),
    ).with_columns(
        _discretize_expr(pl.col("raw_bet_size"), config.tiers).alias(
            "discretized_weight"
        ),
    )

    return result.select(_BET_SIZING_COLUMNS)
