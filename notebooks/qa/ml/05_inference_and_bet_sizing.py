"""
ml/05_inference_and_bet_sizing.py — QA notebook for inference and bet sizing.

Run: uv run python notebooks/qa/ml/05_inference_and_bet_sizing.py
"""
from datetime import UTC, datetime

import mlflow
import numpy as np
import polars as pl

from notebooks.qa.utils import (
    build_label_distribution,
    build_scatter,
    log_diag,
    log_plotly,
    mlflow_component_run,
)
from trading_databricks.ml.config import BetSizingConfig
from trading_databricks.ml.inference.bet_sizing import compute_bet_sizes
from trading_databricks.ml.inference.position import compute_positions


def build_mock_predictions(n: int = 1000) -> pl.DataFrame:
    np.random.seed(42)
    # Generate healthy spectrum of confidences
    primary_conf = np.random.uniform(0.5, 1.0, n)
    meta_conf = np.linspace(0.0, 1.0, n)

    df1 = pl.DataFrame({
        "signal_id": [f"sig_{i}" for i in range(n)],
        "event_id": [f"evt_{i}" for i in range(n)],
        "symbol": ["BTCUSDT"] * n,
        "event_ts": [datetime.now(UTC)] * n,
        "primary_side": [1] * n,  # Long only
        "primary_confidence": primary_conf,
        "meta_label": [1] * n,
        "meta_confidence": meta_conf,
        "primary_model_version": [1] * n,
        "meta_model_version": [1] * n,
    })

    # Inject NaN / Nulls to test filtration
    df2 = pl.DataFrame({
        "signal_id": ["sig_nan1", "sig_nan2"],
        "event_id": ["evt_nan1", "evt_nan2"],
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "event_ts": [datetime.now(UTC), datetime.now(UTC)],
        "primary_side": [1, 1],
        "primary_confidence": [float("nan"), 0.9],
        "meta_label": [1, 1],
        "meta_confidence": [0.8, None],
        "primary_model_version": [1, 1],
        "meta_model_version": [1, 1],
    })
    return pl.concat([df1, df2])

def preds_summary(preds):
    n_nan_meta = preds["meta_confidence"].is_nan().sum() + preds["meta_confidence"].is_null().sum()
    n_nan_primary = preds["primary_confidence"].is_nan().sum() + preds["primary_confidence"].is_null().sum()
    n_nan = n_nan_meta + n_nan_primary
    return pl.DataFrame({
        "n_total": [preds.height],
        "n_nan_or_null": [n_nan],
        "n_below_meta_thresh": [(preds["meta_confidence"] <= BetSizingConfig().meta_threshold).sum()],
        "n_below_primary_thresh": [(preds["primary_confidence"] <= BetSizingConfig().primary_threshold).sum()]
    })

config = BetSizingConfig()

with mlflow_component_run(
    layer="ml", component="inference_and_bet_sizing",
    params={
        "bet_sizing.meta_threshold": config.meta_threshold,
        "bet_sizing.primary_threshold": config.primary_threshold,
        "bet_sizing.tiers": str(config.tiers),
        "sigma_probability": 0.1
    }
):
    # Section 1 — Load input
    preds = build_mock_predictions(n=1000)
    log_diag("raw_predictions_summary", preds_summary(preds))

    # Section 2 — Exercise component
    bet_sizes = compute_bet_sizes(preds, sigma_probability=0.1, config=config)
    positions = compute_positions(bet_sizes)

    # Section 3 — Diagnostic tables
    bet_summary = pl.DataFrame({
        "n_passed": [bet_sizes.height],
        "raw_bet_size_mean": [bet_sizes["raw_bet_size"].mean() if bet_sizes.height else 0.0],
        "raw_bet_size_std": [bet_sizes["raw_bet_size"].std() if bet_sizes.height else 0.0]
    })
    log_diag("bet_sizes_summary", bet_summary)

    tier_mapping = []
    for upper_bound, expected_weight in reversed(config.tiers):
        mask = bet_sizes["discretized_weight"] == expected_weight
        subset = bet_sizes.filter(mask)
        tier_mapping.append({
            "expected_weight": expected_weight,
            "upper_bound": upper_bound,
            "observed_weight_min": subset["discretized_weight"].min() if subset.height else None,
            "observed_weight_max": subset["discretized_weight"].max() if subset.height else None,
            "n_in_tier": subset.height
        })
    log_diag("tier_mapping_report", pl.DataFrame(tier_mapping))
    log_diag("positions_report", positions)

    mlflow.log_metrics({
        "predictions.n_total": preds.height,
        "predictions.n_passed": bet_sizes.height,
        "bet_sizes.raw_mean": bet_sizes["raw_bet_size"].mean() if bet_sizes.height else 0.0,
        "positions.weight": positions["weight"][0] if positions.height else 0.0
    })

    # Section 4 — Visual inspection
    if bet_sizes.height > 0:
        log_plotly("meta_conf_vs_discretized",
                   build_scatter(bet_sizes, x="meta_confidence", y="discretized_weight", color="raw_bet_size"))
        # Step plot (using scatter, marker mode)
        log_plotly("raw_vs_discretized_step",
                   build_scatter(bet_sizes.sort("raw_bet_size"), x="raw_bet_size", y="discretized_weight"))
    if positions.height > 0:
        log_plotly("position_side_bar", build_label_distribution(positions["side"]))

    # Section 5 — Sanity assertions
    assert bet_sizes.filter(pl.col("meta_confidence") <= config.meta_threshold).is_empty()
    assert bet_sizes.filter(pl.col("primary_confidence") <= config.primary_threshold).is_empty()
    assert (bet_sizes["raw_bet_size"].is_between(0, 1)).all()
    assert set(bet_sizes["discretized_weight"].unique().to_list()).issubset({t[1] for t in config.tiers})
    assert positions.height == 1
    assert positions["side"][0] in {"long", "flat"}
