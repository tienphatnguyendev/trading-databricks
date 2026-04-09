# notebooks/qa/05_inference_and_bet_sizing_qa.py
# Databricks notebook source
# MAGIC %md
# MAGIC # 05: Inference & Bet Sizing QA
# MAGIC Verifies the discretization of meta-label probabilities into bet sizes.

# COMMAND ----------
from datetime import UTC, datetime

import numpy as np
import plotly.express as px
import polars as pl

from trading_databricks.ml.config import BetSizingConfig
from trading_databricks.ml.inference.bet_sizing import compute_bet_sizes
from trading_databricks.ml.inference.position import compute_positions

# Generate dummy predictions covering full 0-1 probability spectrum
n_preds = 1000
mock_preds = pl.DataFrame({
    "signal_id": [f"sig_{i}" for i in range(n_preds)],
    "event_id": [f"evt_{i}" for i in range(n_preds)],
    "symbol": ["BTCUSDT"] * n_preds,
    "event_ts": [datetime.now(UTC)] * n_preds,
    "primary_side": [1] * n_preds,
    "primary_confidence": np.random.uniform(0.5, 1.0, n_preds),
    "meta_label": [1] * n_preds,
    "meta_confidence": np.linspace(0.0, 1.0, n_preds),
    "primary_model_version": [1] * n_preds,
    "meta_model_version": [1] * n_preds,
})

# COMMAND ----------
# MAGIC %md
# MAGIC ## Execution & Mathematical Assertions

# COMMAND ----------
config = BetSizingConfig(meta_threshold=0.5, primary_threshold=0.55)
bet_sizes = compute_bet_sizes(mock_preds, sigma_probability=0.1, config=config)

if not bet_sizes.is_empty():
    assert bet_sizes.filter(pl.col("meta_confidence") <= 0.5).is_empty(), (
        "Thresholding failed!"
    )
    print("✅ Bet sizes threshold assertions passed.")

    positions = compute_positions(bet_sizes)
    assert not positions.is_empty(), "Positions failed to generate."
    print("✅ Positions generated successfully.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Visualization: Bet Sizing Discretization Step-Function

# COMMAND ----------
if not bet_sizes.is_empty():
    df = bet_sizes.to_pandas()
    fig = px.scatter(
        df,
        x="meta_confidence",
        y="discretized_weight",
        color="raw_bet_size",
        title="Meta Confidence vs Discretized Bet Weight",
        labels={
            "discretized_weight": "Bet Weight",
            "meta_confidence": "Probability (Meta-Model)",
        },
    )
    fig.show()
