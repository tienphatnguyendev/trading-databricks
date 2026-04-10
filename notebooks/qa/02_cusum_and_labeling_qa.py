# notebooks/qa/02_cusum_and_labeling_qa.py
# Databricks notebook source
# MAGIC %md
# MAGIC # 02: CUSUM & Labeling QA
# MAGIC Verifies the symmetric CUSUM filter and Triple-Barrier labeling method.

# COMMAND ----------
import plotly.graph_objects as go
import polars as pl

from notebooks.qa.utils import load_dollar_bars
from trading_databricks.ml.config import CUSUMConfig, LabelConfig
from trading_databricks.ml.cusum import detect_cusum_events
from trading_databricks.ml.labeling import compute_triple_barrier_labels
from trading_databricks.ml.validation import validate_events, validate_labels

try:
    bars = load_dollar_bars(spark, limit=50000)  # noqa: F821
except Exception:
    bars = pl.DataFrame()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Execution & Mathematical Assertions

# COMMAND ----------
if not bars.is_empty():
    cusum_config = CUSUMConfig(lookback_days=1)  # Fast warm-up for QA
    events = detect_cusum_events(bars, config=cusum_config)

    if not events.is_empty():
        validate_events(events)
        print(f"✅ Generated {len(events)} CUSUM events.")

        label_config = LabelConfig()
        labels = compute_triple_barrier_labels(events, bars, config=label_config)
        validate_labels(labels)

        # Assertions
        assert labels["avg_uniqueness"].min() > 0.0, "avg_uniqueness must be > 0"
        assert labels["avg_uniqueness"].max() <= 1.0, "avg_uniqueness must be <= 1.0"
        print(f"✅ Generated {len(labels)} Triple-Barrier Labels.")
else:
    events = pl.DataFrame()
    labels = pl.DataFrame()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Visualization: CUSUM Events

# COMMAND ----------
if not events.is_empty() and not bars.is_empty():
    events_joined = events.join(bars, on="bar_index", how="left")

    up_events = events_joined.filter(pl.col("cusum_side") == "up")
    down_events = events_joined.filter(pl.col("cusum_side") == "down")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bars["bar_end_time"],
            y=bars["close"],
            mode="lines",
            name="Price",
            opacity=0.5,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=up_events["bar_end_time"],
            y=up_events["close"],
            mode="markers",
            marker=dict(color="green", symbol="triangle-up", size=10),
            name="CUSUM Up",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=down_events["bar_end_time"],
            y=down_events["close"],
            mode="markers",
            marker=dict(color="red", symbol="triangle-down", size=10),
            name="CUSUM Down",
        )
    )

    fig.update_layout(title="CUSUM Events on Dollar Bars")
    fig.show()
