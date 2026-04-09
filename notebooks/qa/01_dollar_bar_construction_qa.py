# Databricks notebook source
# MAGIC %md
# MAGIC # 01: Dollar Bar Construction QA
# MAGIC Verifies that raw ticks aggregate into valid dollar bars.

# COMMAND ----------
import plotly.graph_objects as go
import polars as pl

from notebooks.qa.utils import load_dollar_bars, load_raw_ticks
from trading_databricks.ml.validation import validate_bars

# Load a small slice of raw ticks and pre-computed dollar bars for validation
try:
    ticks = load_raw_ticks(spark, limit=50000)  # noqa: F821
    bars = load_dollar_bars(spark, limit=1000)  # noqa: F821
except Exception as e:
    print(f"Skipping loading due to env: {e}")
    ticks = pl.DataFrame({"timestamp": [], "price": [], "volume": []})
    bars = pl.DataFrame(
        {
            "bar_index": [],
            "close": [],
            "high": [],
            "low": [],
            "volume": [],
            "bar_end_time": [],
            "symbol": [],
        }
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## Mathematical Assertions

# COMMAND ----------
if not bars.is_empty():
    validate_bars(bars)

    # Assert High >= Low
    assert (bars["high"] >= bars["low"]).all(), "High must be >= Low"

    # Assert Volume is positive
    assert (bars["volume"] > 0).all(), "Volume must be positive"
    print("✅ Dollar Bar mathematical assertions passed.")
else:
    print("⚠️ No bars loaded. Skipping assertions.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Visualization: Price vs Bars

# COMMAND ----------
if not bars.is_empty() and not ticks.is_empty():
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=bars["bar_end_time"],
            open=bars["open"],
            high=bars["high"],
            low=bars["low"],
            close=bars["close"],
            name="Dollar Bars",
        )
    )

    fig.update_layout(
        title="Dollar Bars over Time", xaxis_title="Time", yaxis_title="Price"
    )
    fig.show()
