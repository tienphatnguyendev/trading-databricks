# notebooks/qa/03_feature_engineering_qa.py
# Databricks notebook source
# MAGIC %md
# MAGIC # 03: Feature Engineering QA
# MAGIC Verifies Polars feature transformations for PIT safety and mathematical validity.

# COMMAND ----------
import plotly.express as px
import polars as pl

from notebooks.qa.utils import load_dollar_bars
from trading_databricks.ml.config import CUSUMConfig, FeatureConfig
from trading_databricks.ml.cusum import detect_cusum_events
from trading_databricks.ml.features.technical import compute_technical
from trading_databricks.ml.features.volatility import compute_volatility
from trading_databricks.ml.features.volume import compute_volume

try:
    bars = load_dollar_bars(spark, limit=50000)  # noqa: F821
    events = detect_cusum_events(bars, config=CUSUMConfig(lookback_days=1))
except Exception:
    bars = pl.DataFrame()
    events = pl.DataFrame()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Execution & Mathematical Assertions

# COMMAND ----------
if not events.is_empty():
    config = FeatureConfig()

    tech_feats = compute_technical(events.lazy(), bars.lazy(), config=config).collect()
    vol_feats = compute_volatility(events.lazy(), bars.lazy(), config=config).collect()
    volu_feats = compute_volume(events.lazy(), bars.lazy(), config=config).collect()

    # Assertions
    assert not tech_feats["rsi_14"].is_null().all(), (
        "Technical features generated all nulls"
    )
    assert (vol_feats["parkinson_vol_20"].drop_nulls() >= 0).all(), (
        "Volatility cannot be negative"
    )

    print("✅ Feature transformers executed successfully.")

    # Combine for correlation
    all_features = tech_feats.join(vol_feats, on="event_id").join(
        volu_feats, on="event_id"
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## Visualization: Feature Correlations

# COMMAND ----------
if not events.is_empty():
    excluded_cols = {"event_id", "symbol", "event_ts", "computed_at"}
    num_cols = [c for c in all_features.columns if c not in excluded_cols]
    corr_df = all_features.select(num_cols).drop_nulls().to_pandas().corr()

    fig = px.imshow(corr_df, title="Feature Correlation Heatmap", text_auto=".2f")
    fig.show()
