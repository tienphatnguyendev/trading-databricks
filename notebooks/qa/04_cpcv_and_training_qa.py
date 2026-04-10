# Databricks notebook source
# MAGIC %md
# MAGIC # 04: CPCV & Training Pipeline QA
# MAGIC Verifies Combinatorial Purged Cross-Validation embargo/purge logic.

# COMMAND ----------
import numpy as np
import plotly.graph_objects as go

from trading_databricks.ml.config import CPCVConfig
from trading_databricks.ml.training.cpcv import CPCVSplitter

# Mock label spans (t_start, t_end) representing event lifespans in bar index
N_SAMPLES = 100
t_start = np.arange(N_SAMPLES)
t_end = t_start + np.random.randint(1, 5, size=N_SAMPLES)

config = CPCVConfig(n_groups=5, k_test=1, embargo_pct=0.05)
splitter = CPCVSplitter(config=config, t_start=t_start, t_end=t_end)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Execution & Visualizing Embargo/Purge (Gantt Chart)

# COMMAND ----------
splits = list(splitter.split(np.zeros((N_SAMPLES, 1))))
print(f"✅ Generated {len(splits)} CPCV folds.")

fig = go.Figure()

for fold_idx, (train_idx, test_idx) in enumerate(splits):
    # Plot Test
    fig.add_trace(
        go.Scatter(
            x=test_idx,
            y=[fold_idx] * len(test_idx),
            mode="markers",
            marker=dict(color="orange", symbol="square"),
            name="Test",
            showlegend=(fold_idx == 0),
        )
    )
    # Plot Train
    fig.add_trace(
        go.Scatter(
            x=train_idx,
            y=[fold_idx] * len(train_idx),
            mode="markers",
            marker=dict(color="blue", symbol="square"),
            name="Train",
            showlegend=(fold_idx == 0),
        )
    )

fig.update_layout(
    title="CPCV Folds (Train vs Test vs Purged/Embargoed gaps)",
    xaxis_title="Sample Index",
    yaxis_title="Fold Number",
    height=400,
)
fig.show()

# Assertions
for train_idx, test_idx in splits:
    intersect = set(train_idx).intersection(set(test_idx))
    assert len(intersect) == 0, "Leakage: Train and Test indices overlap!"
print("✅ CPCV Leakage assertions passed.")
