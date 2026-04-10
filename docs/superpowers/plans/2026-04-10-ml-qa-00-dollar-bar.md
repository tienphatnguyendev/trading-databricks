# ML QA Notebook 00: Dollar Bar Construction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exercise the dollar-bar construction logic that feeds everything downstream.

**Architecture:** A plain `.py` script runnable locally via Databricks Connect. Uses shared QA utilities for MLflow logging and Plotly charts. Currently raises a `NotImplementedError` with a "TRD-5 blocker" message until the underlying function is ported.

**Tech Stack:** Python 3.12, Polars, MLflow, Plotly, PySpark (Databricks Connect)

---

### Task 1: Create Notebook 00

**Files:**
- Create: `notebooks/qa/ml/00_dollar_bar_construction.py`

- [ ] **Step 1: Write the notebook script**

```python
"""
ml/00_dollar_bar_construction.py — QA notebook for dollar bar construction.

Run: uv run python notebooks/qa/ml/00_dollar_bar_construction.py
"""
from notebooks.qa.utils import (
    mlflow_component_run, load_raw_aggtrades, log_diag, log_plotly,
    summarize, build_candlestick,
)
from trading_databricks.common.config import DollarBarConfig
from trading_databricks.ml.validation import validate_bars

# TRD-5 Blocker Check
try:
    from trading_databricks.ticks.gold.dollar_bars import build_dollar_bars
except Exception:
    pass

# We will raise explicitly as requested by the spec until TRD-5 is complete
raise NotImplementedError("Dollar bar construction not yet implemented — see TRD-5 blocker.")

config = DollarBarConfig(threshold=1_000_000.0)

with mlflow_component_run(
    layer="ml", component="dollar_bar_construction",
    params={"dollar_bar.threshold": config.threshold},
):
    # Section 1 — Load input
    ticks = load_raw_aggtrades(limit=500_000)
    log_diag("input_ticks_summary", summarize(ticks, ["price", "quantity", "timestamp"]))

    # Section 2 — Exercise component
    bars = build_dollar_bars(ticks, config=config)

    # Section 3 — Diagnostic tables
    log_diag("bars_summary", summarize(bars, ["close", "volume", "dollar_volume"]))

    # Section 4 — Visual inspection
    log_plotly("bars_candlestick", build_candlestick(bars))

    # Section 5 — Sanity assertions
    assert bars.height > 0, "no bars generated"
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()
    assert bars["bar_index"].diff().drop_nulls().min() > 0
    validate_bars(bars)
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/qa/ml/00_dollar_bar_construction.py
git commit -m "test(ml): add 00_dollar_bar_construction QA notebook with TRD-5 blocker"
```