# MAS QA Notebook 02: Context Retriever Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the V1 stub returns the three context fields as `None` and never raises — enforcing the schema contract for downstream nodes.

**Architecture:** A plain `.py` script runnable locally via Databricks Connect. Exercises the context retriever node logic in isolation.

**Tech Stack:** Python 3.12, Polars, MLflow, Plotly

---

### Task 1: Create Notebook 02

**Files:**
- Create: `notebooks/qa/mas/02_context_retriever.py`

- [ ] **Step 1: Write the notebook script**

```python
"""
mas/02_context_retriever.py — QA notebook for Context Retriever.

Run: uv run python notebooks/qa/mas/02_context_retriever.py
"""
import polars as pl
from notebooks.qa.utils import (
    mlflow_component_run, load_article_sample, log_diag, log_plotly, build_label_distribution
)
from trading_databricks.agents.nodes.context_retriever import context_retriever
from trading_databricks.agents.config import MASConfig

config = MASConfig()

with mlflow_component_run(
    layer="mas", component="context_retriever",
    params={"sample_size": 20, "stub_version": "v1"}
):
    # Section 1 — Load input
    articles = load_article_sample(n=20)
    initial_states = [
        {"headline": a["headline"], "body": a["body"], "published_at": a["published_at"]}
        for a in articles
    ]

    # Section 2 — Exercise node
    results = [context_retriever(s, config=config) for s in initial_states]
    df = pl.DataFrame(results)

    # Section 3 — Diagnostic tables
    log_diag("context_retriever_outputs", df)

    # Section 4 — Visual inspection
    if df.height > 0:
        # Show null coverage for context fields
        null_counts = pl.DataFrame({
            "market_vol": [df["market_vol_at_publish"].null_count()],
            "market_trend": [df["market_trend_at_publish"].null_count()],
            "dollar_bar_close": [df["dollar_bar_close"].null_count()]
        })
        log_plotly("null_coverage", build_label_distribution(null_counts.melt()["value"]))

    # Section 5 — Sanity assertions
    assert df["market_vol_at_publish"].null_count() == len(df)
    assert df["market_trend_at_publish"].null_count() == len(df)
    assert df["dollar_bar_close"].null_count() == len(df)
    
    for r, inp in zip(results, articles):
        assert r["headline"] == inp["headline"]
        assert "body" in r and "published_at" in r
```

- [ ] **Step 2: Run test to verify it executes successfully**

Run: `uv run python notebooks/qa/mas/02_context_retriever.py`
Expected: Exit code 0.

- [ ] **Step 3: Commit**

```bash
git add notebooks/qa/mas/02_context_retriever.py
git commit -m "test(mas): add 02_context_retriever QA notebook"
```