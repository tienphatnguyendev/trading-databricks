# MAS QA Notebook 01: Gatekeeper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exercise the Stage-1 spaCy EntityRuler NER filter on a real sample and measure crypto-entity coverage across the dataset.

**Architecture:** A plain `.py` script runnable locally via Databricks Connect. Uses shared QA utilities for MLflow logging and Plotly charts. Extracts NER features and evaluates `is_crypto_relevant`.

**Tech Stack:** Python 3.12, Polars, MLflow, Plotly, spaCy

---

### Task 1: Create Notebook 01

**Files:**
- Create: `notebooks/qa/mas/01_gatekeeper.py`

- [ ] **Step 1: Write the notebook script**

```python
"""
mas/01_gatekeeper.py — QA notebook for Gatekeeper (Stage 1) component.

Run: uv run python notebooks/qa/mas/01_gatekeeper.py
"""
import polars as pl
from notebooks.qa.utils import (
    mlflow_component_run, load_article_sample, log_diag, log_plotly,
    summarize, build_label_distribution, build_histogram
)
from trading_databricks.agents.gatekeeper import is_crypto_relevant, extract_ner_features, build_nlp
from trading_databricks.agents.config import MASConfig

config = MASConfig()

with mlflow_component_run(
    layer="mas", component="gatekeeper",
    params={"sample_size": 200, "text_window": config.gatekeeper.text_window, "ner_only_mode": True}
):
    # Section 1 — Load input
    articles = load_article_sample(n=200)

    # Section 2 — Exercise node
    nlp = build_nlp()
    rows = []
    for a in articles:
        doc = nlp(a["headline"] + " " + a["body"][:config.gatekeeper.text_window])
        feats = extract_ner_features(doc)
        is_impactful, conf = is_crypto_relevant(a["headline"], a["body"], config=config.gatekeeper)
        rows.append({**a, **feats, "is_impactful": is_impactful, "confidence": conf})
    
    df = pl.DataFrame(rows)

    # Section 3 — Diagnostic tables
    log_diag("ner_per_article", df)
    
    summary_cols = ["is_impactful", "entity_count", "has_crypto_asset", "has_exchange", "has_stablecoin", "has_protocol", "has_concept"]
    log_diag("ner_coverage_summary", summarize(df, summary_cols))

    # Section 4 — Visual inspection
    if df.height > 0:
        log_plotly("entity_distribution", build_label_distribution(df["has_crypto_asset"]))
        log_plotly("category_coverage", build_label_distribution(df["is_impactful"]))
        log_plotly("entity_count_histogram", build_histogram(df["entity_count"]))

    # Section 5 — Sanity assertions
    assert df["is_impactful"].dtype == pl.Boolean
    assert df["confidence"].min() >= 0.0 and df["confidence"].max() <= 1.0
    assert build_nlp() is build_nlp(), "build_nlp must be cached"
```

- [ ] **Step 2: Run test to verify it executes successfully**

Run: `uv run python notebooks/qa/mas/01_gatekeeper.py`
Expected: Exit code 0.

- [ ] **Step 3: Commit**

```bash
git add notebooks/qa/mas/01_gatekeeper.py
git commit -m "test(mas): add 01_gatekeeper QA notebook"
```