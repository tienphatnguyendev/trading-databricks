# MAS QA Notebook 05: Causality Extractor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exhaustively exercise all three extraction paths, the JSON-parsing helper, and the majority-vote consensus logic.

**Architecture:** A script running the causality extractor via a mock LLM toggle (`MAS_QA_MOCK_LLM=1`) to prevent API costs during automated testing, while supporting real endpoint validation when triggered natively.

**Tech Stack:** Python 3.12, Polars, MLflow, Plotly

---

### Task 1: Create Notebook 05

**Files:**
- Create: `notebooks/qa/mas/05_causality_extractor.py`

- [ ] **Step 1: Write the notebook script**

```python
"""
mas/05_causality_extractor.py — QA notebook for Causality Extractor.

Run: uv run python notebooks/qa/mas/05_causality_extractor.py
"""
import os
import time
import polars as pl
from notebooks.qa.utils import (
    mlflow_component_run, load_article_sample, log_diag, log_plotly,
    build_scatter, build_label_distribution
)
from trading_databricks.agents.nodes.causality_extractor import causality_extractor, parse_json_response, majority_vote
from trading_databricks.agents.nodes.few_shot_retriever import STATIC_FEW_SHOT_EXAMPLES
from trading_databricks.agents.config import MASConfig

# Default to mock LLM for QA tests to save tokens and ensure offline reliability
os.environ["MAS_QA_MOCK_LLM"] = os.environ.get("MAS_QA_MOCK_LLM", "1")

def _run_parse_unit_tests():
    tests = [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"b": 2}\n```', {"b": 2}),
        ('```\n{"c": 3}\n```', {"c": 3}),
        ('invalid', None),
        ('', None)
    ]
    return pl.DataFrame([
        {"input": t[0], "parsed": str(parse_json_response(t[0])), "match": parse_json_response(t[0]) == t[1]}
        for t in tests
    ])

def _run_majority_vote_unit_tests():
    ex1 = {"treatment_class": "A", "mechanism": "m", "expected_direction": "bullish", "confidence": 0.9}
    res = majority_vote([ex1, ex1, ex1])
    return pl.DataFrame([{"match": res == ex1}])

config = MASConfig()

with mlflow_component_run(
    layer="mas", component="causality_extractor",
    params={"sample_size": 6}
):
    # Section 1 — Load input
    articles = load_article_sample(n=6)
    results_by_path = {}
    
    # Section 2 — Exercise node, 3 paths
    for path in ("fast", "standard", "careful"):
        rows = []
        for a in articles:
            state = {
                "headline": a["headline"],
                "body": a["body"],
                "few_shot_examples": STATIC_FEW_SHOT_EXAMPLES,
                "extraction_path": path
            }
            t0 = time.time()
            out = causality_extractor(state, config)
            latency_ms = (time.time() - t0) * 1000
            
            ext = out.get("extraction") or {}
            rows.append({
                **a,
                "latency_ms": latency_ms,
                "path": path,
                "parse_success": bool(ext),
                "treatment_class": ext.get("treatment_class"),
                "mechanism": ext.get("mechanism"),
                "confidence": ext.get("confidence")
            })
        results_by_path[path] = pl.DataFrame(rows)

    # Section 3 — Diagnostic tables
    for path, df in results_by_path.items():
        log_diag(f"extractor_{path}_outputs", df, metrics={f"{path}_parse_success_rate": df["parse_success"].mean()})

    log_diag("parse_json_response_unit", _run_parse_unit_tests())
    log_diag("majority_vote_unit", _run_majority_vote_unit_tests())

    # Section 4 — Visual inspection
    combined = pl.concat(list(results_by_path.values()))
    if combined.height > 0:
        log_plotly("latency_by_path", build_scatter(combined, x="path", y="latency_ms"))
        log_plotly("parse_success_by_path", build_label_distribution(combined["parse_success"]))
    
    # Section 5 — Sanity assertions
    assert all(df["parse_success"].sum() >= 1 for df in results_by_path.values())
```

- [ ] **Step 2: Run test to verify it executes successfully**

Run: `uv run python notebooks/qa/mas/05_causality_extractor.py`
Expected: Exit code 0.

- [ ] **Step 3: Commit**

```bash
git add notebooks/qa/mas/05_causality_extractor.py
git commit -m "test(mas): add 05_causality_extractor QA notebook"
```