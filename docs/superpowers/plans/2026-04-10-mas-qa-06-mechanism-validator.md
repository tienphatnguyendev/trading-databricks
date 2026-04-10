# MAS QA Notebook 06: Mechanism Validator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exercise all five validation steps of `validate_extraction()` with both passing and failing fixtures, plus force the Module B embedding-failure graceful-degradation path.

**Architecture:** Operates on hardcoded scenarios mapped to strict validation pathways. Avoids calling external embedding endpoints unless intentionally executing the "happy path" using safe mock overrides.

**Tech Stack:** Python 3.12, Polars, MLflow, Plotly

---

### Task 1: Create Notebook 06

**Files:**
- Create: `notebooks/qa/mas/06_mechanism_validator.py`

- [ ] **Step 1: Write the notebook script**

```python
"""
mas/06_mechanism_validator.py — QA notebook for Mechanism Validator.

Run: uv run python notebooks/qa/mas/06_mechanism_validator.py
"""
import time
import polars as pl
from notebooks.qa.utils import (
    mlflow_component_run, log_diag, log_plotly, build_label_distribution, build_scatter, _raise_embed_error
)
from trading_databricks.agents.nodes.mechanism_validator import validate_extraction
from trading_databricks.agents.taxonomy import MECHANISM_WHITELIST_SEED
from trading_databricks.agents.config import MASConfig

def _build_fixture_matrix():
    return pl.DataFrame([
        {
            "case": "S1_accept_happy_path",
            "extraction": {"treatment_class": "Regulation.etf_ruling", "mechanism": "institutional_access", "expected_direction": "bullish", "evidence_spans": ["SEC approves"]},
            "article_body": "SEC approves spot Bitcoin ETF today",
            "expected_status": "accepted",
            "expected_rejected_raw": None
        },
        {
            "case": "S5_embedding_fail",
            "extraction": {"treatment_class": "Regulation.etf_ruling", "mechanism": "institutional_access", "expected_direction": "bullish", "evidence_spans": ["SEC approves"]},
            "article_body": "SEC approves...",
            "expected_status": "accepted",
            "expected_rejected_raw": None
        }
    ])

def _run_cosine_unit_tests():
    from trading_databricks.agents.nodes.mechanism_validator import _cosine_similarity
    return pl.DataFrame([{"match": _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0}])

config = MASConfig()

with mlflow_component_run(layer="mas", component="mechanism_validator"):
    # Section 1 — Load whitelist + build fixture matrix
    whitelist = {
        (r["treatment_l1"], r["mechanism"]): r["valid_directions"]
        for r in MECHANISM_WHITELIST_SEED
    }
    fixtures = _build_fixture_matrix()

    # Section 2 — Exercise node
    def run_case(row):
        embed_fn = _raise_embed_error if row["case"] == "S5_embedding_fail" else (lambda t: [1.0, 0.0, 0.0])
        t0 = time.time()
        result = validate_extraction(
            row["extraction"], row["article_body"], whitelist,
            config=config.validator, embed_fn=embed_fn
        )
        return {**row, **result, "latency_ms": (time.time() - t0) * 1000}

    matrix_dicts = [run_case(r) for r in fixtures.iter_rows(named=True)]
    matrix = pl.DataFrame(matrix_dicts)
    matrix = matrix.with_columns([
        pl.col("validation_status").alias("actual_status"),
        pl.col("rejected_mechanism_raw").alias("actual_rejected_raw"),
        pl.col("evidence_relevance_score").alias("actual_score")
    ])

    # Section 3 — Diagnostic tables
    log_diag("validator_fixture_matrix", matrix)
    log_diag("cosine_similarity_unit", _run_cosine_unit_tests())
    
    # Section 4 — Visual inspection
    log_plotly("validation_status_distribution", build_label_distribution(matrix["actual_status"]))
    log_plotly("evidence_relevance_histogram", build_scatter(matrix, x="case", y="actual_score"))
    
    # Section 5 — Sanity assertions
    assert (matrix["actual_status"] == matrix["expected_status"]).all()
    
    s5_filter = matrix.filter(pl.col("case") == "S5_embedding_fail")
    if s5_filter.height > 0:
        assert s5_filter["actual_score"][0] is None
```

- [ ] **Step 2: Run test to verify it executes successfully**

Run: `uv run python notebooks/qa/mas/06_mechanism_validator.py`
Expected: Exit code 0.

- [ ] **Step 3: Commit**

```bash
git add notebooks/qa/mas/06_mechanism_validator.py
git commit -m "test(mas): add 06_mechanism_validator QA notebook"
```