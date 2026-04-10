# ML QA Notebook 99: E2E Training Job Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the logical equivalent of `run_training_job` end-to-end against real bars and log the best model as an MLflow artifact only (no UC registry push).

**Architecture:** An end-to-end pipeline execution encapsulated in a single notebook. Exercises CUSUM event detection, labeling, feature union building, hyperparameter search, and model wrapping inside an MLflow component run context. Avoids writing to Unity Catalog registries and logs artifacts locally.

**Tech Stack:** Python 3.12, Polars, MLflow, scikit-learn, Plotly

---

### Task 1: Create Notebook 99

**Files:**
- Create: `notebooks/qa/ml/99_e2e_training_job.py`

- [ ] **Step 1: Write the notebook script**

```python
"""
ml/99_e2e_training_job.py — QA notebook for E2E Training Job.

Run: uv run python notebooks/qa/ml/99_e2e_training_job.py
"""
import time
import polars as pl
import pandas as pd
import dataclasses
from notebooks.qa.utils import (
    mlflow_component_run, load_raw_dollar_bars, log_diag, log_plotly, build_scatter
)
from trading_databricks.ml.config import MLPipelineConfig, TrainingConfig
from trading_databricks.ml.cusum import detect_cusum_events
from trading_databricks.ml.labeling import compute_triple_barrier_labels
from trading_databricks.ml.transformers.pipeline import build_training_pipeline
from trading_databricks.ml.transformers.search import build_hyperparameter_search
from trading_databricks.ml.training.sample_weights import compute_sample_weights
from trading_databricks.ml.transformers.mlflow_model import TradingPipelineModel
import mlflow

base_config = MLPipelineConfig()
config = dataclasses.replace(base_config, training=TrainingConfig(n_iter_search=5))

with mlflow_component_run(
    layer="ml", component="e2e_training_job",
    params={"registry_push": "false"}
):
    durations = {}
    
    # Load
    t0 = time.time()
    bars = load_raw_dollar_bars(symbol="BTCUSDT", limit=100_000)
    durations["load"] = time.time() - t0
    
    # CUSUM
    t0 = time.time()
    events = detect_cusum_events(bars, config=config.cusum)
    durations["cusum"] = time.time() - t0
    
    # Labeling
    t0 = time.time()
    labels = compute_triple_barrier_labels(events, bars, config=config.label)
    durations["labeling"] = time.time() - t0
    
    if labels.is_empty():
        raise RuntimeError("No labels generated for training.")
        
    # Pipeline Build
    t0 = time.time()
    weights = compute_sample_weights(labels)
    pipeline, splitter = build_training_pipeline(config, bars, events, labels)
    durations["pipeline_build"] = time.time() - t0
    
    # Search Fit
    t0 = time.time()
    search = build_hyperparameter_search(pipeline, splitter, config.training)
    search.fit(events, labels["label"].to_numpy(), classifier__sample_weight=weights)
    durations["search_fit"] = time.time() - t0
    
    # Artifact Log
    t0 = time.time()
    model = TradingPipelineModel(pipeline=search.best_estimator_, config=config)
    mlflow.pyfunc.log_model(artifact_path="model", python_model=model)
    durations["artifact_log"] = time.time() - t0

    # Section 3 — Diagnostic tables
    log_diag("stage_durations", pl.DataFrame([durations]))
    
    row_counts = {
        "n_bars": bars.height,
        "n_events": events.height,
        "n_labels": labels.height,
        "n_cv_folds": splitter.get_n_splits(),
        "n_search_iters": config.training.n_iter_search
    }
    log_diag("stage_row_counts", pl.DataFrame([row_counts]))
    
    cv_res_df = pl.from_pandas(pd.DataFrame(search.cv_results_))
    log_diag("final_cv_results", cv_res_df)
    log_diag("best_params_report", pl.DataFrame([search.best_params_]))
    
    # Log metrics
    mlflow.log_metrics({
        "best_cv_score": search.best_score_,
        "n_bars": bars.height,
        "n_events": events.height,
        "n_labels": labels.height
    })
    for k, v in durations.items():
        mlflow.log_metric(f"stage_duration_{k}", v)

    # Section 4 — Visual inspection
    log_plotly("cv_score_vs_rank", build_scatter(cv_res_df, x="rank_test_score", y="mean_test_score"))
    
    # Section 5 — Sanity assertions
    assert search.best_estimator_ is not None
    
    run = mlflow.active_run()
    assert run.data.tags.get("registered_model_name") is None, "QA must not push to UC"
    assert "best_cv_score" in run.data.metrics
```

- [ ] **Step 2: Run test to verify it executes successfully**

Run: `uv run python notebooks/qa/ml/99_e2e_training_job.py`
Expected: Exit code 0.

- [ ] **Step 3: Commit**

```bash
git add notebooks/qa/ml/99_e2e_training_job.py
git commit -m "test(ml): add 99_e2e_training_job QA notebook"
```