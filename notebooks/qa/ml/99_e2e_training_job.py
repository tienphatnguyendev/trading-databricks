"""
ml/99_e2e_training_job.py — QA notebook for E2E Training Job.

Run: uv run python notebooks/qa/ml/99_e2e_training_job.py
"""
import dataclasses
import time

import mlflow
import pandas as pd
import polars as pl

from notebooks.qa.utils import (
    build_scatter,
    load_raw_dollar_bars,
    log_diag,
    log_plotly,
    mlflow_component_run,
)
from trading_databricks.ml.config import MLPipelineConfig, TrainingConfig
from trading_databricks.ml.cusum import detect_cusum_events
from trading_databricks.ml.labeling import compute_triple_barrier_labels
from trading_databricks.ml.training.sample_weights import compute_sample_weights
from trading_databricks.ml.transformers.mlflow_model import TradingPipelineModel
from trading_databricks.ml.transformers.pipeline import build_training_pipeline
from trading_databricks.ml.transformers.search import build_hyperparameter_search

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
