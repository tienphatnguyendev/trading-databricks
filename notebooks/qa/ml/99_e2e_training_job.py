"""
ml/99_e2e_training_job.py — QA notebook for E2E Training Job.

Run: uv run python notebooks/qa/ml/99_e2e_training_job.py
"""

import dataclasses
import time

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from notebooks.qa.utils import (
    build_candlestick,
    build_heatmap,
    build_histogram,
    build_label_distribution,
    build_scatter,
    build_triple_barrier_chart,
    load_raw_dollar_bars,
    log_dataset,
    log_diag,
    log_plotly,
    mlflow_component_run,
)
from trading_databricks.ml.config import CUSUMConfig, MLPipelineConfig, TrainingConfig
from trading_databricks.ml.cusum import detect_cusum_events
from trading_databricks.ml.labeling import compute_triple_barrier_labels
from trading_databricks.ml.training.sample_weights import compute_sample_weights
from trading_databricks.ml.transformers.mlflow_model import TradingPipelineModel
from trading_databricks.ml.transformers.pipeline import build_training_pipeline
from trading_databricks.ml.transformers.search import build_hyperparameter_search

mlflow.sklearn.autolog(
    log_models=False,
    log_input_examples=False,
    silent=True,
    max_tuning_runs=None,
)

def print_header(title: str):
    print(f"\n{'=' * 80}\n{title.upper()}\n{'=' * 80}")


def print_observation(text: str):
    print(f"🔍 OBSERVATION: {text}")


def print_comparison(text: str):
    print(f"⚖️  COMPARISON: {text}")


base_config = MLPipelineConfig()
config = dataclasses.replace(base_config, training=TrainingConfig(n_iter_search=5), cusum=CUSUMConfig(lookback_days=1))

with mlflow_component_run(
    layer="ml", component="e2e_training_job", params={"registry_push": "false"}
):
    durations = {}

    print_header("Step 1: Load Raw Dollar Bars")
    t0 = time.time()
    bars = load_raw_dollar_bars(symbol="BTCUSDT", limit=100_000)
    log_dataset(bars, name="dollar_bars", source="trading-signal-gen-dev.gold.dollar_bars")
    durations["load"] = time.time() - t0

    print_observation(
        f"Loaded {bars.height:,} raw dollar bars for symbol {config.symbol}."
    )
    if bars.height > 0:
        dt_min, dt_max = bars["bar_end_time"].min(), bars["bar_end_time"].max()
        dt_span = dt_max - dt_min
        days_span = dt_span.total_seconds() / (24 * 3600)
        bars_per_day = bars.height / days_span if days_span > 0 else 0

        print_observation(
            f"Data spans {dt_span} ({days_span:.2f} days). First bar at {dt_min}, last bar at {dt_max}."
        )
        print_observation(f"Average density: {bars_per_day:.2f} bars/day.")
        print_comparison(
            "Dollar bars sample the market proportionally to volume/notional exchanged, stabilizing variance compared to regular time bars (which suffer from heteroscedasticity)."
        )

        log_plotly("step1_bars_candlestick_sample", build_candlestick(bars.head(500)))

    print_header("Step 2: CUSUM Event Detection")
    t0 = time.time()
    events = detect_cusum_events(bars, config=config.cusum)
    durations["cusum"] = time.time() - t0

    print_observation(
        f"Detected {events.height:,} CUSUM events based on a threshold multiplier of {config.cusum.threshold_multiplier}."
    )
    retention_rate = events.height / bars.height if bars.height > 0 else 0
    print_comparison(
        f"CUSUM retention rate: {retention_rate:.2%} of total bars triggered an event."
    )
    print_comparison(
        "This adaptive sampling technique reduces the dataset to only moments of structural break, discarding uninformative low-volatility periods."
    )

    if events.height > 1:
        # Plot overlay of events on price series for the first 1000 bars
        bars_sample = bars.head(1000).to_pandas()
        max_bar_index = bars_sample["bar_index"].max()
        events_sample = events.filter(pl.col("bar_index") <= max_bar_index).to_pandas()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=bars_sample["bar_end_time"],
                y=bars_sample["close"],
                mode="lines",
                name="Close Price",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=events_sample["event_ts"],
                y=events_sample["close_bar"]
                if "close_bar" in events_sample.columns
                else bars_sample.set_index("bar_index")
                .loc[events_sample["bar_index"]]["close"]
                .values,
                mode="markers",
                marker=dict(color="red", size=8, symbol="x"),
                name="CUSUM Event",
            )
        )
        fig.update_layout(
            title="CUSUM Trigger Points Overlay", xaxis_title="Time", yaxis_title="Price"
        )
        log_plotly("step2_cusum_overlay", fig)

        time_diffs = events["event_ts"].diff().drop_nulls()
        time_diffs_sec = time_diffs.dt.total_seconds()
        log_plotly(
            "step2_cusum_interval_histogram",
            build_histogram(
                time_diffs_sec, bins=50, title="CUSUM Inter-event Time (seconds)"
            ),
        )

    print_header("Step 3: Triple Barrier Labeling")
    t0 = time.time()
    labels = compute_triple_barrier_labels(events, bars, config=config.label)
    durations["labeling"] = time.time() - t0

    if labels.is_empty():
        raise RuntimeError("No labels generated for training.")

    print_observation(f"Computed {labels.height:,} labels across barriers.")
    label_rate = labels.height / events.height if events.height > 0 else 0
    print_comparison(
        f"Effective labeling rate: {label_rate:.2%} of CUSUM events were successfully labeled (matured before the end of dataset)."
    )

    # Label Distribution
    log_plotly("step3_label_distribution", build_label_distribution(labels["label"]))
    label_counts = labels["label"].value_counts().to_pandas()
    print_observation(f"Label distribution:\n{label_counts.to_string(index=False)}")

    # Hit type pie chart
    hit_counts = labels["barrier_hit"].value_counts().to_pandas()
    fig_hits = px.pie(
        hit_counts, names="barrier_hit", values="count", title="Barrier Hit Types"
    )
    log_plotly("step3_barrier_hit_pie", fig_hits)

    avg_holding = (labels["matured_at"] - labels["event_ts"]).mean()
    print_observation(f"Average holding period for a labeled trade is {avg_holding}.")

    # Visualize the triple barrier trajectories
    log_plotly("step3_triple_barrier_trajectories", build_triple_barrier_chart(events, labels, bars, n_samples=5))
    print_observation("Logged interactive triple barrier trajectories visualization showing price paths and barrier hits.")

    print_header("Step 4: Pipeline Assembly & Feature Extraction")
    t0 = time.time()
    weights = compute_sample_weights(labels)
    log_plotly(
        "step4_sample_weights_histogram",
        build_histogram(pl.Series(weights), title="Sample Weights Distribution"),
    )
    print_observation(
        "Computed sample weights arrays to resolve label class imbalance and overlapping returns (concurrency)."
    )
    print_observation(
        f"Sample weight range: [min: {np.min(weights):.4f}, max: {np.max(weights):.4f}, mean: {np.mean(weights):.4f}]"
    )

    pipeline, splitter = build_training_pipeline(config, bars, events, labels)
    durations["pipeline_build"] = time.time() - t0

    print_observation(
        "Constructed sklearn Pipeline integrating Technical, Volatility, and Volume features, and converted to Numpy arrays via internal bridge."
    )

    # Extract features for correlation analysis
    features_union = pipeline.named_steps["features"]
    features_df = features_union.fit_transform(events, bars=bars)
    numeric_features = features_df.select(
        [c for c in features_df.columns if features_df[c].dtype.is_numeric()]
    )
    if numeric_features.shape[1] > 0:
        log_plotly(
            "step4_feature_correlation_heatmap",
            build_heatmap(numeric_features, corr=True),
        )
        print_observation(
            f"Extracted {numeric_features.shape[1]} numeric features. Logged feature correlation heatmap to investigate multicollinearity."
        )

    print_header("Step 5: Baseline Model Evaluation")
    t0 = time.time()
    y = labels["label"].to_numpy()
    X = pipeline.named_steps["to_numpy"].fit_transform(features_df)

    # Baseline 1: Dummy Classifier
    dummy = DummyClassifier(strategy="prior")
    dummy_scores = cross_val_score(dummy, X, y, cv=splitter, scoring="accuracy", n_jobs=1)
    dummy_acc = np.mean(dummy_scores)

    # Baseline 2: Naive RF on single basic feature
    # We will just use the first column of X (often a raw return or simple indicator)
    X_naive = X[:, :1]
    naive_rf = RandomForestClassifier(
        random_state=config.training.random_state, n_estimators=20
    )
    naive_scores = cross_val_score(
        naive_rf, X_naive, y, cv=splitter, scoring="accuracy", n_jobs=1
    )
    naive_acc = np.mean(naive_scores)

    print_observation(f"Dummy Classifier (Majority Class) CPCV Accuracy: {dummy_acc:.4f}")
    print_observation(f"Naive RF (Single Feature) CPCV Accuracy: {naive_acc:.4f}")
    durations["baselines"] = time.time() - t0

    print_header("Step 6: Hyperparameter Search & Final Evaluation")
    t0 = time.time()
    search = build_hyperparameter_search(pipeline, splitter, config.training)
    search.fit(events, y, classifier__sample_weight=weights)
    durations["search_fit"] = time.time() - t0

    print_observation(
        f"Hyperparameter Combinatorial Purged CV grid search completed over {config.training.n_iter_search} iterations."
    )
    print_comparison(f"Best Tuned Pipeline CV Score: {search.best_score_:.4f}")
    print_comparison(
        f"Alpha generated by feature pipeline vs Dummy: {search.best_score_ - dummy_acc:+.4f}"
    )
    print_comparison(
        f"Alpha generated by feature pipeline vs Naive RF: {search.best_score_ - naive_acc:+.4f}"
    )
    print_observation(f"Best parameters: {search.best_params_}")

    print_header("Step 7: Artifact Logging & Diagnostics")
    t0 = time.time()
    model = TradingPipelineModel(pipeline=search.best_estimator_, config=config)
    mlflow.pyfunc.log_model(artifact_path="model", python_model=model)
    durations["artifact_log"] = time.time() - t0
    print_observation(
        "Logged final model to MLflow as a PyFunc artifact containing the fitted feature pipeline and estimator."
    )

    log_diag("stage_durations", pl.DataFrame([durations]))

    row_counts = {
        "n_bars": bars.height,
        "n_events": events.height,
        "n_labels": labels.height,
        "n_cv_folds": splitter.get_n_splits(),
        "n_search_iters": config.training.n_iter_search,
        "n_features": numeric_features.shape[1],
    }
    log_diag("stage_row_counts", pl.DataFrame([row_counts]))

    cv_res_df = pl.from_pandas(pd.DataFrame(search.cv_results_))
    log_diag("final_cv_results", cv_res_df)
    log_diag("best_params_report", pl.DataFrame([search.best_params_]))

    # Log metrics
    mlflow.log_metrics(
        {
            "best_cv_score": search.best_score_,
            "baseline_dummy_acc": dummy_acc,
            "baseline_naive_acc": naive_acc,
            "n_bars": bars.height,
            "n_events": events.height,
            "n_labels": labels.height,
        }
    )
    for k, v in durations.items():
        mlflow.log_metric(f"stage_duration_{k}", v)

    log_plotly(
        "cv_score_vs_rank",
        build_scatter(
            cv_res_df, x="rank_test_score", y="mean_test_score", color="mean_test_score"
        ),
    )
    print_observation(
        "Logged 'cv_score_vs_rank' scatter plot evaluating convergence of the hyperparameter search."
    )

    print_header("Sanity assertions")
    assert search.best_estimator_ is not None

    run = mlflow.active_run()
    assert run.data.tags.get("registered_model_name") is None, "QA must not push to UC"
    assert "best_cv_score" in run.data.metrics

    print_observation("Sanity assertions passed. Pipeline QA run completed successfully.")
