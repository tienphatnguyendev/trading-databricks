"""
ml/01_cusum_and_labeling.py — QA notebook for CUSUM and labeling.

Run: uv run python notebooks/qa/ml/01_cusum_and_labeling.py
"""
import polars as pl

from notebooks.qa.utils import (
    build_label_distribution,
    build_scatter,
    load_raw_dollar_bars,
    log_dataset,
    log_diag,
    log_plotly,
    mlflow_component_run,
    summarize,
)
from trading_databricks.ml.config import CUSUMConfig, LabelConfig
from trading_databricks.ml.cusum import compute_cusum_stats, detect_cusum_events
from trading_databricks.ml.labeling import compute_triple_barrier_labels
from trading_databricks.ml.validation import validate_events, validate_labels

cusum_config = CUSUMConfig(lookback_days=1)
label_config = LabelConfig()

flat_params = {
    "cusum.ewma_lambda": cusum_config.ewma_lambda,
    "cusum.threshold_multiplier": cusum_config.threshold_multiplier,
    "cusum.min_inter_event_bars": cusum_config.min_inter_event_bars,
    "cusum.lookback_days": cusum_config.lookback_days,
    "label.upper_barrier_multiplier": label_config.upper_barrier_multiplier,
    "label.lower_barrier_multiplier": label_config.lower_barrier_multiplier,
    "label.vertical_barrier_days": label_config.vertical_barrier_days,
    "label.zero_label_threshold": label_config.zero_label_threshold,
    "input.n_bars": 50000
}

with mlflow_component_run(layer="ml", component="cusum_and_labeling", params=flat_params):
    # Section 1 — Load input
    bars = load_raw_dollar_bars(symbol="BTCUSDT", limit=50_000)
    log_dataset(bars, name="dollar_bars", source="trading-signal-gen-dev.gold.dollar_bars")
    log_diag("bars_input_summary", summarize(bars, ["close", "volume"]))

    # Section 2 — Exercise component
    events = detect_cusum_events(bars, config=cusum_config)
    labels = compute_triple_barrier_labels(events, bars, config=label_config)

    # Section 3 — Diagnostic tables
    events_summary = pl.DataFrame({
        "n_events": [len(events)],
        "n_up": [len(events.filter(pl.col("cusum_side") == "up"))],
        "n_down": [len(events.filter(pl.col("cusum_side") == "down"))],
        "ewma_vol_mean": [events["ewma_vol"].mean() if not events.is_empty() else 0.0],
        "threshold_used_mean": [events["threshold_used"].mean() if not events.is_empty() else 0.0]
    })
    log_diag("events_summary", events_summary)
    log_diag("events_per_day", compute_cusum_stats(events))

    labels_summary = pl.DataFrame({
        "n_labels": [len(labels)],
        "pct_upper": [len(labels.filter(pl.col("barrier_hit") == "upper")) / len(labels) if not labels.is_empty() else 0],
        "pct_lower": [len(labels.filter(pl.col("barrier_hit") == "lower")) / len(labels) if not labels.is_empty() else 0],
        "pct_vertical": [len(labels.filter(pl.col("barrier_hit") == "vertical")) / len(labels) if not labels.is_empty() else 0],
    })
    log_diag("labels_summary", labels_summary)

    # Section 4 — Visual inspection
    if not events.is_empty():
        event_overlay = events.join(
            bars.select("bar_index", "bar_end_time", "close"), on="bar_index", how="left"
        )
        log_plotly("cusum_events_on_price", build_scatter(bars, x="bar_end_time", y="close", overlay=event_overlay))
    if not labels.is_empty():
        log_plotly("label_class_balance", build_label_distribution(labels["label"]))
        log_plotly("barrier_hit_distribution", build_label_distribution(labels["barrier_hit"]))

    # Section 5 — Sanity assertions
    assert events.height > 0, "warm-up guard likely fired"
    assert (events["cusum_side"].is_in(["up", "down"])).all()
    assert labels.height > 0, "no matured events"
    assert (labels["avg_uniqueness"] > 0).all()
    assert (labels["avg_uniqueness"] <= 1.0).all()
    assert (labels["label"].is_in([-1, 0, 1])).all()

    validate_events(events)
    validate_labels(labels)
