"""
ml/03_training_assembly.py — QA notebook for training dataset assembly.

Run: uv run python notebooks/qa/ml/03_training_assembly.py
"""
import mlflow
import polars as pl

from notebooks.qa.utils import (
    build_heatmap,
    build_histogram,
    build_scatter,
    load_raw_dollar_bars,
    log_dataset,
    log_diag,
    log_plotly,
    mlflow_component_run,
)
from trading_databricks.common.uc_identifiers import DOLLAR_BARS_FQN
from trading_databricks.ml.config import CUSUMConfig, MLPipelineConfig
from trading_databricks.ml.cusum import detect_cusum_events
from trading_databricks.ml.features.assembly import assemble_training_dataset
from trading_databricks.ml.features.news import compute_news
from trading_databricks.ml.features.technical import compute_technical
from trading_databricks.ml.features.volatility import compute_volatility
from trading_databricks.ml.features.volume import compute_volume
from trading_databricks.ml.labeling import compute_triple_barrier_labels
from trading_databricks.ml.schemas import TRAINING_DATASET_SCHEMA
from trading_databricks.ml.validation import validate_training_dataset


def row_counts_report(events, labels, feature_frames, training_df):
    return pl.DataFrame({
        "n_events": [events.height],
        "n_labels": [labels.height],
        "n_features_tech": [feature_frames[0].collect().height],
        "n_features_vol": [feature_frames[1].collect().height],
        "n_features_volume": [feature_frames[2].collect().height],
        "n_features_news": [feature_frames[3].collect().height],
        "n_joined": [training_df.height]
    })

def schema_diff_report(training_df):
    expected_cols = set(TRAINING_DATASET_SCHEMA.fieldNames())
    actual_cols = set(training_df.columns)
    missing = list(expected_cols - actual_cols)
    extra = list(actual_cols - expected_cols)
    return pl.DataFrame({
        "missing_columns": [", ".join(missing) if missing else "None"],
        "extra_columns": [", ".join(extra) if extra else "None"]
    })

config = MLPipelineConfig()

with mlflow_component_run(
    layer="ml", component="training_assembly",
    params={
        "cpcv.n_groups": config.cpcv.n_groups,
        "input.n_bars": 50000
    }
):
    # Section 1 — Load input
    bars = load_raw_dollar_bars(symbol="BTCUSDT", limit=50_000)
    log_dataset(bars, name="dollar_bars", source=DOLLAR_BARS_FQN)
    events = detect_cusum_events(bars, config=CUSUMConfig(lookback_days=1))
    labels = compute_triple_barrier_labels(events, bars, config=config.label)

    # Section 2 — Exercise component
    tech = compute_technical(events.lazy(), bars.lazy(), config=config.feature)
    vol = compute_volatility(events.lazy(), bars.lazy(), config=config.feature)
    volu = compute_volume(events.lazy(), bars.lazy(), config=config.feature)
    news_zero = compute_news(events.lazy(), None, config=config.feature)

    feature_frames = [tech, vol, volu, news_zero]
    training_df = assemble_training_dataset(events.lazy(), labels.lazy(), feature_frames)

    # Log params discovered during execution
    mlflow.log_params({"input.n_events": events.height, "input.n_labels": labels.height})

    # Section 3 — Diagnostic tables
    log_diag("assembly_row_counts", row_counts_report(events, labels, feature_frames, training_df))

    missing_join_report = training_df.filter(pl.col("label").is_null())
    log_diag("missing_join_report", missing_join_report)

    cpcv_dist = training_df.group_by("cpcv_group").agg(pl.len().alias("n")).sort("cpcv_group")
    log_diag("cpcv_group_distribution", cpcv_dist)

    log_diag("schema_diff_report", schema_diff_report(training_df))

    version_val = training_df["dataset_version"][0] if training_df.height > 0 else "N/A"
    log_diag("dataset_version_report", pl.DataFrame({"dataset_version": [version_val]}))

    # Log metrics
    mlflow.log_metrics({
        "assembled.n_rows": training_df.height,
        "assembled.n_cols": len(training_df.columns),
        "assembled.label_null_count": missing_join_report.height
    })

    # Section 4 — Visual inspection
    if training_df.height > 0:
        log_plotly("cpcv_group_bar", build_scatter(cpcv_dist, x="cpcv_group", y="n"))
        # 2-D heatmap of cpcv_group x label counts
        cpcv_label_counts = training_df.group_by(["cpcv_group", "label"]).agg(pl.len().alias("count"))
        if cpcv_label_counts.height > 0:
            cpcv_label_pivot = cpcv_label_counts.pivot(index="cpcv_group", on="label", values="count").fill_null(0)
            log_plotly("label_vs_group_heatmap", build_heatmap(cpcv_label_pivot, corr=False))
        log_plotly("event_ts_histogram", build_histogram(training_df["event_ts"]))

    # Section 5 — Sanity assertions
    assert training_df.height > 0
    assert set(training_df["cpcv_group"].drop_nulls().unique().to_list()).issubset({1,2,3,4,5,6})
    assert training_df.filter(pl.col("cpcv_group").is_null()).is_empty()
    assert training_df["dataset_version"].n_unique() == 1

    # Verify monotonic chronological grouping
    assert training_df.sort("event_ts")["cpcv_group"].is_sorted(descending=False)

    # Strict validation schema
    validate_training_dataset(training_df, n_groups=config.cpcv.n_groups)
