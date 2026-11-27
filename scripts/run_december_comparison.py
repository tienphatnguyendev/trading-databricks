"""Compare the December BTCUSDT forecasts using the published hourly dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import polars as pl
import xgboost
from databricks.sdk import WorkspaceClient

from trading_databricks.ml.december_comparison import compare_models

SOURCE_RUN = "d81f0937b75a4dd587197b0bb1c9f76d"
DATASET_SHA256 = "21f72dc89bf23abe83d3563eabd22d0d4b769191fc0c167ed85507a2ca87c8f6"
EXPERIMENT = "/Shared/BTCUSDT December 2025 Persistence"


def summarize_test(
    predictions: pl.DataFrame,
) -> tuple[dict[str, float], pl.DataFrame, pl.DataFrame]:
    """Compare the same forecast hours and measure sensitivity to each UTC day."""
    expected = pl.datetime_range(
        datetime(2025, 12, 25, tzinfo=UTC),
        datetime(2026, 1, 1, tzinfo=UTC),
        interval="1h",
        closed="left",
        eager=True,
    )
    if not predictions["forecast_at"].sort().equals(expected):
        raise ValueError("The final evaluation requires all 168 matching forecast hours.")
    paired = predictions.with_columns(
        utc_date=pl.col("forecast_at").dt.date(),
        qlike_delta=pl.col("challenger_qlike") - pl.col("baseline_qlike"),
    )
    means = [
        pl.col(name).mean()
        for name in (
            "baseline_qlike",
            "challenger_qlike",
            "qlike_delta",
            "baseline_abs_vol_error_bps",
            "challenger_abs_vol_error_bps",
        )
    ]
    daily = (
        paired.group_by("utc_date").agg(pl.len().alias("hours"), *means).sort("utc_date")
    )
    omitted = daily.select(
        pl.col("utc_date").alias("omitted_utc_date"),
        (
            (
                (pl.col("qlike_delta") * pl.col("hours")).sum()
                - pl.col("qlike_delta") * pl.col("hours")
            )
            / (pl.col("hours").sum() - pl.col("hours"))
        ).alias("qlike_delta"),
    )
    summary = paired.select(pl.len().alias("test_hours"), *means).row(0, named=True)
    summary["days_challenger_won"] = daily.filter(pl.col("qlike_delta") < 0).height
    summary["leave_one_day_out_min_delta"] = omitted["qlike_delta"].min()
    summary["leave_one_day_out_max_delta"] = omitted["qlike_delta"].max()
    return summary, daily, omitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="dbx-codex")
    args = parser.parse_args()
    workspace = WorkspaceClient(profile=args.profile)
    mlflow.set_tracking_uri(f"databricks://{args.profile}")
    source = mlflow.get_run(SOURCE_RUN)
    if (
        source.info.status != "FINISHED"
        or source.data.params.get("source_version") != "5"
    ):
        raise ValueError(
            "Expected the finished December dataset run from Delta version 5."
        )
    dataset_path = f"{source.data.params['dataset_path']}/hourly_forecast_dataset.parquet"
    with workspace.files.download(dataset_path).contents as contents:
        dataset_bytes = contents.read()
    if hashlib.sha256(dataset_bytes).hexdigest() != DATASET_SHA256:
        raise ValueError(
            "The December forecast dataset differs from the verified snapshot."
        )
    dataset = pl.read_parquet(BytesIO(dataset_bytes))
    if (
        dataset.height != 719
        or dataset.filter(pl.col("max_feature_trade_at") >= pl.col("forecast_at")).height
    ):
        raise ValueError(
            "The source dataset has unexpected coverage or forecast cutoffs."
        )

    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    experiment = mlflow.set_experiment(EXPERIMENT)
    with (
        TemporaryDirectory() as temp,
        mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name="BTCUSDT December model comparison",
        ) as run,
    ):
        mlflow.log_params(
            {
                "source_run": SOURCE_RUN,
                "source_version": 5,
                "dataset_path": dataset_path,
                "dataset_sha256": DATASET_SHA256,
                "git_revision": revision,
                "validation_start_utc": "2025-12-18",
                "holdout_start_utc": "2025-12-25",
                "xgboost_version": xgboost.__version__,
                "lightgbm_version": version("lightgbm"),
                "scikit_learn_version": version("scikit-learn"),
            }
        )
        result = compare_models(dataset)
        summary, daily, omitted = summarize_test(result.test_predictions)
        mlflow.log_metrics(summary)
        payload = {"selection": result.selection.to_dicts(), "holdout": summary}
        files = Path(temp)
        result.validation_scores.write_csv(files / "validation_scores.csv")
        result.test_predictions.write_parquet(files / "holdout_predictions.parquet")
        daily.write_csv(files / "holdout_daily_scores.csv")
        omitted.write_csv(files / "holdout_leave_one_day_out.csv")
        (files / "comparison.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        mlflow.log_artifacts(temp)
        print(json.dumps({"run_id": run.info.run_id, **payload}, indent=2))


if __name__ == "__main__":
    main()
