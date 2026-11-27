"""Build and publish the December BTCUSDT benchmark on Databricks."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep

import mlflow
import polars as pl
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

from trading_databricks.ml.december_persistence import (
    build_hourly_dataset,
    score_persistence,
)

SOURCE = "smart_trading_dev.01_bronze.raw_aggtrades"
ROOT = Path(__file__).resolve().parents[1]
BAR_QUERY = ROOT / "queries" / "btcusdt_december_bars.sql"
BAR_SCHEMA = {
    "bar_end_us": pl.Int64,
    "last_trade_us": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "trade_count": pl.Int64,
    "duplicate_rows": pl.Int64,
    "conflicting_trade_ids": pl.Int64,
}


def load_bars(
    workspace: WorkspaceClient, *, warehouse_id: str, source_version: int
) -> tuple[pl.DataFrame, str]:
    """Read the ordered and deduplicated bar result from SQL."""
    if source_version < 0:
        raise ValueError("The Delta source version must be nonnegative.")
    sql = BAR_QUERY.read_text(encoding="utf-8").replace(
        "__SOURCE_VERSION__", str(source_version)
    )
    execution = workspace.statement_execution
    response = execution.execute_statement(
        warehouse_id=warehouse_id, statement=sql, wait_timeout="20s"
    )
    statement_id = response.statement_id
    if statement_id is None:
        raise RuntimeError("The SQL API returned no statement ID.")
    deadline = monotonic() + 300
    while response.status and response.status.state in {
        StatementState.PENDING,
        StatementState.RUNNING,
    }:
        if monotonic() >= deadline:
            execution.cancel_execution(statement_id)
            raise TimeoutError(f"Bar query timed out: {statement_id}")
        sleep(2)
        response = execution.get_statement(statement_id)

    if response.status is None or response.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"Bar query {statement_id} failed: {response.status}")
    if (
        response.manifest is None
        or response.manifest.schema is None
        or response.manifest.truncated
        or response.result is None
    ):
        raise RuntimeError(f"Bar query {statement_id} returned an incomplete result.")
    rows = list(response.result.data_array or [])
    for index in range(1, response.manifest.total_chunk_count or 1):
        chunk = execution.get_statement_result_chunk_n(statement_id, index)
        rows.extend(chunk.data_array or [])
    if len(rows) != response.manifest.total_row_count:
        raise RuntimeError(f"Bar query {statement_id} returned an unexpected row count.")
    columns = tuple(column.name for column in response.manifest.schema.columns)
    if columns != tuple(BAR_SCHEMA):
        raise RuntimeError(f"Unexpected five-minute bar columns: {columns}")
    bars = pl.DataFrame(rows, schema=list(BAR_SCHEMA), orient="row").cast(BAR_SCHEMA)
    return bars, statement_id


def summarize(predictions: pl.DataFrame, *groups: str) -> pl.DataFrame:
    """Aggregate the hour-level scores for reporting."""
    return (
        predictions.group_by(*groups)
        .agg(
            pl.len().alias("forecasts"),
            pl.col("qlike").mean(),
            pl.col("absolute_volatility_error_bps").mean().alias("volatility_mae_bps"),
        )
        .sort(*groups)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="dbx-codex")
    parser.add_argument("--warehouse-id", required=True)
    parser.add_argument("--source-version", required=True, type=int)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-volume", required=True)
    args = parser.parse_args()
    output_volume = args.output_volume.rstrip("/")
    if not output_volume.startswith("/Volumes/"):
        parser.error("--output-volume must be a managed Unity Catalog volume path.")

    workspace = WorkspaceClient(profile=args.profile)
    bars, statement_id = load_bars(
        workspace, warehouse_id=args.warehouse_id, source_version=args.source_version
    )
    if bars.height != 8928 or bars["conflicting_trade_ids"].sum():
        raise ValueError(
            "The December bars have incomplete coverage or conflicting trades."
        )
    duplicate_rows = bars["duplicate_rows"].sum()
    input_rows = bars["trade_count"].sum() + duplicate_rows
    if input_rows != 33_782_084:
        raise ValueError(f"Expected 33,782,084 input trades, received {input_rows:,}.")

    dataset = build_hourly_dataset(bars)
    predictions = score_persistence(dataset)
    splits = summarize(predictions, "split")
    expected = {"train": 383, "validation": 168, "test": 168}
    if dict(splits.select("split", "forecasts").iter_rows()) != expected:
        raise ValueError(f"Unexpected December forecast coverage: {splits.to_dicts()}")
    daily = summarize(
        predictions.with_columns(pl.col("forecast_at").dt.date().alias("utc_date")),
        "split",
        "utc_date",
    )

    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    mlflow.set_tracking_uri(f"databricks://{args.profile}")
    experiment = mlflow.set_experiment(args.experiment)
    with (
        TemporaryDirectory() as temp,
        mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name="BTCUSDT December 2025 persistence",
        ) as run,
    ):
        destination = f"{output_volume}/btcusdt/2025-12/{run.info.run_id}"
        metadata = {
            "model": "last_hour_persistence",
            "source": SOURCE,
            "source_version": args.source_version,
            "sql_statement_id": statement_id,
            "git_revision": revision,
            "batch_start_utc": "2025-12-01T00:00:00Z",
            "batch_end_utc_exclusive": "2026-01-01T00:00:00Z",
            "validation_start_utc": "2025-12-18T00:00:00Z",
            "test_start_utc": "2025-12-25T00:00:00Z",
            "prediction_floor": 1e-12,
            "input_trade_rows": input_rows,
            "duplicate_trade_rows": duplicate_rows,
            "dataset_path": destination,
        }
        mlflow.log_params(metadata)
        mlflow.log_metrics(
            {
                f"{row['split']}_{metric}": row[metric]
                for row in splits.iter_rows(named=True)
                for metric in ("forecasts", "qlike", "volatility_mae_bps")
            }
        )
        artifact_dir = Path(temp)
        bars.write_parquet(artifact_dir / "five_minute_bars.parquet")
        dataset.write_parquet(artifact_dir / "hourly_forecast_dataset.parquet")
        predictions.write_parquet(artifact_dir / "persistence_predictions.parquet")
        daily.write_csv(artifact_dir / "daily_scores.csv")
        (artifact_dir / "manifest.json").write_text(
            json.dumps({**metadata, "scores": splits.to_dicts()}, indent=2) + "\n",
            encoding="utf-8",
        )
        workspace.files.create_directory(destination)
        for artifact in sorted(artifact_dir.iterdir()):
            with artifact.open("rb") as contents:
                workspace.files.upload(f"{destination}/{artifact.name}", contents)
        mlflow.log_artifacts(temp)

    print(
        json.dumps(
            {
                "experiment_id": experiment.experiment_id,
                "run_id": run.info.run_id,
                "dataset_path": destination,
                "scores": splits.to_dicts(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
