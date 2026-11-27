# December BTCUSDT persistence implementation plan

> For agentic workers, use superpowers:subagent-driven-development. The approved design is a batch-only fixed-hour variance forecast from the December 2025 trade snapshot.

1. Build and verify the SQL five-minute bar query in `queries/btcusdt_december_bars.sql`. Pin Delta version 5, order equal timestamps by aggregate trade ID, and reconcile input counts and conflicts.
2. Build the hourly features and persistence scores in Polars. Test forecast cutoffs, missing data, UTC splits, and QLIKE. Use the complete 24-hour history required by the future model comparisons.
3. Publish the five-minute bars, hourly labels and features, predictions, and daily scores to the managed volume. Track the SQL statement ID, code revision, source version, and split metrics in the workspace MLflow experiment.
4. Verify the workspace run and artifacts alongside the repository tests.

Run from the repo root with an authenticated Free Edition profile that has SQL, Files, and MLflow access.

```sh
uv run --locked python scripts/run_december_persistence.py \
  --profile dbx-codex \
  --warehouse-id 02f57952835cb9fe \
  --source-version 5 \
  --experiment '/Shared/BTCUSDT December 2025 Persistence' \
  --output-volume /Volumes/smart_trading_dev/00_volumes/batch_research
```

The benchmark expects 8,928 bars and 719 eligible forecast hours, of which 168 are in the final week. The last forecast starts at 23:00 UTC on December 31 and uses the close from the last five-minute bar of December as its label endpoint.

## Model comparison

Run the comparison against the published hourly dataset and its verified SHA-256.
On macOS, install the Homebrew `libomp` formula before importing the boosters.

```sh
uv run --locked --extra forecasting python scripts/run_december_comparison.py --profile dbx-codex
```

The comparison chooses one simple baseline and one boosted model on December 18–24 using daily expanding training windows. It fits both choices once before December 25 and compares their forecasts on the same 168 test hours. MLflow records the validation grid, the selected configurations, the paired predictions, and the change in average QLIKE when each test day is omitted.
