07 — Pipeline Operations (Inference Pipeline + Approval Flow)
Cross-references: trading-logic.md (CUSUM/features/models), validation.md (promotion gates),
docs/daily-workflow.md (S3 data model, dev/staging/prod environments),
resources/*.pipeline.yml (SDP pipeline definitions)

Trading Data Flow — Pipeline Sequence
```
[1] Tick Ingestion (SDP) → [2] Dollar Bar Construction (SDP) → [3] CUSUM Event Detection
→ [4] Triple-Barrier Label Computation → [5] Feature Engineering (all families)
→ [6] Feature Assembly → [7] Model Inference (primary) → [8] Meta-Label Inference
→ [9] Bet Sizing → [10] Signal Write → [11] Position Computation
→ [12] Position Write → [13] Recommendation Notification
```
Note: Steps [1]-[2] run continuously as SDP streaming pipelines. Steps [3]-[13] run on a scheduled inference cadence.

Inference Schedule
| Parameter | Value |
|-----------|-------|
| Inference cadence | Every 4 hours (6 runs/day) |
| Cron | `0 */4 * * *` (every 4 hours UTC) |
| Market hours | 24/7 — no holiday skip needed |
| Estimated pipeline duration | 10-20 minutes |
| Human review window | 4 hours from recommendation |
| Position execution | Manual via Binance after approval |

Pipeline Tasks (Lakeflow Job DAG)
```
task_cusum_events
    ↓
task_label_computation (labels for matured events only)
    ↓
task_feature_engineering (all families in parallel)
    ↓
task_feature_assembly (PIT-safe join)
    ↓
task_primary_inference
    ↓
task_meta_label_inference
    ↓
task_bet_sizing
    ↓
task_signal_write
    ↓
task_position_compute
    ↓
task_recommendation_notify
```
Each task is a separate Python script, orchestrated in a single Lakeflow Job defined in resources/ml_pipeline.pipeline.yml.

Tick Ingestion & Bar Construction (Steps 1-2) — Already Implemented
SDP streaming pipeline (resources/tick_pipeline.pipeline.yml).

Auto Loader reads NDJSON from s3://ai-trading-raw-ap-southeast-1/ticks/binance/...

bronze.ticks_raw → silver.ticks_cleaned → gold.dollar_bars (+ volume, imbalance bars).

Runs continuously on serverless compute.

Code: src/trading_databricks/ticks/pipeline.py.

CUSUM Event Detection (Step 3)
Reads gold.dollar_bars, computes log returns.

Applies CUSUM filter with EWMA volatility threshold (see trading-logic.md).

New events → silver.cusum_events.

Stats → silver.cusum_event_stats.

Label Computation (Step 4)
Computes labels for events whose vertical barriers have expired (N_bars old).

Today's events labeled N_bars later (label lag is inherent).

Model trains on historically labeled events; predicts on current (unlabeled) events.

Labels → silver.triple_barrier_labels.

Feature Engineering and Assembly (Steps 5-6)
Each feature family computed independently → respective silver table.

Assembly joins all families per event using PIT-safe temporal predicates.

Assembled matrix → silver.training_datasets with dataset_version.

Inference and Position Computation (Steps 7-11)
Uses currently promoted model from MLflow Model Registry.

Reads latest features for recent CUSUM events → predictions.

Meta-labeling filters → bet sizing translates to position weights.

Constraints: portfolio budget <= 1.0, single-asset max 70%, max 2 concurrent positions.

No settlement delay (instant crypto).

Results → silver.signals and silver.positions.

Recommendation Output (Step 13)
Notification sent after pipeline completes.

Contains: asset (BTC/ETH), side (long), discretized position weight, confidence, model version.

Delivery: Email (Phase 1), Slack/webhook (Phase 2).

Data Quality Checks
Integrated into SDP pipeline via @dp.expect_or_drop expectations.

Additional checks in inference pipeline:
- Dollar bar production rate: bars/hour within [min_expected, max_expected] range.
- Tick ingestion freshness: max gap between S3 file arrivals < 30 minutes.
- Price reasonableness: no single-tick price jumps > 10%.

Quality results → silver.data_quality.

Freshness < threshold → alert, downstream tasks continue with stale data warning.

Completeness failure → downstream tasks skipped for affected asset.

Human Approval Flow

Delivery:
Notification sent after inference pipeline completes.
Includes summary table + link to approval notebook in Databricks.

Approval process:
Developer opens approval notebook in Databricks.
Per position: Approve / Reject (mandatory reason) / Modify (new weight with reason).
Approved → release_status = 'released'.
No action within review window (4 hours) → positions remain pending, NOT executed.

Constraints:
Only developer's personal identity can approve (not service principal).
Every action logged (append-only audit trail in silver.approval_log).
Platform never auto-approves.

Manual Execution
After approval, developer manually executes trades via Binance.
Execution details optionally logged for performance tracking (Phase 1B+).
Phase 1A: manual tracking only.
Phase 2: API execution via Binance SDK with automated logging.

Continuous Data Pipelines (Running Independently)
These SDP pipelines run independently of the inference schedule:

| Pipeline | Resource | Schedule | Tables |
|----------|----------|----------|--------|
| Tick pipeline | tick_pipeline.pipeline.yml | Continuous (streaming) | bronze.ticks_raw → silver.ticks_cleaned → gold.dollar_bars, volume_bars, tick/dollar_imbalance_bars |
| News pipeline | news_pipeline.pipeline.yml | Every 15 minutes | bronze.news_raw → silver.news_cleaned → gold.news_causal_features |
| ML inference | ml_pipeline.pipeline.yml | Every 4 hours | silver.cusum_events → ... → silver.positions |
