# BTCUSDT Databricks SQL Control-Tower Dashboard v1 Specification

## 1) Project goal and scope

Build a Databricks SQL-first control-tower dashboard for `BTCUSDT` market-data and offline model observability on Databricks.

The project goal is to demonstrate production-style SQL analytics for an algorithmic trading platform by turning raw minute-level operational signals into a one-page executive monitoring experience. The dashboard must emphasize:

* data reliability
* freshness and latency monitoring
* missing-minute detection
* duplicate detection
* trade-count anomaly detection
* volatility-regime monitoring
* offline model-quality monitoring
* model-vs-baseline edge tracking
* configurable SLA and severity thresholds

v1 scope is intentionally constrained to:

* one symbol: `BTCUSDT`
* one canonical grain: `1-minute`
* one page: `3x3` executive dashboard
* one gold dashboard source: `smart-trading-dev.03_gold.btc_usdt_minute_observability_mart`
* batch refresh only
* managed Delta tables, not views, for materialized datasets

v1 is not a trading PnL dashboard, order-execution dashboard, or online inference latency dashboard. It is a SQL control-tower dashboard for market-data and offline model observability.

## 2) Source context

This specification follows repository guidance from `/Users/nthuan2109@gmail.com/trading-databricks/CLAUDE.md` and extends the existing BTCUSDT research and modeling work already present in the project.

Primary raw source:

* `smart-trading-dev.01_bronze.aggtrades`

Existing notebooks informing the SQL design:

* `BTCUSDT EDA Parameter Optimization` — notebook ID `4155794356659631`
  * establishes bar-quality, sampling, and volatility-oriented research context
  * confirms the project is working from BTCUSDT aggregated trade data
* `BTCUSDT Direction Prediction AutoML` — notebook ID `4155794356659628`
  * establishes the minute-bar, feature, target, and offline model evaluation context
  * informs minute-level prediction outputs, confidence fields, and rolling evaluation metrics

Repository-aligned context:

* catalog: `smart-trading-dev`
* bronze schema: `01_bronze`
* silver schema: `02_silver`
* gold schema: `03_gold`
* v1 uses batch SQL transformations over trading data already landed in Databricks
* the dashboard is an operational SQL deliverable layered on top of the existing research notebooks

## 3) Final v1 table inventory

### Materialized tables

* `smart-trading-dev.02_silver.btc_usdt_minute_base_metrics`
* `smart-trading-dev.02_silver.btc_usdt_minute_model_scoring`
* `smart-trading-dev.03_gold.btc_usdt_minute_observability_mart`
* `smart-trading-dev.03_gold.observability_metric_thresholds`

### Dashboard dataset query

* `btc_usdt_observability_dashboard_v1`

### Intended roles

| Object | Role |
| --- | --- |
| `smart-trading-dev.02_silver.btc_usdt_minute_base_metrics` | Minute-grain market-data reliability and observability foundation built from bronze aggtrades plus a complete minute spine |
| `smart-trading-dev.02_silver.btc_usdt_minute_model_scoring` | Minute-grain offline model outputs, persistence baseline outputs, and rolling evaluation metrics |
| `smart-trading-dev.03_gold.btc_usdt_minute_observability_mart` | Unified dashboard-ready gold mart containing raw metrics, derived KPI fields, status labels, severity codes, and platform health score |
| `smart-trading-dev.03_gold.observability_metric_thresholds` | Versioned SQL config table containing active thresholds and sample-size rules used to resolve KPI statuses |
| `btc_usdt_observability_dashboard_v1` | Final dashboard dataset query reading only from the gold mart and shaping the one-page `3x3` layout |

## 4) Architecture and grain decisions

The following design decisions are finalized for v1.

### Grain and keys

* `BTCUSDT` only
* `1-minute` grain only
* `bar_minute_ts TIMESTAMP` is the logical time key
* `symbol, bar_minute_ts` is the logical composite business key
* every minute-grain table also carries `bar_date DATE`
* every table retains a `symbol` column even though v1 is single-symbol

### Spine and joins

* build a complete minute spine in SQL for the covered time window
* left join minute base metrics onto the minute spine so missing minutes are explicit rows
* keep all minute rows even when model fields are null
* left join model scoring onto base metrics by `symbol, bar_minute_ts`
* dashboard reads only from the gold mart

### Source handling

* deduplicate bronze trades by `agg_trade_id`
* keep both raw and deduplicated trade counts
* define missing minutes from the complete minute spine
* treat empty minutes as missing minutes, not zero-volume trading minutes
* compute latency from the maximum trade timestamp observed within each minute

### Refresh and physical design

* batch refresh, not streaming
* initial history load uses `CREATE OR REPLACE`
* later incremental maintenance can switch to `MERGE`
* use managed Delta tables for all v1 materialized objects
* avoid partitioning in v1 because the dataset is small
* skip liquid clustering in v1
* add table comments documenting purpose and grain
* add column comments in all DDL
* add `run_date` and `pipeline_loaded_at` metadata columns to every materialized table

## 5) Dashboard specification

The dashboard is a one-page executive control tower with a `3x3` layout and a color-first monitoring design.

### Dashboard defaults

* default time window: last `24` hours
* filters:
  * date range
  * severity
* design emphasis:
  * `green / yellow / red`
  * highest-risk issues visually prioritized
  * executives can quickly identify whether the platform is healthy, degraded, or failing

### Page layout

| Position | Widget | Purpose |
| --- | --- | --- |
| Top-left | Overall Platform Health Score | Primary KPI card showing `platform_health_score` on a `0-100` scale plus `platform_health_status` |
| Top-center | Freshness SLA Tile | KPI card showing freshness state derived from `freshness_lag_seconds` and freshness status logic |
| Top-right | Model Edge vs Persistence Tile | KPI card showing rolling uplift versus persistence baseline and `model_edge_status` |
| Middle-left | Missing Minutes Trend | Time-series chart showing missing-minute events across the selected range |
| Middle-center | Trade Count and Anomaly Trend | Time-series chart of minute trade counts with anomaly highlighting from rolling z-score status |
| Middle-right | Volatility Regime Trend | Time-series chart showing `low / normal / high` regime changes and regime status coloring |
| Bottom-left | Confidence Bucket Distribution | Distribution view of `low / medium / high` prediction confidence buckets |
| Bottom-center | Rolling ROC-AUC and Precision | Time-series chart for rolling model quality metrics, nullable where the evaluation window is not valid |
| Bottom-right | Latest 20 Red-Status Minutes | Detail table listing the newest `20` red-status minutes sorted descending by `bar_minute_ts` |

### Dashboard data contract

The saved dashboard dataset query `btc_usdt_observability_dashboard_v1` must read only from:

* `smart-trading-dev.03_gold.btc_usdt_minute_observability_mart`

The dashboard query must not reach back into bronze or silver tables directly.

## 6) Exact threshold and config decisions

Thresholds are externalized to a managed Unity Catalog SQL table seeded with explicit `INSERT` statements.

### Config table design rules

* managed Unity Catalog SQL table
* one row per metric
* seeded with explicit `INSERT` statements
* threshold lookup resolves from the latest active row by `effective_from`
* supports future threshold versioning without rewriting the gold mart logic

### Required config columns

* `metric_name`
* `yellow_threshold`
* `red_threshold`
* `comparison_operator`
* `min_sample_size`
* `effective_from`
* `is_active`

### Finalized threshold logic

| Metric | Yellow | Red | Notes |
| --- | --- | --- | --- |
| Freshness lag | `> 90` seconds | `> 300` seconds | Evaluated from `freshness_lag_seconds` |
| Duplicate rate | `> 0.001` | `> 0.01` | Equivalent to `> 0.1%` yellow and `> 1%` red |
| Missing minute | not used | `true` | Any missing minute is red |
| Late minute | `true` | not used | Any late minute is yellow |
| Trade-count anomaly | `abs(z) >= 2` | `abs(z) >= 3` | Uses `rolling_60m_trade_count_zscore` |
| Model edge uplift | `0 < uplift <= 0.02` | `uplift <= 0` | Green only when uplift `> 0.02` |
| Minimum scored samples | `30` | `30` | Used for model status eligibility via `meets_min_sample_flag` |
| Platform health weighting | `60% data / 40% model` | `60% data / 40% model` | Weighting is fixed in score construction |

### Config interpretation notes

* `comparison_operator` must allow simple metric-specific semantics such as `gt`, `abs_gte`, `boolean_true_red_only`, `boolean_true_yellow_only`, and `uplift_band`
* `min_sample_size` is especially important for model-edge metrics so sparse windows do not incorrectly show green
* missing-minute and late-minute metrics are intentionally boolean-style rules rather than continuous thresholds

## 7) KPI and status definitions

### Core raw and derived KPI definitions

* `freshness_lag_seconds`
  * difference in seconds between `pipeline_loaded_at` and `bar_minute_ts`
  * used for dashboard freshness SLA evaluation
* `data_latency_seconds`
  * difference in seconds between the minute boundary and the latest trade timestamp observed inside that minute
  * used to detect delayed event arrival within the minute itself
* `duplicate_rate`
  * `(raw_trade_count - dedup_trade_count) / raw_trade_count`
  * null-safe handling required when `raw_trade_count = 0`
* `rolling_60m_trade_count_zscore`
  * z-score of minute trade count relative to a `7-day` hour-of-day baseline
* `rolling_60m_volatility`
  * trailing `60-minute` realized volatility metric used for regime classification
* `volatility_regime`
  * categorical value: `low`, `normal`, `high`
* `volatility_regime_code`
  * integer code for regime ordering and dashboard rendering
* `confidence_bucket`
  * prediction-confidence bucket from `predicted_probability`
* `confidence_bucket_code`
  * numeric ordering for confidence-bucket display
* `rolling_model_accuracy`
  * trailing `60-minute` accuracy over scored rows
* `rolling_baseline_accuracy`
  * trailing `60-minute` persistence-baseline accuracy over scored rows
* `rolling_roc_auc`
  * trailing `60-minute` ROC-AUC, nullable when the window does not contain both classes
* `rolling_precision`
  * trailing `60-minute` precision over scored rows
* `rolling_60m_model_uplift`
  * `rolling_model_accuracy - rolling_baseline_accuracy`
* `meets_min_sample_flag`
  * `true` only when at least `30` scored minutes are present in the trailing evaluation window
* `model_edge_status`
  * green only when uplift `> 0.02` and sample threshold is met
  * yellow when uplift is positive but not strong enough
  * red when uplift `<= 0` or sample threshold is not met
* `platform_health_score`
  * composite score on a `0-100` scale
  * weighted `60%` data reliability and `40%` model quality
* `platform_health_status`
  * categorical health interpretation of the score for executive display

### Severity conventions

Status columns use both text labels and numeric severity codes.

* `green = 0`
* `yellow = 1`
* `red = 2`

### Volatility regime and confidence bucket defaults

These defaults should be documented in SQL comments and can be tuned later if necessary.

* `volatility_regime`
  * `low` if volatility is materially below the rolling baseline
  * `normal` if volatility is within the normal operating band
  * `high` if volatility is materially above the rolling baseline
* `volatility_regime_code`
  * `0 = low`
  * `1 = normal`
  * `2 = high`
* `confidence_bucket`
  * `low` when probability `< 0.55`
  * `medium` when probability `>= 0.55 and <= 0.70`
  * `high` when probability `> 0.70`
* `confidence_bucket_code`
  * `0 = low`
  * `1 = medium`
  * `2 = high`

### Baseline definition

The offline baseline is the persistence rule:

* predict the next minute's direction to be the same as the previous realized minute direction

This baseline must be stored explicitly in the model-scoring layer as `baseline_predicted_label` so model-vs-baseline comparison is transparent and queryable.

## 8) Final v1 table schemas

Recommended column order puts metadata fields first, then business keys, then raw metrics, then derived metrics, then statuses.

### 8.1 `smart-trading-dev.03_gold.observability_metric_thresholds`

Table comment:

> Versioned threshold configuration table for BTCUSDT observability KPI status evaluation. One row per metric. Latest active row by `effective_from` is authoritative.

| Column | Type | Comment |
| --- | --- | --- |
| `run_date` | `DATE` | Batch run date that loaded the threshold row |
| `pipeline_loaded_at` | `TIMESTAMP` | Timestamp when the config load executed |
| `metric_name` | `STRING` | Unique metric identifier used by gold status logic |
| `yellow_threshold` | `DOUBLE` | Yellow threshold value when applicable |
| `red_threshold` | `DOUBLE` | Red threshold value when applicable |
| `comparison_operator` | `STRING` | Operator semantics such as `gt`, `abs_gte`, or boolean-style rules |
| `min_sample_size` | `INT` | Minimum sample-size rule for metrics requiring eligibility checks |
| `effective_from` | `TIMESTAMP` | Effective-start timestamp for threshold versioning |
| `is_active` | `BOOLEAN` | Whether the row is currently active for resolution |
| `metric_description` | `STRING` | Human-readable description of what the metric controls |

### 8.2 `smart-trading-dev.02_silver.btc_usdt_minute_base_metrics`

Table comment:

> BTCUSDT `1-minute` base observability metrics built from a complete minute spine and bronze aggtrades. Contains raw and deduplicated market-data reliability signals before status assignment.

| Column | Type | Comment |
| --- | --- | --- |
| `run_date` | `DATE` | Batch run date for the table load |
| `pipeline_loaded_at` | `TIMESTAMP` | Timestamp when the silver load completed |
| `symbol` | `STRING` | Trading symbol, fixed to `BTCUSDT` in v1 |
| `bar_date` | `DATE` | Calendar date derived from `bar_minute_ts` |
| `bar_minute_ts` | `TIMESTAMP` | Canonical minute timestamp and logical time key |
| `raw_trade_count` | `BIGINT` | Count of raw bronze aggtrades in the minute before deduplication |
| `dedup_trade_count` | `BIGINT` | Count of trades remaining after deduplication by `agg_trade_id` |
| `duplicate_trade_count` | `BIGINT` | Difference between raw and deduplicated counts |
| `trade_volume_base` | `DOUBLE` | Total traded base quantity for the minute |
| `notional_usd` | `DOUBLE` | Total dollar notional for the minute |
| `open_price` | `DOUBLE` | Minute open price |
| `high_price` | `DOUBLE` | Minute high price |
| `low_price` | `DOUBLE` | Minute low price |
| `close_price` | `DOUBLE` | Minute close price |
| `minute_return` | `DOUBLE` | Minute close-to-close return |
| `abs_minute_return` | `DOUBLE` | Absolute value of minute return |
| `rolling_60m_volatility` | `DOUBLE` | Trailing `60-minute` realized volatility metric |
| `rolling_60m_trade_count_zscore` | `DOUBLE` | Trade-count anomaly z-score versus `7-day` hour-of-day baseline |
| `minute_max_trade_ts` | `TIMESTAMP` | Latest trade timestamp observed within the minute |
| `data_latency_seconds` | `INT` | Intra-minute latency derived from the max observed trade timestamp |
| `is_missing_minute` | `BOOLEAN` | Whether the minute exists only because of the generated spine |
| `is_late_minute` | `BOOLEAN` | Whether the minute breached the late-arrival rule |

### 8.3 `smart-trading-dev.02_silver.btc_usdt_minute_model_scoring`

Table comment:

> BTCUSDT `1-minute` model-scoring and offline evaluation table containing minute predictions, persistence baseline outputs, confidence buckets, and rolling `60-minute` model metrics.

| Column | Type | Comment |
| --- | --- | --- |
| `run_date` | `DATE` | Batch run date for the model-scoring load |
| `pipeline_loaded_at` | `TIMESTAMP` | Timestamp when the model-scoring load completed |
| `symbol` | `STRING` | Trading symbol, fixed to `BTCUSDT` in v1 |
| `bar_date` | `DATE` | Calendar date derived from `bar_minute_ts` |
| `bar_minute_ts` | `TIMESTAMP` | Canonical minute timestamp for scored predictions |
| `predicted_label` | `INT` | Model-predicted class for the minute target |
| `predicted_probability` | `DOUBLE` | Model confidence score used for bucketing and dashboard display |
| `true_label` | `INT` | Realized label used for offline evaluation |
| `baseline_predicted_label` | `INT` | Persistence-baseline prediction using previous realized minute direction |
| `confidence_bucket` | `STRING` | Confidence category: `low`, `medium`, or `high` |
| `confidence_bucket_code` | `INT` | Numeric ordering for confidence-bucket visualization |
| `scored_minute_count_60m` | `BIGINT` | Number of scored minutes in the trailing `60-minute` window |
| `rolling_model_accuracy` | `DOUBLE` | Trailing `60-minute` model accuracy |
| `rolling_baseline_accuracy` | `DOUBLE` | Trailing `60-minute` persistence-baseline accuracy |
| `rolling_roc_auc` | `DOUBLE` | Trailing `60-minute` ROC-AUC, nullable if the window lacks both classes |
| `rolling_precision` | `DOUBLE` | Trailing `60-minute` precision |
| `rolling_positive_label_rate` | `DOUBLE` | Positive-label share in the trailing evaluation window |
| `rolling_60m_model_uplift` | `DOUBLE` | Accuracy uplift of model versus persistence baseline |
| `meets_min_sample_flag` | `BOOLEAN` | Whether at least `30` scored minutes are available for valid status evaluation |

### 8.4 `smart-trading-dev.03_gold.btc_usdt_minute_observability_mart`

Table comment:

> Unified BTCUSDT `1-minute` gold observability mart for executive dashboarding. Contains raw metrics, derived KPI fields, status labels, severity codes, model edge signals, and platform health score.

| Column | Type | Comment |
| --- | --- | --- |
| `run_date` | `DATE` | Batch run date for the gold mart load |
| `pipeline_loaded_at` | `TIMESTAMP` | Timestamp when the gold mart load completed |
| `symbol` | `STRING` | Trading symbol, fixed to `BTCUSDT` in v1 |
| `bar_date` | `DATE` | Calendar date derived from `bar_minute_ts` |
| `bar_minute_ts` | `TIMESTAMP` | Canonical minute timestamp and dashboard join key |
| `raw_trade_count` | `BIGINT` | Raw minute trade count from silver base metrics |
| `dedup_trade_count` | `BIGINT` | Deduplicated minute trade count from silver base metrics |
| `duplicate_rate` | `DOUBLE` | Duplicate ratio derived from raw and deduplicated trade counts |
| `notional_usd` | `DOUBLE` | Total traded dollar notional for the minute |
| `minute_return` | `DOUBLE` | Minute return |
| `abs_minute_return` | `DOUBLE` | Absolute minute return |
| `rolling_60m_trade_count_zscore` | `DOUBLE` | Trade-count anomaly z-score |
| `rolling_60m_volatility` | `DOUBLE` | Trailing realized volatility |
| `freshness_lag_seconds` | `INT` | Seconds between `pipeline_loaded_at` and `bar_minute_ts` |
| `data_latency_seconds` | `INT` | Intra-minute data latency from event timestamp behavior |
| `is_missing_minute` | `BOOLEAN` | Missing-minute indicator from the generated spine |
| `is_late_minute` | `BOOLEAN` | Late-arriving-minute indicator |
| `volatility_regime` | `STRING` | Volatility regime: `low`, `normal`, `high` |
| `volatility_regime_code` | `INT` | Numeric ordering for regime rendering |
| `predicted_label` | `INT` | Model-predicted class |
| `predicted_probability` | `DOUBLE` | Model confidence score |
| `true_label` | `INT` | Realized class label |
| `baseline_predicted_label` | `INT` | Persistence-baseline prediction |
| `confidence_bucket` | `STRING` | Confidence category |
| `confidence_bucket_code` | `INT` | Numeric ordering for confidence category |
| `rolling_model_accuracy` | `DOUBLE` | Trailing `60-minute` model accuracy |
| `rolling_baseline_accuracy` | `DOUBLE` | Trailing `60-minute` baseline accuracy |
| `rolling_roc_auc` | `DOUBLE` | Trailing `60-minute` ROC-AUC |
| `rolling_precision` | `DOUBLE` | Trailing `60-minute` precision |
| `rolling_60m_model_uplift` | `DOUBLE` | Model accuracy uplift versus persistence baseline |
| `meets_min_sample_flag` | `BOOLEAN` | Whether minimum scored-sample eligibility is met |
| `freshness_status` | `STRING` | SLA status for freshness |
| `freshness_severity_code` | `INT` | Numeric severity for freshness |
| `duplicate_rate_status` | `STRING` | Status for duplicate-rate health |
| `duplicate_rate_severity_code` | `INT` | Numeric severity for duplicate-rate health |
| `missing_data_status` | `STRING` | Status summarizing missing-minute and late-minute conditions |
| `missing_data_severity_code` | `INT` | Numeric severity for missing-data status |
| `trade_count_anomaly_status` | `STRING` | Status for trade-count anomaly detection |
| `trade_count_anomaly_severity_code` | `INT` | Numeric severity for trade-count anomalies |
| `volatility_regime_status` | `STRING` | Status associated with the volatility regime classification |
| `volatility_regime_severity_code` | `INT` | Numeric severity for volatility-regime status |
| `model_edge_status` | `STRING` | Status for model-vs-baseline uplift |
| `model_edge_severity_code` | `INT` | Numeric severity for model-edge status |
| `platform_health_score` | `DOUBLE` | Composite `0-100` executive health score |
| `platform_health_status` | `STRING` | Executive health interpretation |
| `platform_health_severity_code` | `INT` | Numeric severity for platform health |

## 9) Status logic summary

### Data reliability statuses

* `freshness_status`
  * green when `freshness_lag_seconds <= 90`
  * yellow when `freshness_lag_seconds > 90 and <= 300`
  * red when `freshness_lag_seconds > 300`
* `duplicate_rate_status`
  * green when `duplicate_rate <= 0.001`
  * yellow when `duplicate_rate > 0.001 and <= 0.01`
  * red when `duplicate_rate > 0.01`
* `missing_data_status`
  * red when `is_missing_minute = true`
  * otherwise yellow when `is_late_minute = true`
  * otherwise green
* `trade_count_anomaly_status`
  * green when `abs(rolling_60m_trade_count_zscore) < 2`
  * yellow when `abs(rolling_60m_trade_count_zscore) >= 2 and < 3`
  * red when `abs(rolling_60m_trade_count_zscore) >= 3`

### Volatility status

Initial v1 interpretation:

* `normal` regime maps to green
* `low` regime maps to yellow
* `high` regime maps to red

This mapping should remain configurable in a future iteration if operating requirements change.

### Model-edge status

* if `meets_min_sample_flag = false`, treat the minute as red for model eligibility purposes
* if `rolling_60m_model_uplift > 0.02` and `meets_min_sample_flag = true`, status is green
* if `rolling_60m_model_uplift > 0 and <= 0.02` and `meets_min_sample_flag = true`, status is yellow
* if `rolling_60m_model_uplift <= 0`, status is red

### Platform health score

The score is a weighted composite on a `0-100` scale.

Implementation intent:

* derive a data-health subscore from the data reliability and volatility status fields
* derive a model-health subscore from model-edge and evaluation fields
* combine them as:
  * `platform_health_score = 0.60 * data_health_subscore + 0.40 * model_health_subscore`

Recommended initial display bands:

* green when score `>= 80`
* yellow when score `>= 60 and < 80`
* red when score `< 60`

## 10) SQL implementation plan

### File locations

All build SQL should live under the `queries` folder.

### Required SQL files

| File | Purpose |
| --- | --- |
| `queries/01_config.sql` | Create and seed `smart-trading-dev.03_gold.observability_metric_thresholds` with explicit `INSERT` statements |
| `queries/02_base_metrics.sql` | Build `smart-trading-dev.02_silver.btc_usdt_minute_base_metrics` from `smart-trading-dev.01_bronze.aggtrades` plus a complete minute spine |
| `queries/03_model_scoring.sql` | Build `smart-trading-dev.02_silver.btc_usdt_minute_model_scoring` from minute-level scoring outputs aligned to `bar_minute_ts` |
| `queries/04_gold_mart.sql` | Build `smart-trading-dev.03_gold.btc_usdt_minute_observability_mart` by joining silver tables and resolving active thresholds |

### Dashboard query

Create a saved query named:

* `btc_usdt_observability_dashboard_v1`

The query should:

* read only from `smart-trading-dev.03_gold.btc_usdt_minute_observability_mart`
* support the one-page `3x3` dashboard layout
* default to the last `24` hours
* expose the date-range and severity filters required by the dashboard
* support the latest `20` red-status minutes table sorted newest first

### Build order

1. create and seed config table
2. build minute base metrics table
3. build minute model-scoring table
4. build unified gold mart
5. create dashboard dataset query
6. build dashboard from the saved query

### Refresh pattern

* use `CREATE OR REPLACE` for the initial history backfill
* move to `MERGE` for incremental refresh once the table contracts stabilize
* keep v1 as scheduled batch SQL, not streaming SQL

## 11) Minimal assumptions and remaining open questions

Major design decisions are finalized. Only a few implementation-level assumptions remain.

* The exact minute-level upstream source for model predictions must be promoted into SQL-accessible form if it is not already materialized by the modeling workflow.
* The initial volatility-regime cutoffs should be documented explicitly in SQL comments during implementation; the regime labels are finalized even if the precise numeric boundaries may be tuned later.
* The recommended platform-health display bands are `>= 80` green, `>= 60 and < 80` yellow, and `< 60` red unless stakeholder review changes them.

## 12) Final v1 summary

v1 is a Databricks SQL control-tower dashboard for `BTCUSDT` that combines market-data observability and offline model monitoring into a single minute-grain gold mart. It is backed by managed Delta tables, a versioned threshold config table, a one-page `3x3` executive dashboard, and SQL-first status logic that remains configurable without changing visualization code.

The resulting deliverable is intentionally portfolio-ready: it demonstrates SQL data modeling, KPI design, SLA management, model-vs-baseline monitoring, and dashboard-first operational analytics on Databricks.