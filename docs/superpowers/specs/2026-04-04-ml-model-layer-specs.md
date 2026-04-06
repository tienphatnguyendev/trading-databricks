# ML Model Layer Specification

**Date:** 2026-04-04
**Status:** Approved
**Cross-references:**
- `docs/advanced-ml-modelling/trading-logic.md` (CUSUM, features, models, bet sizing)
- `docs/advanced-ml-modelling/validation.md` (CPCV, PSR, DSR, promotion gates)
- `docs/advanced-ml-modelling/trading-pipeline.md` (inference pipeline, job DAG, schedules)
- `docs/superpowers/specs/2026-03-31-sdp-architecture-design.md` (SDP pipeline contracts)

---

## 1. Overview & Scope

**Platform role:** Databricks is used exclusively for model development and model serving. No trade execution occurs here. The platform's two outputs are:

1. **Batch predictions** written to `silver.signals` — consumed by monitoring, research, and backtesting
2. **REST endpoint** (Databricks Model Serving) — called by the external trading system for live signal generation

**MVP scope (Phase 1):** BTC-only (BTCUSDT), tick microstructure signal model, 4 feature families.

**Phase 2 additions:** ETH (ETHUSDT), news/MAS features (`silver.features_news` from `gold.news_causal_features`).

**Out of scope for this repo:** Position sizing discretisation tiers, portfolio constraints, order execution, approval workflows, trade logging. These belong in the external trading system.

---

## 2. Pipeline Architecture

```
[Continuous SDP]   gold.dollar_bars  ← tick_pipeline.pipeline.yml (already built, runs 24/7)
                         │
                         ▼
[Training Job]     CUSUM detection → Triple-barrier labeling → Feature engineering (4 families)
  every 5 days     → PIT-safe assembly → CPCV → Promotion gates → MLflow Registry
  single-node               resources/ml_training_job.yml
                         │
                         ▼
[Inference Job]    CUSUM detection → Feature engineering → Primary inference
  cron TBD         → Meta-label inference → Bet sizing → silver.signals
  serverless                resources/ml_inference_job.yml
                         │
                         ▼
[Model Serving]    REST endpoint: {symbol, event_ts} → {prediction, probabilities, raw_bet_size}
  always-on                 resources/ml_serving_endpoint.yml
```

Steps [1]–[2] (tick ingestion + bar construction) run continuously as SDP streaming pipelines and are independent of the ML jobs.

---

## 3. Silver Table Contracts

All new silver tables are **symbol-partitioned** from day 1. ETH is added in Phase 2 with zero schema migration required.

### Signal Generation

**`silver.cusum_events`**
| Column | Type | Notes |
|---|---|---|
| event_id | STRING | Primary key. Format: `{symbol}_{bar_index}` |
| symbol | STRING | Partition key. "BTCUSDT" in Phase 1 |
| event_ts | TIMESTAMP | Timestamp of the dollar bar that triggered the event |
| bar_index | LONG | Dollar bar index from gold.dollar_bars |
| log_return | DOUBLE | Log return at event bar: log(close_t / close_{t-1}) |
| cusum_plus | DOUBLE | S+ value at trigger |
| cusum_minus | DOUBLE | S- value at trigger |
| threshold_used | DOUBLE | EWMA vol × sqrt(avg_bars_per_day) at trigger time |
| ewma_vol | DOUBLE | EWMA(λ=0.94) volatility at trigger |
| avg_bars_per_day | DOUBLE | Empirical bars/day over lookback window |
| ingested_at | TIMESTAMP | Pipeline write time |

**`silver.cusum_event_stats`**
| Column | Type | Notes |
|---|---|---|
| symbol | STRING | |
| date | DATE | Calendar date |
| event_count | INT | Events triggered on this date |
| threshold_used | DOUBLE | Threshold value used on this date |
| ewma_vol | DOUBLE | EWMA vol on this date |
| avg_bars_per_day | DOUBLE | |

**`silver.triple_barrier_labels`**
| Column | Type | Notes |
|---|---|---|
| event_id | STRING | FK to silver.cusum_events |
| symbol | STRING | |
| event_ts | TIMESTAMP | |
| label | INT | +1 (upper hit), -1 (lower hit), 0 (vertical expiry) |
| barrier_hit | STRING | "upper" / "lower" / "vertical" |
| return_at_label | DOUBLE | Realised return from event_ts to label_ts |
| upper_barrier | DOUBLE | Profit-taking level (absolute price) |
| lower_barrier | DOUBLE | Stop-loss level (absolute price) |
| vertical_bars | INT | N bars until vertical barrier expiry |
| label_ts | TIMESTAMP | Timestamp when barrier was hit |
| avg_uniqueness | DOUBLE | Sample weight — AFML Ch. 4 average uniqueness |

### Feature Tables

All feature tables share this key contract:

| Column | Type |
|---|---|
| event_id | STRING |
| symbol | STRING |
| event_ts | TIMESTAMP |
| computed_at | TIMESTAMP |

**`silver.features_microstructure`**
`vpin` DOUBLE · `kyle_lambda` DOUBLE · `amihud_illiquidity` DOUBLE · `roll_spread` DOUBLE

**`silver.features_technical`**
`rsi_14` DOUBLE · `macd_line` DOUBLE · `macd_signal` DOUBLE · `macd_histogram` DOUBLE · `bb_width` DOUBLE · `sma_ratio` DOUBLE

**`silver.features_volatility`**
`realized_vol_20` DOUBLE · `parkinson_vol_20` DOUBLE · `garman_klass_vol_20` DOUBLE

**`silver.features_volume`**
`vwap_deviation_20` DOUBLE · `relative_volume_20` DOUBLE · `obv_momentum_20` DOUBLE

**`silver.training_datasets`**
Key + label + weights + all 14 feature columns:

| Column | Type | Notes |
|---|---|---|
| dataset_version | STRING | Immutable identifier for this training run's dataset |
| event_id | STRING | |
| symbol | STRING | |
| event_ts | TIMESTAMP | |
| label | INT | From silver.triple_barrier_labels |
| avg_uniqueness | DOUBLE | Sample weight |
| cpcv_group | INT | 1–6, chronologically assigned, stored for reproducibility |
| vpin | DOUBLE | |
| kyle_lambda | DOUBLE | |
| amihud_illiquidity | DOUBLE | Fracdiff-adjusted if ADF fails |
| roll_spread | DOUBLE | |
| rsi_14 | DOUBLE | |
| macd_line | DOUBLE | Fracdiff-adjusted if ADF fails |
| macd_signal | DOUBLE | Fracdiff-adjusted if ADF fails |
| macd_histogram | DOUBLE | Fracdiff-adjusted if ADF fails |
| bb_width | DOUBLE | |
| sma_ratio | DOUBLE | |
| realized_vol_20 | DOUBLE | |
| parkinson_vol_20 | DOUBLE | |
| garman_klass_vol_20 | DOUBLE | |
| vwap_deviation_20 | DOUBLE | |
| relative_volume_20 | DOUBLE | |
| obv_momentum_20 | DOUBLE | |
| created_at | TIMESTAMP | |

### Inference Output

**`silver.signals`**
| Column | Type | Notes |
|---|---|---|
| signal_id | STRING | Primary key |
| event_id | STRING | FK to silver.cusum_events |
| symbol | STRING | |
| event_ts | TIMESTAMP | |
| primary_prediction | INT | +1 / -1 / 0 |
| primary_probability | DOUBLE | Max class probability from primary model |
| meta_label_probability | DOUBLE | P(take) from meta-labeler |
| raw_bet_size | DOUBLE | AFML Ch.10 sizing formula output — NOT discretised |
| model_version | STRING | MLflow model version |
| pipeline_run_id | STRING | Databricks job run ID |
| created_at | TIMESTAMP | |

### Observability

**`silver.data_quality`**
`check_id` STRING · `pipeline_run_id` STRING · `check_name` STRING · `symbol` STRING · `status` STRING (`pass`/`warn`/`fail`) · `metric_value` DOUBLE · `threshold` DOUBLE · `message` STRING · `checked_at` TIMESTAMP

**`silver.promotion_evidence`**
`run_id` STRING · `symbol` STRING · `dataset_version` STRING · `psr` DOUBLE · `dsr` DOUBLE · `max_drawdown` DOUBLE · `time_under_water_bars` INT · `strategy_failure_prob` DOUBLE · `meta_precision` DOUBLE · `meta_recall` DOUBLE · `meta_f1` DOUBLE · `meta_log_loss` DOUBLE · `promoted` BOOLEAN · `failure_reasons` STRING · `evaluated_at` TIMESTAMP

**`silver.trial_registry`** (append-only — DSR denominator)
`trial_id` STRING · `run_id` STRING · `symbol` STRING · `model_type` STRING (`primary`/`meta`) · `algorithm` STRING · `hyperparams` STRING (JSON) · `dataset_version` STRING · `sharpe_mean` DOUBLE · `registered_at` TIMESTAMP

---

## 4. Feature Engineering Contracts

### PIT-Safety Rule

Every feature function queries strictly `< event_ts`. No feature may use data from the bar or tick that triggered the CUSUM event itself. Violation = lookahead bias.

### Fracdiff Rule

Applied at training time only, to features failing the ADF unit root test (p ≥ 0.05). `d` selected per feature: minimum value such that ADF p < 0.05, maximising correlation with original series. Expected range: `d ∈ [0.3, 0.5]` (AFML Ch. 5).

`d` values are stored as MLflow params at training time and serialised into the `mlflow.pyfunc.PythonModel` artifact. At inference time, `serving/model.py` loads and applies the same `d` values — no recomputation. Batch/live parity is guaranteed because both paths use the same artifact.

### Uniform Function Signature

```python
def compute_<family>(
    spark: SparkSession,
    *,
    symbol: str,
    event_ts: datetime,
    catalog: str,
) -> dict[str, float]
```

All functions are pure PySpark, have no side effects, and are testable locally via `DatabricksSession.builder.serverless()`.

### Family 1 — Microstructure (`silver.ticks_cleaned`)

| Feature | Method | Lookback | Stationarity |
|---|---|---|---|
| `vpin` | Easley et al. (2012). Bucket size = avg daily BTC volume ÷ 50. VPIN = mean(abs(buy_vol − sell_vol) / bucket_size) over last 50 buckets. `side` from `silver.ticks_cleaned.side` | 50 volume buckets ≈ 1 trading day | Bounded [0,1] — no fracdiff |
| `kyle_lambda` | OLS slope of `Δprice ~ signed_volume_imbalance`. Signed volume = +quantity if side=buy, −quantity if side=sell | Last 1,000 ticks before event_ts | Stationary by construction — no fracdiff |
| `amihud_illiquidity` | `mean(abs(log_return) / dollar_volume)` per dollar bar. `dollar_volume = close × volume` | Last 20 dollar bars before event_ts | Apply fracdiff if ADF fails |
| `roll_spread` | `2 × sqrt(max(0, −cov(Δp_t, Δp_{t−1})))` on mid-prices | Last 500 ticks before event_ts | Stationary — no fracdiff |

### Family 2 — Technical (`gold.dollar_bars`)

| Feature | Method | Window | Stationarity |
|---|---|---|---|
| `rsi_14` | Wilder RSI on `close` | 14 bars | Bounded [0,100] — no fracdiff |
| `macd_line` | EMA(12) − EMA(26) of `close` | 26 bars | Apply fracdiff if ADF fails |
| `macd_signal` | EMA(9) of `macd_line` | 9 bars on macd_line | Apply fracdiff if ADF fails |
| `macd_histogram` | `macd_line − macd_signal` | — | Apply fracdiff if ADF fails |
| `bb_width` | `(bb_upper − bb_lower) / bb_middle` where middle = SMA(20), σ = rolling std(20) of `close` | 20 bars | Stationary ratio — no fracdiff |
| `sma_ratio` | `SMA(10) / SMA(30)` of `close` | 30 bars | Stationary ratio — no fracdiff |

### Family 3 — Volatility (`gold.dollar_bars`)

Annualisation factor: `sqrt(avg_bars_per_day × 365)` — bar-denominated, consistent with CUSUM threshold computation.

| Feature | Method | Window | Stationarity |
|---|---|---|---|
| `realized_vol_20` | EWMA(λ=0.94) of squared log returns, annualised. Same λ as CUSUM threshold — vol feature and threshold share the same regime view by design | 20 bars | Stationary — no fracdiff |
| `parkinson_vol_20` | `sqrt(1/(4×20×ln2) × Σln(high/low)²)`, annualised | 20 bars | Stationary — no fracdiff |
| `garman_klass_vol_20` | Garman-Klass formula on OHLC, annualised | 20 bars | Stationary — no fracdiff |

### Family 4 — Volume (`gold.dollar_bars`)

| Feature | Method | Window | Stationarity |
|---|---|---|---|
| `vwap_deviation_20` | `close / VWAP(20) − 1` where `VWAP = sum(close × volume) / sum(volume)` | 20 bars | Stationary ratio — no fracdiff |
| `relative_volume_20` | `current_bar_volume / rolling_mean_volume(20)` | 20 bars | Stationary ratio — no fracdiff |
| `obv_momentum_20` | `(OBV_t − OBV_{t−20}) / abs(OBV_{t−20})`. 20-bar OBV return — NOT raw OBV (which is non-stationary) | 20 bars | Stationary — no fracdiff |

---

## 5. CUSUM Event Detection

Source: `gold.dollar_bars`. Implementation: `ml/jobs/task_cusum_events.py`.

| Parameter | Value | Notes |
|---|---|---|
| Input series | Log returns of `close` per symbol | `log(close_t / close_{t-1})` |
| Volatility estimator | EWMA λ=0.94 on dollar bar log returns | Same estimator used in `realized_vol_20` |
| Threshold | 1× daily vol × `sqrt(avg_bars_per_day)` | `avg_bars_per_day = total bars / calendar days in lookback`. Replaces clock-based `sqrt(390)` |
| Reset | Symmetric | S+ and S− both reset to 0 on event trigger |
| Minimum inter-event spacing | 5 dollar bars | Prevents clustered events during flash crashes |
| Target event frequency | 20–50 events/asset/month | |
| Alert condition | Outside [15, 70] events/asset/month for 2 consecutive months | Triggers threshold recalibration |

CUSUM is a **meta-hyperparameter** — not tuned via grid search. Validated through CPCV sensitivity analysis: if CPCV performance is sensitive to ±20% threshold perturbation, the model is suspect.

---

## 6. Triple-Barrier Labeling

Source: `silver.cusum_events` + `gold.dollar_bars`. Implementation: `ml/jobs/task_label_computation.py`.

| Barrier | Level | Notes |
|---|---|---|
| Upper (profit-taking) | 2× daily vol from event_ts close | Asymmetric: upper is 2×, lower is 1× |
| Lower (stop-loss) | 1× daily vol from event_ts close | |
| Vertical (time expiry) | `avg_bars_per_day × 5` bars | ≈ 5 calendar days. Recomputed monthly. |

Labeling rules:
- Upper barrier hit first → `label = +1`
- Lower barrier hit first → `label = -1`
- Vertical barrier hit first → `label = sign(return)`; `label = 0` if `abs(return) < 0.1 × daily_vol`
- Barriers use `close` prices only — no intraday high/low — to prevent within-bar lookahead
- All barrier levels computed using only data available at `event_ts` (PIT)

Label lag is inherent: an event at time T is labeled only after its vertical barrier expires (~5 days). The inference pipeline predicts on current unlabeled events; training uses historically labeled events only.

---

## 7. Model Specifications

### Primary Model

| Attribute | Value |
|---|---|
| Task | Ternary classification: +1 / -1 / 0 |
| Input features | 14 features from `silver.training_datasets`, fracdiff-adjusted |
| Target | `silver.triple_barrier_labels.label` |
| Sample weights | `avg_uniqueness` per event (AFML Ch. 4 — handles overlapping label windows) |
| Class imbalance | Via sample weights only — no oversampling, no undersampling |
| Framework | scikit-learn, single-node cluster — no Spark ML |
| Tuning | Randomised search within CPCV — never grid search on a single split |

**Candidate algorithms and hyperparameter search spaces:**

| Model | Parameters |
|---|---|
| ElasticNet | `alpha ∈ [0.0001, 1.0]` log-uniform · `l1_ratio ∈ [0.0, 1.0]` uniform |
| RandomForest | `n_estimators ∈ [100, 500]` · `max_depth ∈ [3, 15]` · `min_samples_leaf ∈ [5, 50]` |
| BaggedDecisionTrees | `n_estimators ∈ [100, 500]` · `max_samples ∈ [0.5, 1.0]` · `max_features ∈ [0.5, 1.0]` |

Output per prediction: predicted class + predicted probability per class (3 values).

### Meta-Labeler

| Attribute | Value |
|---|---|
| Task | Binary classification: 1 (take) / 0 (pass) |
| Input | Primary model OOS predictions (predicted side + probability) + all 14 original features |
| Target | 1 if primary model's predicted side resulted in profit (upper barrier hit), else 0 |
| Training dependency | Trains sequentially after primary CPCV, within the same 5-day training run. Uses OOS predictions from all 15 CPCV splits as training signal. OOS predictions held in memory — not persisted to Delta. |
| Candidates | GradientBoostingClassifier · RandomForest (scikit-learn) |
| Inference thresholds | `meta_label_probability > 0.50` AND `primary_probability > 0.55` — both conditions required to produce a signal |
| Evaluation metrics | Precision > 0.60 · Recall > 0.50 · F1 > 0.55 · NLL < 0.65 |

### Bet Sizing (AFML Ch. 10)

Applied only to signals passing both meta-label thresholds.

```
z_score    = (meta_label_probability − 0.5) / σ_probability
raw_bet_size = 2 × Φ(z_score) − 1
```

`σ_probability` estimated from the meta-labeler's OOS calibration curve at training time. Stored as MLflow artifact.

`raw_bet_size` is written to `silver.signals` as-is. Discretisation into position tiers and portfolio constraint enforcement (max 2 positions, single-asset ≤ 70%, total exposure ≤ 1.0) are the **external trading system's responsibility**.

---

## 8. Validation Framework

### Combinatorially Purged Cross-Validation (CPCV — AFML Ch. 12)

| Parameter | Value | Rationale |
|---|---|---|
| N (groups) | 6 | C(6,2) = 15 splits, φ[6,2] = 5 backtest paths |
| k (test groups per split) | 2 | Balances path diversity vs training set size |
| Purge window | `avg_bars_per_day × 5` bars | Bar-denominated. Matches vertical barrier window. Prevents leakage through overlapping triple-barrier labels. |
| Embargo | 1% of total sample size | Applied after each test group boundary |

Group formation: dataset sorted chronologically by `event_ts`, split into 6 equal groups. `cpcv_group ∈ {1,…,6}` stored in `silver.training_datasets` for reproducibility.

OOS predictions from all 15 splits are held in memory during the training run — the meta-labeler trains on them in the same run before they are discarded.

### Statistical Tests

**Probabilistic Sharpe Ratio (PSR — AFML Ch. 14):**
```
PSR(SR*=0) = Φ( SR_hat / σ_SR )

σ_SR = sqrt( (1 − γ₃·SR_hat + (γ₄−1)/4·SR_hat²) / (T−1) )
```
`γ₃` = skewness, `γ₄` = kurtosis of path returns. BTC kurtosis is typically 5–15; the `γ₄` term widens `σ_SR`, correctly penalising strategies that appear profitable only due to fat-tail sampling noise.
**Threshold: PSR > 0.95**

**Deflated Sharpe Ratio (DSR — AFML Ch. 14):**
Trial count read from `silver.trial_registry` (all independent trials across all runs for this symbol). Expected maximum Sharpe under null computed from trial count. Used as `SR*` in PSR formula.
**Threshold: DSR > 0.90**

**Strategy-Failure Probability (AFML Ch. 15):**
**Threshold: < 10%**

### Promotion Gates — ALL must pass

| Metric | Threshold |
|---|---|
| PSR | > 0.95 |
| DSR | > 0.90 |
| Max drawdown (CPCV paths) | < 25% |
| Time under water | < 90 dollar bars |
| Execution-cost robustness | Profitable at 2× estimated costs (Binance 0.1% fees + slippage) |
| Strategy-failure probability | < 10% |
| Meta-label precision | > 0.60 |
| Meta-label recall | > 0.50 |
| Meta-label F1 | > 0.55 |
| Meta-label NLL | < 0.65 |

**Promotion flow:**
```
All gates pass  → MLflow Model Registry stage = "Staging" (automatic)
Any gate fails  → model NOT promoted; failure reasons written to silver.promotion_evidence
Staging → Production → manual trigger; second CPCV pass on staging catalog data required
```

### Feature Importance (AFML Ch. 8)

All three methods run within the CPCV framework after primary model training:

| Method | Type | Rule |
|---|---|---|
| MDI (Mean Decrease Impurity) | In-sample, tree-based | Context only — biased toward high-cardinality features |
| MDA (Mean Decrease Accuracy) | OOS, permutation-based | Features with negative MDA averaged across splits → removal candidates |
| SFI (Single Feature Importance) | OOS, per-feature | Detects features only useful in combination |

**Robust feature set** = intersection of top-20 by MDI and top-20 by MDA. Recomputed every training run. Stored as MLflow artifact.

### Walk-Forward Analysis

Secondary to CPCV. Run within same training job for diagnostics only — does not feed promotion gates.

| Parameter | Value |
|---|---|
| Training window | Expanding (all history up to split point) |
| OOS window | 30 calendar days |
| Purge/embargo | Same bar-denominated rules as CPCV |

Results stored as MLflow tagged artifacts.

---

## 9. Serving Endpoint

**Implementation:** `ml/serving/model.py` — `mlflow.pyfunc.PythonModel`

**Input:** `{"symbol": "BTCUSDT", "event_ts": "<ISO8601>"}`

**Output:**
```json
{
  "prediction": 1,
  "primary_probability": 0.62,
  "meta_label_probability": 0.71,
  "raw_bet_size": 0.44,
  "model_version": "3"
}
```

**`load_context()` responsibilities:**
- Load primary model weights from MLflow artifact
- Load meta-labeler weights from MLflow artifact
- Load fracdiff `d` values per feature from MLflow artifact
- Load feature window parameters from MLflow artifact
- Initialise SparkSession with Unity Catalog access

**`predict()` responsibilities:**
1. Read `gold.dollar_bars` and `silver.ticks_cleaned` with PIT lookback strictly `< event_ts`
2. Call `ml/features/` functions (same code as batch job — zero duplication)
3. Apply fracdiff using loaded `d` values
4. Score primary model → prediction + probabilities
5. Score meta-labeler → meta probability
6. Compute `raw_bet_size` via AFML Ch. 10 formula
7. Return JSON response

**Batch/live parity guarantee:** `ml/features/` module is shared between `task_feature_engineering.py` (batch) and `serving/model.py` (live). Same code path, same fracdiff `d` values. Identical predictions for identical inputs.

---

## 10. Job Resources

### `resources/ml_training_job.yml` — Lakeflow Job

**Cron:** `0 0 */5 * *` (midnight UTC, every 5 days)
**Compute:** Single-node job cluster (scikit-learn only — no Spark ML)
**Compute budget:** $15–20/month for promotion-grade CPCV. Local/notebook for research iterations ($0).

**Task DAG:**
```
task_cusum_events
    ↓
task_label_computation          (labels matured events only — label lag is inherent)
    ↓
task_feature_engineering        (4 families dispatched in parallel)
    ↓
task_feature_assembly           (PIT-safe join → silver.training_datasets + cpcv_group assignment)
    ↓
task_train                      (CPCV primary model → sequential meta-labeler → walk-forward diagnostics)
    ↓
task_promote                    (promotion gates → silver.promotion_evidence → MLflow Registry)
```

### `resources/ml_inference_job.yml` — Lakeflow Job

**Cron:** TBD
**Compute:** Serverless

**Task DAG:**
```
task_data_quality               (bar production rate, freshness, price sanity → silver.data_quality)
    ↓
task_cusum_events               (detect new events since last run)
    ↓
task_label_computation          (label any matured events for future training)
    ↓
task_feature_engineering
    ↓
task_feature_assembly           (inference matrix — no label join)
    ↓
task_primary_inference          (load promoted model from MLflow Registry → predictions)
    ↓
task_meta_label_inference       (filter by thresholds)
    ↓
task_bet_sizing                 (AFML Ch. 10 → raw_bet_size)
    ↓
task_signal_write               (→ silver.signals)
```

Data quality failure modes:
- Freshness failure → stale-data warning appended to `silver.data_quality`, pipeline continues
- Completeness failure → downstream tasks skipped for affected symbol

### `resources/ml_serving_endpoint.yml` — Model Serving

- Source: MLflow Model Registry `Production` stage
- Auto-scales to 0 when idle
- Unity Catalog access enabled (required for `predict()` to query `gold.dollar_bars` + `silver.ticks_cleaned`)

### `resources/ml_pipeline.pipeline.yml` — DELETED

Replaced by `ml_training_job.yml` + `ml_inference_job.yml` + `ml_serving_endpoint.yml`.

---

## 11. Code Structure

```
src/trading_databricks/ml/
├── pipeline.py               ← DELETED (was SDP stub)
├── features.py               ← DELETED (replaced by features/ sub-package)
│
├── features/                 Pure PySpark functions. Fully testable with pytest.
│   ├── __init__.py           Re-exports all compute_* functions
│   ├── microstructure.py     compute_microstructure() → {vpin, kyle_lambda, amihud_illiquidity, roll_spread}
│   ├── technical.py          compute_technical() → {rsi_14, macd_line, macd_signal, macd_histogram, bb_width, sma_ratio}
│   ├── volatility.py         compute_volatility() → {realized_vol_20, parkinson_vol_20, garman_klass_vol_20}
│   └── volume.py             compute_volume() → {vwap_deviation_20, relative_volume_20, obv_momentum_20}
│
├── jobs/                     Lakeflow Job task entry points. One script per task.
│   ├── task_cusum_events.py
│   ├── task_label_computation.py
│   ├── task_feature_engineering.py   Dispatches 4 family sub-tasks
│   ├── task_feature_assembly.py
│   ├── task_data_quality.py
│   ├── task_primary_inference.py
│   ├── task_meta_label_inference.py
│   ├── task_bet_sizing.py
│   ├── task_signal_write.py
│   ├── train.py                      CPCV: primary model → sequential meta-labeler → walk-forward
│   └── promote.py                    Gates → silver.promotion_evidence → MLflow Registry
│
└── serving/
    ├── __init__.py
    └── model.py              mlflow.pyfunc.PythonModel: {symbol, event_ts} → signal

tests/ml/
├── features/
│   ├── test_microstructure.py    Synthetic tick DataFrame → assert VPIN ∈ [0,1], kyle_lambda > 0
│   ├── test_technical.py         Synthetic bar DataFrame → assert RSI ∈ [0,100], MACD shape
│   ├── test_volatility.py        Synthetic bar DataFrame → assert vol > 0, annualisation correct
│   └── test_volume.py            Synthetic bar DataFrame → assert stationarity of ratios
└── serving/
    └── test_model.py             Mock SparkSession → assert predict() output schema
```

---

## 12. Verification Checklist

| Step | Command / Method |
|---|---|
| Unit tests | `uv run pytest tests/ml/ -x -v` |
| Bundle validation | `databricks bundle validate` |
| Research CPCV (non-promotion-grade) | Run `ml/jobs/train.py` locally with dataset subset; verify MLflow run logged with all expected params |
| Promotion-grade CPCV | `databricks bundle run ml_training_job -t dev`; verify `silver.promotion_evidence` written with `promoted` column |
| Feature parity (batch vs live) | Run `task_feature_engineering.py` + call serving endpoint for same `(symbol, event_ts)`; assert all 14 feature values match |
| Serving endpoint response | `mcp__databricks__query_serving_endpoint` with test entity key; assert response contains `prediction`, `primary_probability`, `meta_label_probability`, `raw_bet_size`, `model_version` |
| Silver table schemas | `mcp__databricks__execute_sql "DESCRIBE TABLE silver.<table>"` for each of the 12 new silver tables |
| CUSUM event frequency | Query `silver.cusum_event_stats`; assert event count per month ∈ [15, 70] |
