# ML Pipeline Phase 1A — End-to-End Specification

**Date:** 2026-04-07
**Status:** Draft
**Supersedes:** `2026-04-04-ml-model-layer-specs.md` (for Phase 1A scope only)
**Cross-references:**
- `docs/advanced-ml-modelling/trading-logic.md` (CUSUM, features, models, bet sizing)
- `docs/advanced-ml-modelling/validation.md` (CPCV, PSR, DSR, promotion gates)
- `docs/advanced-ml-modelling/trading-pipeline.md` (inference pipeline, job DAG, schedules)
- `docs/superpowers/specs/2026-04-04-ml-model-layer-specs.md` (original ML spec — feature contracts, schemas, validation math)
- `docs/superpowers/specs/2026-04-04-mas-final-spec.md` (MAS pipeline producing `gold.news_causal_features`)
- `docs/coordination/TRD-6-gen-ai-agent-layer.md` (MAS coordination)

---

## 1. Overview & Scope

### What This Pipeline Does

A batch ML pipeline that generates **trading signals** for BTCUSDT using information-driven dollar bars, CUSUM event detection, triple-barrier labeling, and a primary + meta-label model pair. Validated via CPCV with 10 promotion gates.

### What This Pipeline Does NOT Do

- No live trading execution. No Binance API calls. No order management.
- No Model Serving endpoint (deferred to Phase 1B).
- Terminal output is `silver.signals` + `silver.positions` + email notification.
- Discretization into position tiers and manual execution are human responsibilities.

### Phase 1A Boundary

| In scope | Deferred |
|----------|----------|
| BTC only (BTCUSDT) | ETH (Phase 1B) |
| Dollar bars (primary bar series) | Dollar imbalance bars (Phase 1B) |
| CUSUM event detection | Fractional differentiation |
| Triple-barrier labeling | Microstructure features (VPIN, Kyle's lambda) |
| 3 feature families: technical, volatility, volume | Crypto-specific features (funding rate, OI) |
| News/LLM features (optional, activates when MAS completes) | Serving endpoint (Phase 1B) |
| Primary model (RandomForest baseline) | Walk-forward analysis |
| Meta-labeling (GradientBoosting) | Slack/webhook notifications |
| Bet sizing + signal write | Short positions (Phase 2) |
| CPCV validation + promotion gates | Staging → Prod second-pass |
| MLflow tracking + UC Model Registry | Execution logging |
| Email notification | API execution via Binance SDK |
| Historical backfill from Binance Vision | — |

### Key Design Decisions

1. **Signal generation only** — no execution, no order management.
2. **Dollar bars** as primary time series (Lopez de Prado AFML Ch. 2).
3. **BTC-only** — single asset simplifies pipeline, ETH added in Phase 1B with zero schema migration.
4. **Backfill first** — 6 months of historical data from Binance Vision bootstraps training.
5. **News features are optional** — pipeline works without them; they plug in when MAS (TRD-6) completes.
6. **No microstructure features** — VPIN/Kyle's lambda require tick-level compute; deferred to Phase 1B.
7. **UC Model Registry with aliases** — Champion/Challenger pattern, not deprecated stages.
8. **Parallel agent development** — schema contracts first, then independent agent sessions.

---

## 2. Architecture

```
LAYER 1 — DATA FOUNDATION (SDP streaming + backfill)
═════════════════════════════════════════════════════
bronze.raw_aggtrades_backfill (6mo BTCUSDT from Binance Vision)
        │
        ▼  [silver transform: fix timestamps, cast types, dedup, dollar_value]
silver.ticks_cleaned
        │
        ▼  [gold transform: dollar bar construction, stateful accumulation]
gold.dollar_bars  (OHLCV bars, est. 8K–40K bars over 6 months)


LAYER 2 — EVENT DETECTION + LABELING (Lakeflow Job, backfill then 4h cadence)
══════════════════════════════════════════════════════════════════════════════
gold.dollar_bars
        │
        ├──▶ silver.cusum_events          (CUSUM filter, est. 120–300 events)
        │           │
        │           ▼
        │    silver.triple_barrier_labels  (matured events only)
        │
        ▼
silver.features_technical    ─┐
silver.features_volatility   ─┼──▶ silver.training_datasets
silver.features_volume       ─┤    (PIT-safe join per event)
silver.features_news         ─┘    (news optional at launch)


LAYER 3 — MODEL TRAINING + INFERENCE (Lakeflow Job)
════════════════════════════════════════════════════
silver.training_datasets
        │
        ├──▶ [TRAIN] CPCV → promotion gates → MLflow UC Registry
        │
        └──▶ [INFER] primary → meta-label → bet sizing
                    │
                    ▼
            silver.signals → silver.positions
                    │
                    ▼
            Email notification (manual approval + execution)
```

---

## 3. Layer 1 — Data Foundation

### 3.1 Data Backfill from Binance Vision

**Source:** `https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT/`

**URL pattern:**
```
https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{YYYY}-{MM}.zip
```

**Target months:** 2024-10 through 2025-03 (6 months)

**Binance aggTrade CSV schema (no header row):**

| Column Index | Name | Type | Description |
|---|---|---|---|
| 0 | agg_trade_id | LONG | Binance aggregate trade ID |
| 1 | price | STRING | Trade price |
| 2 | quantity | STRING | Trade quantity |
| 3 | first_trade_id | LONG | First trade ID in aggregation |
| 4 | last_trade_id | LONG | Last trade ID in aggregation |
| 5 | timestamp | LONG | Unix milliseconds |
| 6 | is_buyer_maker | BOOLEAN | True = seller initiated (sell), False = buyer initiated (buy) |
| 7 | is_best_match | BOOLEAN | Best price match flag |

**Ingestion workflow (UC Volume approach):**

1. Download 6 ZIPs via `requests` in a Databricks notebook
2. Upload raw ZIPs to UC Volume: `Volumes/trading-signal-gen-dev/bronze/binance_raw/`
3. Unzip and read CSVs with Spark, apply transform:
   - `timestamp`: Unix millis → ISO-8601 TIMESTAMP via `F.from_unixtime(F.col("timestamp") / 1000)`
   - `side`: `is_buyer_maker` → "buy" (False) / "sell" (True)
   - `trade_id`: from `agg_trade_id`
   - Add: `symbol = "BTCUSDT"`, `exchange = "binance"`, `batch_id` from filename
4. Write to `bronze.raw_aggtrades_backfill` as Delta
5. Verify: checksums, row counts, timestamp range, price sanity

**Note:** This replaces `bronze.raw_aggtrades_samples` (which has a millis-as-seconds timestamp bug, 77 days, BTC only). The backfill data has correct timestamps and 6 months of coverage.

### 3.2 Silver Transform — ticks_cleaned

**Input:** `bronze.raw_aggtrades_backfill`
**Output:** `silver.ticks_cleaned`

| Column | Type | Transform |
|---|---|---|
| trade_id | LONG | CAST from STRING |
| symbol | STRING | Pass through |
| timestamp | TIMESTAMP | Already fixed in backfill ingest |
| price | DOUBLE | CAST from STRING |
| quantity | DOUBLE | CAST from STRING |
| side | STRING | "buy" / "sell" |
| dollar_value | DOUBLE | `price * quantity` (computed) |
| exchange | STRING | Pass through |
| batch_id | STRING | Pass through |
| _ingested_at | TIMESTAMP | `current_timestamp()` |

**Dedup:** By `trade_id` (keep first occurrence).
**Quality gate:** `@dp.expect_or_drop("valid_price", "price > 0 AND price < 1000000")`

### 3.3 Dollar Bar Construction

**Input:** `silver.ticks_cleaned` ordered by `timestamp`
**Output:** `gold.dollar_bars`

**Algorithm:** Accumulate `dollar_value` until threshold reached → emit bar.

**Threshold calibration:**
- Compute total dollar volume for first 14 days of data
- Divide by 14 to get average daily dollar volume
- Target: ~50-100 bars/day → `threshold = daily_dollar_volume / 75`
- Threshold stored in `common/config.py` as `DollarBarConfig.threshold`

**Output schema (from `ticks/schemas.py`):**

| Column | Type | Description |
|---|---|---|
| bar_index | LONG | Monotonically increasing bar sequence |
| symbol | STRING | "BTCUSDT" |
| bar_start_time | TIMESTAMP | First tick in bar |
| bar_end_time | TIMESTAMP | Last tick in bar |
| open | DOUBLE | First tick price |
| high | DOUBLE | Max tick price |
| low | DOUBLE | Min tick price |
| close | DOUBLE | Last tick price |
| volume | DOUBLE | Sum of quantity |
| dollar_volume | DOUBLE | Sum of dollar_value |
| tick_count | LONG | Number of ticks in bar |

**Implementation:** Stateful loop on pandas DataFrame (`.toPandas()` from ticks_cleaned). SDP materialized view (full recompute, per Lopez de Prado).

---

## 4. Layer 2 — Event Detection & Feature Engineering

### 4.1 CUSUM Event Detection

**Input:** `gold.dollar_bars`
**Output:** `silver.cusum_events`, `silver.cusum_event_stats`

| Parameter | Value | Notes |
|---|---|---|
| Input series | Log returns of `close` per symbol | `log(close_t / close_{t-1})` |
| Volatility estimator | EWMA λ=0.94 on dollar bar log returns | |
| Threshold | 1× daily vol × `sqrt(avg_bars_per_day)` | `avg_bars_per_day = total bars / calendar days in 30-day lookback` |
| Reset | Symmetric | S+ and S− both reset to 0 on event trigger |
| Minimum inter-event spacing | 5 dollar bars | Prevents clustered events during flash crashes |
| Target event frequency | 20–50 events/month | Alert if outside [15, 70] for 2 consecutive months |

**`silver.cusum_events` schema:**

| Column | Type | Description |
|---|---|---|
| event_id | STRING | `{symbol}_{bar_index}` |
| symbol | STRING | Partition key |
| event_ts | TIMESTAMP | Dollar bar timestamp at trigger |
| bar_index | LONG | FK to gold.dollar_bars |
| log_return | DOUBLE | Log return at event bar |
| cusum_side | STRING | "up" / "down" — which CUSUM (S+ or S-) triggered |
| cusum_value | DOUBLE | S+/S- value at trigger |
| threshold_used | DOUBLE | Threshold value at trigger time |
| ewma_vol | DOUBLE | EWMA vol at trigger |
| avg_bars_per_day | DOUBLE | Empirical bars/day |
| _ingested_at | TIMESTAMP | Pipeline write time |

**`silver.cusum_event_stats` schema:**

| Column | Type | Description |
|---|---|---|
| symbol | STRING | |
| date | DATE | Calendar date |
| event_count | INT | Events triggered this date |
| threshold_used | DOUBLE | |
| ewma_vol | DOUBLE | |
| avg_bars_per_day | DOUBLE | |

**Implementation:** Pure Python function on pandas DataFrame. Stateful loop — not vectorizable.

### 4.2 Triple-Barrier Labeling

**Input:** `silver.cusum_events` + `gold.dollar_bars`
**Output:** `silver.triple_barrier_labels`

| Barrier | Level | Notes |
|---|---|---|
| Upper (profit-taking) | `close_event × (1 + 2σ_daily)` | Asymmetric: upper 2×, lower 1× |
| Lower (stop-loss) | `close_event × (1 - 1σ_daily)` | |
| Vertical (time expiry) | `bar_index_event + N` where `N = avg_bars_per_day × 5` | ~5 calendar days, recomputed monthly |

**Labeling rules:**
- Upper hit first → `label = +1`
- Lower hit first → `label = -1`
- Vertical hit → `label = sign(return)`; `label = 0` if `|return| < 0.1 × σ_daily`
- Barriers use `close` prices only — no intrabar high/low lookahead
- All barrier levels from PIT-available data at `event_ts`

**Label lag:** Events are labeled only when vertical barrier expires (~5 days of bars). Training uses matured events; inference predicts on current unlabeled events.

**`silver.triple_barrier_labels` schema:**

| Column | Type | Description |
|---|---|---|
| event_id | STRING | FK to cusum_events |
| symbol | STRING | |
| event_ts | TIMESTAMP | |
| label | INT | +1 / -1 / 0 |
| barrier_hit | STRING | "upper" / "lower" / "vertical" |
| return_pct | DOUBLE | Realized return from event to barrier hit |
| upper_level | DOUBLE | Absolute price level |
| lower_level | DOUBLE | Absolute price level |
| vertical_bar | LONG | Bar index of vertical barrier |
| matured_at | TIMESTAMP | When barrier was hit |
| avg_uniqueness | DOUBLE | Sample weight (AFML Ch. 4) |

### 4.3 Feature Engineering

All features computed **strictly before event_ts** (PIT-safe). Each family is a pure `DataFrame → DataFrame` transform.

#### Family 1 — Technical (from `gold.dollar_bars`)

| Feature | Window | Description | Stationarity |
|---|---|---|---|
| `rsi_14` | 14 bars | Relative Strength Index | Bounded [0,100] |
| `macd_signal` | 12/26/9 bars | MACD signal line crossover value | May need fracdiff |
| `bb_pct` | 20 bars, 2σ | Bollinger Band %B position | Bounded [0,1+] |
| `sma_ratio_5_20` | 5/20 bars | Short/long SMA ratio | Stationary ratio |
| `sma_ratio_10_50` | 10/50 bars | Medium/long SMA ratio | Stationary ratio |
| `log_return_5` | 5 bars | Cumulative log return | Stationary |
| `log_return_20` | 20 bars | Cumulative log return | Stationary |

**Schema: `silver.features_technical`**
Key columns (`event_id`, `symbol`, `event_ts`, `computed_at`) + 7 feature columns.

#### Family 2 — Volatility (from `gold.dollar_bars`)

| Feature | Window | Description | Stationarity |
|---|---|---|---|
| `realized_vol_20` | 20 bars | Std of log returns, annualized | Stationary |
| `realized_vol_50` | 50 bars | Std of log returns, annualized | Stationary |
| `parkinson_vol_20` | 20 bars | High-low range estimator | Stationary |
| `garman_klass_20` | 20 bars | OHLC volatility estimator | Stationary |
| `vol_ratio_20_50` | 20/50 bars | Short/long vol ratio (regime indicator) | Stationary ratio |

Annualization factor: `sqrt(avg_bars_per_day × 365)`.

**Schema: `silver.features_volatility`**
Key columns + 5 feature columns.

#### Family 3 — Volume (from `gold.dollar_bars`)

| Feature | Window | Description | Stationarity |
|---|---|---|---|
| `vwap_dev_20` | 20 bars | `(close - VWAP) / VWAP` | Stationary ratio |
| `relative_volume_20` | 20 bars | `volume / SMA(volume, 20)` | Stationary ratio |
| `obv_slope_20` | 20 bars | OBV linear regression slope | Stationary |
| `dollar_volume_ratio` | 5/20 bars | Short/long dollar volume ratio | Stationary ratio |

**Schema: `silver.features_volume`**
Key columns + 4 feature columns.

#### Family 4 — News/LLM (from `gold.news_causal_features`, optional at launch)

**Activation condition:** MAS pipeline (TRD-6) completes through TRD-34 (evaluation passes). Until then, this family returns all zeros.

**Join logic:** For each CUSUM event at time `t`, query `gold.news_causal_features` in `[t - 24h, t)` where `is_impactful = True` and `validation_status = 'accepted'`.

**Per L1 treatment category** (8 categories: Sentiment, DeFi, Supply, Exchange, Macro, Adoption, Security, Regulation):

| Feature | Description |
|---|---|
| `news_count_{l1}` | Article count in window |
| `news_max_conf_{l1}` | Max confidence score |
| `news_direction_{l1}` | Mode of expected_direction (bullish=+1, bearish=-1, volatile=0) |
| `news_accepted_ratio_{l1}` | Fraction with validation_status='accepted' |

**Global aggregates:**

| Feature | Description |
|---|---|
| `news_volume_24h` | Total article count in 24h window |
| `news_sentiment_score` | Confidence-weighted direction sum |
| `has_news_24h` | Boolean: any articles in window |

**Total: 8 × 4 + 3 = 35 news features.** Fixed-width, PIT-safe, zero-padded when no news.

**Schema: `silver.features_news`**
Key columns + 35 feature columns.

#### Feature Assembly

**Output:** `silver.training_datasets`

PIT-safe join of all feature families on `(event_id, symbol)`. Label and sample weight joined from `silver.triple_barrier_labels`.

| Column | Type | Description |
|---|---|---|
| dataset_version | STRING | Immutable identifier (ISO timestamp of assembly) |
| event_id | STRING | |
| symbol | STRING | |
| event_ts | TIMESTAMP | |
| label | INT | From triple_barrier_labels (NULL for unlabeled events) |
| avg_uniqueness | DOUBLE | Sample weight |
| cpcv_group | INT | 1–6, chronologically assigned |
| (7 technical features) | DOUBLE | |
| (5 volatility features) | DOUBLE | |
| (4 volume features) | DOUBLE | |
| (35 news features) | DOUBLE | All zeros until MAS activates |
| created_at | TIMESTAMP | |

**Total features at launch: ~16** (technical + volatility + volume).
**Total features with news: ~51.**

---

## 5. Layer 3 — Model Training, Registry & Inference

### 5.1 MLflow Configuration

```python
import mlflow
mlflow.set_registry_uri("databricks-uc")
```

**UC Model Registry paths:**

| Component | Path | Aliases |
|---|---|---|
| Primary model | `trading-signal-gen-{env}.gold.btc_primary_model` | `Champion`, `Challenger` |
| Meta-label model | `trading-signal-gen-{env}.gold.btc_meta_label_model` | `Champion`, `Challenger` |

**Experiment names:**
- `/trading-databricks/btc-primary-training`
- `/trading-databricks/btc-meta-label-training`
- `/trading-databricks/btc-cpcv-validation`
- `/trading-databricks/btc-promotion-gates`

### 5.2 Primary Model

| Attribute | Value |
|---|---|
| Task | Ternary classification: +1 / -1 / 0 |
| Input | 16 features (expanding to 51 with news) |
| Target | `silver.triple_barrier_labels.label` |
| Sample weights | `avg_uniqueness` per event (AFML Ch. 4) |
| Class imbalance | Via sample weights only |
| Framework | scikit-learn, single-node — no Spark ML |
| Baseline algorithm | RandomForestClassifier |

**Hyperparameter search space (RandomizedSearchCV within CPCV):**

| Parameter | Range |
|---|---|
| `n_estimators` | [100, 500] |
| `max_depth` | [3, 15] |
| `min_samples_leaf` | [5, 50] |
| `max_features` | ["sqrt", "log2", 0.5, 0.8] |

**MLflow logging:**

```python
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="primary-rf-v1") as run:
    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Log CPCV path-level metrics
    for i, path_metrics in enumerate(cpcv_paths):
        mlflow.log_metrics({
            f"path_{i}_sharpe": path_metrics["sharpe"],
            f"path_{i}_max_dd": path_metrics["max_drawdown"],
        })

    # Register in UC with signature
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.iloc[[0]],
        registered_model_name=f"trading-signal-gen-{env}.gold.btc_primary_model",
    )
```

### 5.3 Meta-Label Model

| Attribute | Value |
|---|---|
| Task | Binary classification: 1 (take) / 0 (pass) |
| Input | Primary OOS predictions (side + probability) + original 16 features |
| Target | 1 if primary's side → profit (upper barrier), else 0 |
| Algorithm | GradientBoostingClassifier |
| Training | Sequential after primary CPCV, using OOS predictions from all 15 splits |

**Inference thresholds (both required):**

| Condition | Threshold |
|---|---|
| Meta-label probability | > 0.50 |
| Primary model confidence | > 0.55 |

### 5.4 Bet Sizing (AFML Ch. 10)

Applied only to signals passing both meta-label thresholds.

```
z_score      = (meta_label_probability - 0.5) / σ_probability
raw_bet_size = 2 × Φ(z_score) - 1
```

`σ_probability` estimated from meta-labeler's OOS calibration at training time. Stored as MLflow artifact.

**Discretization tiers:**

| Raw Range | Discretized Weight |
|---|---|
| [0.00, 0.10) | 0.00 (no position) |
| [0.10, 0.25) | 0.15 (small) |
| [0.25, 0.50) | 0.35 (medium) |
| [0.50, 0.75) | 0.60 (large) |
| [0.75, 1.00] | 0.85 (conviction) |

**Constraints:** Max 1 concurrent position (BTC-only Phase 1A). Single-asset max 85%.

### 5.5 Promotion Workflow

**Promotion aliases (not deprecated stages):**

```python
from mlflow import MlflowClient
client = MlflowClient()

if all_gates_pass:
    client.set_registered_model_alias(
        f"trading-signal-gen-{env}.gold.btc_primary_model",
        "Champion",
        new_version,
    )
    client.set_registered_model_alias(
        f"trading-signal-gen-{env}.gold.btc_meta_label_model",
        "Champion",
        meta_version,
    )
```

**Loading for inference:**

```python
primary = mlflow.pyfunc.load_model(
    f"models:/trading-signal-gen-{env}.gold.btc_primary_model@Champion"
)
meta = mlflow.pyfunc.load_model(
    f"models:/trading-signal-gen-{env}.gold.btc_meta_label_model@Champion"
)
```

### 5.6 CPCV Validation

| Parameter | Value |
|---|---|
| N (groups) | 6 |
| k (test groups per split) | 2 |
| Unique splits | C(6,2) = 15 |
| Backtest paths | φ[6,2] = 5 |
| Purge window | `avg_bars_per_day × 5` bars |
| Embargo | 1% of sample size |

### 5.7 Promotion Gates — ALL Must Pass

| Metric | Threshold |
|---|---|
| PSR | > 0.95 |
| DSR | > 0.90 |
| Max drawdown (CPCV paths) | < 25% |
| Time under water | < 90 bars |
| Execution-cost robustness | Profitable at 2× estimated costs (Binance 0.1% + slippage) |
| Strategy-failure probability | < 10% |
| Meta-label precision | > 0.60 |
| Meta-label recall | > 0.50 |
| Meta-label F1 | > 0.55 |
| Meta-label NLL | < 0.65 |

**Result:** `silver.promotion_evidence` (append-only, one row per evaluation).

---

## 6. Output Tables

### `silver.signals` (append-only)

| Column | Type | Description |
|---|---|---|
| signal_id | STRING | UUID primary key |
| symbol | STRING | "BTCUSDT" |
| event_id | STRING | FK to cusum_events |
| event_ts | TIMESTAMP | Event timestamp |
| primary_side | INT | +1 (long) / -1 |
| primary_confidence | DOUBLE | Max class probability |
| meta_label | INT | 1=take, 0=pass |
| meta_confidence | DOUBLE | P(take) |
| raw_bet_size | DOUBLE | AFML Ch. 10 formula |
| discretized_weight | DOUBLE | 0.00/0.15/0.35/0.60/0.85 |
| primary_model_version | INT | MLflow version |
| meta_model_version | INT | MLflow version |
| created_at | TIMESTAMP | Pipeline run time |

### `silver.positions` (current state, overwrite)

| Column | Type | Description |
|---|---|---|
| symbol | STRING | "BTCUSDT" |
| side | STRING | "long" / "flat" |
| weight | DOUBLE | Discretized position weight |
| confidence | DOUBLE | Meta-label confidence |
| signal_id | STRING | FK to signals |
| updated_at | TIMESTAMP | |

### `silver.promotion_evidence` (append-only)

| Column | Type | Description |
|---|---|---|
| run_id | STRING | MLflow run ID |
| model_type | STRING | "primary" / "meta_label" |
| model_version | INT | Version evaluated |
| dataset_version | STRING | Training dataset version |
| psr | DOUBLE | |
| dsr | DOUBLE | |
| max_drawdown | DOUBLE | |
| time_under_water | INT | Bars |
| cost_robust | BOOLEAN | Profitable at 2x costs |
| strategy_failure_prob | DOUBLE | |
| meta_precision | DOUBLE | |
| meta_recall | DOUBLE | |
| meta_f1 | DOUBLE | |
| meta_logloss | DOUBLE | |
| all_gates_pass | BOOLEAN | Promotion verdict |
| failure_reasons | STRING | NULL if all pass, else comma-separated |
| evaluated_at | TIMESTAMP | |

### `silver.trial_registry` (append-only, DSR denominator)

| Column | Type | Description |
|---|---|---|
| trial_id | STRING | UUID |
| run_id | STRING | MLflow run ID |
| model_type | STRING | "primary" / "meta_label" |
| algorithm | STRING | e.g., "RandomForestClassifier" |
| sharpe | DOUBLE | Observed Sharpe |
| created_at | TIMESTAMP | |

### `silver.data_quality`

| Column | Type | Description |
|---|---|---|
| check_id | STRING | UUID |
| check_name | STRING | e.g., "bar_production_rate" |
| symbol | STRING | |
| status | STRING | "pass" / "warn" / "fail" |
| metric_value | DOUBLE | |
| threshold | DOUBLE | |
| message | STRING | |
| checked_at | TIMESTAMP | |

---

## 7. Complete Table Catalog

```
trading-signal-gen-{env}
├── bronze
│   ├── raw_aggtrades_samples        ← existing (52.5M, timestamp bug, 77 days)
│   ├── raw_aggtrades_backfill       ← NEW (6mo BTCUSDT from Binance Vision)
│   └── news_agent_eval_dataset      ← existing (500 articles)
│
├── silver
│   ├── ticks_cleaned                ← NEW
│   ├── cusum_events                 ← NEW
│   ├── cusum_event_stats            ← NEW
│   ├── triple_barrier_labels        ← NEW
│   ├── features_technical           ← NEW (7 features)
│   ├── features_volatility          ← NEW (5 features)
│   ├── features_volume              ← NEW (4 features)
│   ├── features_news                ← NEW (35 features, optional)
│   ├── training_datasets            ← NEW (assembled matrix)
│   ├── signals                      ← NEW (predictions)
│   ├── positions                    ← NEW (portfolio state)
│   ├── promotion_evidence           ← NEW (gate results)
│   ├── trial_registry               ← NEW (DSR tracking)
│   └── data_quality                 ← NEW (quality checks)
│
└── gold
    ├── dollar_bars                  ← NEW (primary bar series)
    ├── treatment_taxonomy           ← existing (34 rows)
    ├── mechanism_whitelist          ← existing (40 rows)
    └── news_causal_features         ← future (MAS TRD-6 output)
```

---

## 8. Lakeflow Job Definitions

### Training Job (`resources/ml_training_job.yml`)

**Cron:** `0 0 */5 * *` (midnight UTC, every 5 days)
**Compute:** Single-node job cluster (scikit-learn only)

```
task_cusum_events
    ↓
task_label_computation          (matured events only)
    ↓
task_feature_engineering        (3 families in parallel)
    ↓
task_feature_assembly           (PIT-safe join + cpcv_group assignment)
    ↓
task_train                      (CPCV primary → meta-label)
    ↓
task_promote                    (10 gates → promotion_evidence → alias)
```

### Inference Job (`resources/ml_inference_job.yml`)

**Cron:** `0 */4 * * *` (every 4 hours UTC)
**Compute:** Serverless

```
task_data_quality               (bar rate, freshness, price sanity)
    ↓
task_cusum_events               (new events since last run)
    ↓
task_feature_engineering        (3 families)
    ↓
task_feature_assembly           (inference matrix, no label join)
    ↓
task_primary_inference          (load @Champion, predict)
    ↓
task_meta_label_inference       (filter by thresholds)
    ↓
task_bet_sizing                 (raw → discretize)
    ↓
task_signal_write               (→ silver.signals, silver.positions)
    ↓
task_recommendation_notify      (email summary)
```

---

## 9. Code Structure

```
src/trading_databricks/ml/
├── __init__.py
├── config.py                    # Frozen dataclasses: CUSUMConfig, LabelConfig,
│                                #   FeatureConfig, CPCVConfig, BetSizingConfig,
│                                #   TrainingConfig, InferenceConfig, MLPipelineConfig
├── schemas.py                   # All StructType definitions for ML tables
│
├── cusum.py                     # CUSUM event detection (pure function)
├── labeling.py                  # Triple-barrier labeling (pure function)
│
├── features/
│   ├── __init__.py              # Re-exports all compute_* functions
│   ├── technical.py             # compute_technical() → 7 features
│   ├── volatility.py            # compute_volatility() → 5 features
│   ├── volume.py                # compute_volume() → 4 features
│   ├── news.py                  # compute_news() → 35 features (optional)
│   └── assembly.py              # PIT-safe join of all families
│
├── training/
│   ├── __init__.py
│   ├── cpcv.py                  # CPCV splitter (N=6, k=2, purge, embargo)
│   ├── sample_weights.py        # Average uniqueness (AFML Ch. 4)
│   ├── primary.py               # RF training + MLflow logging
│   ├── meta_label.py            # GBM training + MLflow logging
│   └── promotion.py             # 10-gate evaluation + alias promotion
│
├── inference/
│   ├── __init__.py
│   ├── predict.py               # Load @Champion, run predictions
│   ├── bet_sizing.py            # Raw bet → discretize → position weights
│   └── position.py              # Portfolio state computation
│
├── backfill/
│   ├── __init__.py
│   └── run_backfill.py          # One-time historical processing
│
└── jobs/
    ├── train.py                 # Lakeflow Job entry: train + promote
    └── infer.py                 # Lakeflow Job entry: 4h inference

tests/ml/
├── test_cusum.py
├── test_labeling.py
├── test_features_technical.py
├── test_features_volatility.py
├── test_features_volume.py
├── test_features_news.py
├── test_features_assembly.py
├── test_cpcv.py
├── test_sample_weights.py
├── test_bet_sizing.py
├── test_position.py
└── test_promotion.py
```

---

## 10. Parallel Agent Development — Task Decomposition

### Design Principles

1. **Schema-first:** Wave 1 defines every StructType and config dataclass. All agents code against contracts, not each other.
2. **Pure functions + synthetic tests:** Every Wave 2 agent writes pure `DataFrame → DataFrame` transforms testable with synthetic data. No agent needs actual Databricks tables to develop.
3. **Git worktree isolation:** Each agent gets its own worktree. Zero file overlap = clean merges.
4. **Coordination doc:** `docs/coordination/TRD-XX-ml-pipeline.md` — each agent owns one section.

### Wave Structure

```
WAVE 1 — Schema Contracts (single agent, blocks everything)
════════════════════════════════════════════════════════════
  W1: ML schemas (all StructTypes) + frozen config dataclasses
      Files: ml/__init__.py, ml/config.py, ml/schemas.py
      Merges to main → all agents branch from here

WAVE 2 — Parallel Development (5 agents + 1 DE teammate)
═════════════════════════════════════════════════════════

  DE stream (sequential):          Agent stream (ALL parallel):
  ┌──────────────────────┐         ┌──────────────────────────────┐
  │ DE1: Backfill data   │         │ Agent A: CUSUM detection     │
  │   ↓                  │         │ Agent B: Triple-barrier      │
  │ DE2: ticks_cleaned   │    ←→   │ Agent C: Features (3 + news) │
  │   ↓                  │         │ Agent D: CPCV + weights      │
  │ DE3: Dollar bars     │         │ Agent E: Bet sizing + pos    │
  └──────────────────────┘         └──────────────────────────────┘

WAVE 3 — Training Integration (2 agents parallel, needs Wave 2)
═══════════════════════════════════════════════════════════════
  Agent F: Primary + meta-label training + MLflow + promotion
  Agent G: Inference pipeline + Lakeflow Job DAG + notification

WAVE 4 — End-to-End (single agent, needs Wave 3)
═════════════════════════════════════════════════
  Agent H: Backfill orchestration + end-to-end validation
```

### File Ownership Matrix

| Agent | Owns (exclusive write) | Reads (immutable after W1) |
|---|---|---|
| **W1** | `ml/__init__.py`, `ml/config.py`, `ml/schemas.py` | — |
| **A** | `ml/cusum.py`, `tests/ml/test_cusum.py` | `ml/config.py`, `ml/schemas.py` |
| **B** | `ml/labeling.py`, `tests/ml/test_labeling.py` | `ml/config.py`, `ml/schemas.py` |
| **C** | `ml/features/**`, `tests/ml/test_features_*.py` | `ml/config.py`, `ml/schemas.py` |
| **D** | `ml/training/cpcv.py`, `ml/training/sample_weights.py`, `ml/training/promotion.py`, tests | `ml/config.py`, `ml/schemas.py` |
| **E** | `ml/inference/bet_sizing.py`, `ml/inference/position.py`, tests | `ml/config.py`, `ml/schemas.py` |
| **F** | `ml/training/primary.py`, `ml/training/meta_label.py`, tests | All W2 exports |
| **G** | `ml/jobs/infer.py`, `ml/inference/predict.py`, `ml/jobs/train.py`, tests | All W2 exports |
| **DE1** | `scripts/backfill_download.py` | — |
| **DE2** | `ticks/silver.py` (silver transform) | `ticks/schemas.py` |
| **DE3** | `ticks/gold/dollar_bars.py` | `ticks/schemas.py`, `common/config.py` |
| **H** | `ml/backfill/run_backfill.py`, `resources/ml_*.yml` | All above |

### Interface Contracts (defined in Wave 1)

**Schemas (`ml/schemas.py`):**
```
CUSUM_EVENTS_SCHEMA           # Agent A produces, Agent B consumes
CUSUM_EVENT_STATS_SCHEMA      # Agent A produces
TRIPLE_BARRIER_LABELS_SCHEMA  # Agent B produces, Agent C consumes
FEATURES_TECHNICAL_SCHEMA     # Agent C produces
FEATURES_VOLATILITY_SCHEMA    # Agent C produces
FEATURES_VOLUME_SCHEMA        # Agent C produces
FEATURES_NEWS_SCHEMA          # Agent C produces
TRAINING_DATASET_SCHEMA       # Agent C produces (assembly), Agent D/F consumes
SIGNALS_SCHEMA                # Agent E produces
POSITIONS_SCHEMA              # Agent E produces
PROMOTION_EVIDENCE_SCHEMA     # Agent D produces
TRIAL_REGISTRY_SCHEMA         # Agent D produces
DATA_QUALITY_SCHEMA           # Agent G produces
```

**Configs (`ml/config.py`):**
```
CUSUMConfig          # Agent A
LabelConfig          # Agent B
FeatureConfig        # Agent C
CPCVConfig           # Agent D
BetSizingConfig      # Agent E
TrainingConfig       # Agent F
InferenceConfig      # Agent G
MLPipelineConfig     # Top-level, composes all above
```

---

## 11. Verification Checklist

| Step | Command / Method |
|---|---|
| Unit tests | `uv run pytest tests/ml/ -x -v` |
| Bundle validation | `databricks bundle validate` |
| Backfill data integrity | Query `bronze.raw_aggtrades_backfill`: row count, date range, price range |
| Dollar bar sanity | Query `gold.dollar_bars`: bars/day ∈ [50, 100], OHLCV validity |
| CUSUM frequency | Query `silver.cusum_event_stats`: events/month ∈ [15, 70] |
| Label distribution | Query `silver.triple_barrier_labels`: class balance, maturation rate |
| Feature PIT-safety | For known events, verify no feature uses data at or after event_ts |
| CPCV correctness | Verify 15 splits, 5 paths, purge/embargo applied |
| Promotion gates | Query `silver.promotion_evidence`: all 10 metrics logged |
| Signal output | Query `silver.signals`: schema match, model versions populated |
| MLflow registry | Verify models registered in UC with signatures and Champion alias |
