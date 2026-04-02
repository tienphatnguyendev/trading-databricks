# Tick Data Pipeline: Unified Plan

## Context

We're building an end-to-end tick data pipeline for a 5-minute BTC-USDT trading strategy. The work spans two repos and three phases: **prototype** (prove the algorithm), **data collection** (Binance → S3), and **production pipeline** (Databricks SDP). This plan merges all prior plans into a single source of truth.

---

## Phase 1: Prototype (repo: `tick-pipeline-prototype`)

**Goal:** Prove Tick Imbalance Bar construction and feature engineering work on real Binance data. Demonstrate information-driven sampling is superior to time clocks.

### Module Structure

| Module | Purpose | Medallion Layer |
|--------|---------|-----------------|
| `notebooks/tick_prototype_lib.py` | Core Python library (all logic) | — |
| `notebooks/tick_pipeline_prototype.py` | Databricks notebook orchestrator + visualization | — |

### `tick_prototype_lib.py` Functions

1. **`fetch_binance_agg_trades()`** — Fetch raw BTCUSDT aggTrades from Binance REST API (`api.binance.com/api/v3/aggTrades`). Handle pagination. Fetch 12-24h burn-in window. *(Bronze simulation)*
2. **`clean_and_prep_ticks()`** — Assign aggressor direction (`b_t`) via maker/taker flags, compute dollar volume. *(Silver simulation)*
3. **`compute_tick_imbalance_bars()`** — Dynamically sample TIBs using EWMA on expected tick imbalance. Core stateful Pandas loop:
   - Tick rule: `b_t ∈ {-1, +1}` from price changes
   - Burn-in: first N ticks → initialize EWMA of `P(b_t=1)` and `E[T]`
   - Sampling: accumulate `θ_T = Σ b_t`, sample bar when `|θ_T| ≥ E[T] × |2P(b_t=1) - 1|`
   *(Gold simulation)*
4. **`compute_microstructure_features()`** — OFI proxy, trade intensity, volatility spread on TIBs
5. **`generate_polling_dataset()`** — 5-min time series with AS-OF backward join (no lookahead bias), forward 5-min returns as target label

### `tick_pipeline_prototype.py` Notebook

- Setup: `%load_ext autoreload`, import lib
- Execute functions sequentially
- Plot: raw tick prices + TIB sampling points (vertical lines) + trade intensity
- Target: AS-OF join → `training_dataset` DataFrame

### Verification

- [ ] `tick_prototype_lib.py` imports into notebook without errors
- [ ] Plot shows non-uniform bar sampling (dense during volatility, sparse during quiet)
- [ ] AS-OF join binds bar features to 5-min grid strictly backward (no future leakage)

---

## Phase 2: Data Collection Architecture (repo: TBD — likely `tick-scraper-aws`)

**Goal:** Get real-time BTC-USDT tick data from Binance into S3 in the format `trading-databricks` expects. **Not yet built — design after prototype validates the algorithm.**

### Contract (from `trading-databricks` pipeline)

**S3 path:** `s3://ai-trading-raw-ap-southeast-1/ticks/binance/symbol={symbol}/year=YYYY/month=MM/day=DD/{batch_id}.ndjson`

**NDJSON schema:**
```json
{"timestamp": "2026-03-31T12:00:00.123456Z", "price": "67234.56000000", "quantity": "0.00150000", "side": "buy", "trade_id": "123456789", "symbol": "BTCUSDT", "exchange": "binance", "batch_id": "batch_20260331T120000Z_a1b2c3d4"}
```

### Open Decisions (post-prototype)

| Decision | Options to evaluate |
|----------|-------------------|
| Data source API | Binance WebSocket (real-time) vs REST API (poll) |
| Collector runtime | AWS Lambda, EC2/ECS, Databricks job |
| Intermediate storage | Direct S3, Kinesis → S3, SQS → Lambda → S3 |
| Batching strategy | Time-based flush vs size-based vs hybrid |
| Fault tolerance | Reconnection, source-level dedup, exactly-once |
| Deployment | New `tick-scraper-aws` repo (SAM/CDK) |

---

## Phase 3: Production Pipeline (repo: `trading-databricks`)

**Goal:** SDP pipeline transforming S3 tick data through Bronze → Silver → Gold, serving bars to ML and agent pipelines.

### Architecture Decisions (2026-03-31)

| Decision | Answer |
|----------|--------|
| Primary use case | **Live trading** — 5-min strategy |
| Pipeline trigger | **Triggered every 5 min** (`continuous: false`) |
| Gold compute | **Materialized views** with 30-day lookback window |
| Silver dedup | **Yes** — `dropDuplicatesWithinWatermark("trade_id")`, 10-min watermark |
| Clustering | **`["symbol", "timestamp"]`** for Bronze/Silver, **`["symbol", "bar_start_time"]`** for Gold |
| Multi-symbol | **Planned** — clustering keys prepared now |

### Pipeline Layers

| Layer | Table | SDP Construct | Key |
|-------|-------|---------------|-----|
| Bronze | `bronze.ticks_raw` | Streaming table (Auto Loader) | `trade_id, timestamp` |
| Silver | `silver.ticks_cleaned` | Streaming table (with watermark dedup) | `trade_id` |
| Gold | `gold.dollar_bars` | Materialized view (30-day window) | `bar_index` |
| Gold | `gold.volume_bars` | Materialized view (30-day window) | `bar_index` |
| Gold | `gold.tick_imbalance_bars` | Materialized view (30-day window) | `bar_index` |
| Gold | `gold.dollar_imbalance_bars` | Materialized view (30-day window) | `bar_index` |

### Latency SLA

Trade on Binance → scraper flush to S3 (~30-60s) → next pipeline trigger (0-5 min) → execution (~1 min) = **2-7 min end-to-end**

### Stateful Bars: Two-Phase Approach

**Phase 3a (initial):** MVs with 30-day lookback
- `WHERE timestamp >= current_timestamp() - INTERVAL 30 DAYS` in each `build_*_bars()`
- 30 days × ~5M ticks/day = ~150M rows, ~2 GB Pandas — safe on serverless
- Add `lookback_days: int = 30` to bar config dataclasses

**Phase 3b (future):** Incremental streaming via `applyInPandasWithState`
- O(new_ticks) per run instead of O(window_ticks)
- Separate ticket when 30-day window becomes insufficient

### Late Data & Ordering

- **Sort in Gold:** `ticks_df.orderBy("timestamp", "trade_id").toPandas()` in each `build_*_bars()`
- **Dedup in Silver:** `withWatermark("timestamp", "10 minutes").dropDuplicatesWithinWatermark(["trade_id"])`
- **No additional watermarking** for ordering — MV full recompute handles late arrivals

### Files to Modify

| File | Change |
|------|--------|
| `src/trading_databricks/common/config.py` | Add `lookback_days: int = 30` to bar config dataclasses |
| `src/trading_databricks/ticks/pipeline.py` | Update `cluster_by` to include `"symbol"` (6 tables) |
| `src/trading_databricks/ticks/silver.py` | Add watermark-based dedup on `trade_id` |
| `src/trading_databricks/ticks/gold/dollar_bars.py` | Implement with 30-day filter + ORDER BY |
| `src/trading_databricks/ticks/gold/volume_bars.py` | Implement with 30-day filter + ORDER BY |
| `src/trading_databricks/ticks/gold/tick_imbalance_bars.py` | Implement with 30-day filter + ORDER BY |
| `src/trading_databricks/ticks/gold/dollar_imbalance_bars.py` | Implement with 30-day filter + ORDER BY |
| `resources/tick_pipeline.pipeline.yml` | Uncomment schedule when data collector is ready |

---

## Cleanup: Delete Old Plan Files

- [ ] `trading-databricks/docs/plans/2026-03-31-tick-data-pipeline.md`
- [ ] `tick-pipeline-prototype/docs/2026-03-31-tick-prototype-plan.md`
- [ ] `tick-pipeline-prototype/docs/tick-pipeline-prototype-design.md`

---

## Execution Order

```
Phase 1: Prototype (tick-pipeline-prototype)
  → Prove TIB algorithm on real Binance data
  → Validate feature engineering + AS-OF join
  → Visual proof of information-driven sampling

Phase 2: Data Collection (tick-scraper-aws)
  → Design based on prototype learnings
  → Deploy collector writing NDJSON to S3

Phase 3: Production Pipeline (trading-databricks)
  → Port prototype logic to SDP pipeline
  → Implement all 4 bar types
  → Deploy with 5-min triggered schedule
```
