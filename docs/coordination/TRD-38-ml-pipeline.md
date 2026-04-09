# Parallel Work Coordination — TRD-38: Build ML Pipeline Phase 1A

> **Parent issue:** [TRD-38](https://linear.app/aaron-solo/issue/TRD-38/build-ml-pipeline-phase-1a-btcusdt-signal-generation)
>
> Each agent owns ONE section below. Read the full file at session start.
> Update your section during/after work. Never edit another agent's section.
>
> **Cross-worktree access:** Always read/write from the **main worktree** path, not your worktree.
>
> **Status values:** `not_started` → `in_progress` → `done` (or `blocked:<reason>`)

---

## Section 1: Schema & Config — TRD-39 (Wave 1)
- **Status:** done
- **Branch:** feat/TRD-39-ml-schema-config-dataclasses
- **Modified files:**
  - `src/trading_databricks/ml/__init__.py` (NEW)
  - `src/trading_databricks/ml/config.py` (NEW)
  - `src/trading_databricks/ml/schemas.py` (NEW)
  - `tests/ml/__init__.py` (NEW)
  - `tests/ml/test_config.py` (NEW)
  - `tests/ml/test_schemas.py` (NEW)
- **Exports (public APIs):**
  - `config.py`: `CUSUMConfig`, `LabelConfig`, `FeatureConfig`, `CPCVConfig`, `BetSizingConfig`, `TrainingConfig`, `InferenceConfig`, `MLPipelineConfig`
  - `schemas.py`: `CUSUM_EVENTS_SCHEMA`, `CUSUM_EVENT_STATS_SCHEMA`, `TRIPLE_BARRIER_LABELS_SCHEMA`, `FEATURES_TECHNICAL_SCHEMA`, `FEATURES_VOLATILITY_SCHEMA`, `FEATURES_VOLUME_SCHEMA`, `FEATURES_NEWS_SCHEMA`, `TRAINING_DATASET_SCHEMA`, `SIGNALS_SCHEMA`, `POSITIONS_SCHEMA`, `PROMOTION_EVIDENCE_SCHEMA`, `TRIAL_REGISTRY_SCHEMA`, `DATA_QUALITY_SCHEMA`
- **Needs from others:** None (foundation — no dependencies)
- **Notes:** Must merge to main before ANY Wave 2 agent branches. All agents import from these files.

---

## Section 2: Data Backfill — TRD-40 (Wave 2, DE teammate)
- **Status:** not_started
- **Branch:** —
- **Modified files:**
  - `scripts/backfill_download.py` (NEW)
- **Exports (public APIs):** None (produces `bronze.raw_aggtrades_backfill` Delta table)
- **Needs from others:** TRD-39 merged
- **Notes:** Downloads 6 months BTCUSDT from Binance Vision → UC Volume → Delta. No code dependencies on other agents.

---

## Section 3: Silver ticks_cleaned — TRD-41 (Wave 2, DE teammate)
- **Status:** not_started
- **Branch:** —
- **Modified files:**
  - `src/trading_databricks/ticks/silver.py` (MODIFY)
  - `tests/ticks/test_silver.py` (MODIFY)
- **Exports (public APIs):** `clean_ticks()` (Spark DataFrame transform)
- **Needs from others:** TRD-40 (backfill data in bronze)
- **Notes:** Cast types, dedup by trade_id, compute dollar_value. SDP streaming table.

---

## Section 4: Dollar Bar Construction — TRD-42 (Wave 2, DE teammate)
- **Status:** not_started
- **Branch:** —
- **Modified files:**
  - `src/trading_databricks/ticks/gold/dollar_bars.py` (MODIFY)
  - `tests/ticks/test_dollar_bars.py` (MODIFY)
- **Exports (public APIs):** `construct_dollar_bars()` (Polars function)
- **Needs from others:** TRD-41 (silver.ticks_cleaned)
- **Notes:** Stateful accumulation using Polars. SDP materialized view. Threshold calibrated from first 14 days.

---

## Section 5: CUSUM Event Detection — TRD-43 (Wave 2, Agent A)
- **Status:** not_started
- **Branch:** —
- **Modified files:**
  - `src/trading_databricks/ml/cusum.py` (NEW)
  - `tests/ml/test_cusum.py` (NEW)
- **Exports (public APIs):**
  - `detect_cusum_events(bars: pl.DataFrame, *, config: CUSUMConfig) -> pl.DataFrame`
  - `compute_cusum_stats(events: pl.DataFrame) -> pl.DataFrame`
- **Needs from others:** `CUSUMConfig`, `CUSUM_EVENTS_SCHEMA`, `CUSUM_EVENT_STATS_SCHEMA` from Section 1
- **Notes:** Pure Polars. Testable with synthetic bars. EWMA λ=0.94, symmetric CUSUM, 5-bar min spacing.

---

## Section 6: Triple-Barrier Labeling — TRD-44 (Wave 2, Agent B)
- **Status:** not_started
- **Branch:** —
- **Modified files:**
  - `src/trading_databricks/ml/labeling.py` (NEW)
  - `tests/ml/test_labeling.py` (NEW)
- **Exports (public APIs):**
  - `compute_triple_barrier_labels(events: pl.DataFrame, bars: pl.DataFrame, *, config: LabelConfig) -> pl.DataFrame`
- **Needs from others:** `LabelConfig`, `TRIPLE_BARRIER_LABELS_SCHEMA` from Section 1
- **Notes:** Pure Polars. Asymmetric barriers (2σ upper, 1σ lower). Only matured events labeled. avg_uniqueness computed.

---

## Section 7: Feature Engineering — TRD-45 (Wave 2, Agent C)
- **Status:** not_started
- **Branch:** —
- **Modified files:**
  - `src/trading_databricks/ml/features/__init__.py` (NEW)
  - `src/trading_databricks/ml/features/technical.py` (NEW)
  - `src/trading_databricks/ml/features/volatility.py` (NEW)
  - `src/trading_databricks/ml/features/volume.py` (NEW)
  - `src/trading_databricks/ml/features/news.py` (NEW)
  - `src/trading_databricks/ml/features/assembly.py` (NEW)
  - `tests/ml/test_features_technical.py` (NEW)
  - `tests/ml/test_features_volatility.py` (NEW)
  - `tests/ml/test_features_volume.py` (NEW)
  - `tests/ml/test_features_news.py` (NEW)
  - `tests/ml/test_features_assembly.py` (NEW)
- **Exports (public APIs):**
  - `compute_technical(events, bars, *, config) -> pl.DataFrame`
  - `compute_volatility(events, bars, *, config) -> pl.DataFrame`
  - `compute_volume(events, bars, *, config) -> pl.DataFrame`
  - `compute_news(events, news_features, *, config) -> pl.DataFrame`
  - `assemble_training_dataset(events, labels, *feature_dfs) -> pl.DataFrame`
- **Needs from others:** `FeatureConfig`, all `FEATURES_*_SCHEMA`, `TRAINING_DATASET_SCHEMA` from Section 1
- **Notes:** Pure Polars. All PIT-safe (strictly < event_ts). News features return zeros when no data. 16 features at launch, 51 with news.

---

## Section 8: CPCV + Sample Weights + Promotion — TRD-46 (Wave 2, Agent D)
- **Status:** not_started
- **Branch:** —
- **Modified files:**
  - `src/trading_databricks/ml/training/__init__.py` (NEW)
  - `src/trading_databricks/ml/training/cpcv.py` (NEW)
  - `src/trading_databricks/ml/training/sample_weights.py` (NEW)
  - `src/trading_databricks/ml/training/promotion.py` (NEW)
  - `tests/ml/test_cpcv.py` (NEW)
  - `tests/ml/test_sample_weights.py` (NEW)
  - `tests/ml/test_promotion.py` (NEW)
- **Exports (public APIs):**
  - `CPCVSplitter` (sklearn-compatible CV splitter)
  - `compute_sample_weights(events, *, n_bars) -> pl.Series`
  - `evaluate_promotion_gates(cpcv_path_results, meta_label_metrics, trial_count) -> pl.DataFrame`
- **Needs from others:** `CPCVConfig`, `PROMOTION_EVIDENCE_SCHEMA`, `TRIAL_REGISTRY_SCHEMA` from Section 1
- **Notes:** Pure Python/numpy. N=6, k=2 → 15 splits, 5 paths. PSR/DSR formulas from AFML Ch. 14. 10 promotion gates.

---

## Section 9: Bet Sizing + Position — TRD-47 (Wave 2, Agent E)
- **Status:** not_started
- **Branch:** —
- **Modified files:**
  - `src/trading_databricks/ml/inference/__init__.py` (NEW)
  - `src/trading_databricks/ml/inference/bet_sizing.py` (NEW)
  - `src/trading_databricks/ml/inference/position.py` (NEW)
  - `tests/ml/test_bet_sizing.py` (NEW)
  - `tests/ml/test_position.py` (NEW)
- **Exports (public APIs):**
  - `compute_bet_sizes(predictions, *, sigma_probability, config) -> pl.DataFrame`
  - `compute_positions(signals) -> pl.DataFrame`
- **Needs from others:** `BetSizingConfig`, `SIGNALS_SCHEMA`, `POSITIONS_SCHEMA` from Section 1
- **Notes:** Pure Polars. AFML Ch. 10 formula. 5 discretization tiers. Thresholds: meta > 0.50 AND primary > 0.55.

---

## Section 10: Training Pipeline — TRD-48 (Wave 3, Agent F)
- **Status:** not_started
- **Branch:** —
- **Modified files:**
  - `src/trading_databricks/ml/training/primary.py` (NEW)
  - `src/trading_databricks/ml/training/meta_label.py` (NEW)
  - `tests/ml/test_training_primary.py` (NEW)
  - `tests/ml/test_training_meta_label.py` (NEW)
- **Exports (public APIs):**
  - `train_primary_model(training_data, *, config, cpcv_config, catalog) -> str`
  - `train_meta_label_model(primary_oos_predictions, features, *, config, cpcv_config, catalog) -> str`
- **Needs from others:** All exports from Sections 5–9. MLflow + UC Model Registry access.
- **Notes:** mlflow.sklearn.autolog(), RandomizedSearchCV with CPCVSplitter. Register in UC, set Champion alias on promotion. Polars → pandas only at sklearn .fit() boundary.

---

## Section 11: Inference Pipeline — TRD-49 (Wave 3, Agent G)
- **Status:** not_started
- **Branch:** —
- **Modified files:**
  - `src/trading_databricks/ml/inference/predict.py` (NEW)
  - `src/trading_databricks/ml/jobs/train.py` (NEW)
  - `src/trading_databricks/ml/jobs/infer.py` (NEW)
  - `resources/ml_training_job.yml` (NEW)
  - `resources/ml_inference_job.yml` (NEW)
- **Exports (public APIs):**
  - `run_inference(events, features, *, catalog) -> pl.DataFrame`
- **Needs from others:** All exports from Sections 5–9. `@Champion` models in MLflow UC.
- **Notes:** Lakeflow Job DAGs for training and inference. 4h inference cadence. Email notification.

---

## Section 12: Backfill Orchestration — TRD-50 (Wave 4, Agent H)
- **Status:** not_started
- **Branch:** —
- **Modified files:**
  - `src/trading_databricks/ml/backfill/__init__.py` (NEW)
  - `src/trading_databricks/ml/backfill/run_backfill.py` (NEW)
- **Exports (public APIs):** None (one-time orchestration script)
- **Needs from others:** ALL sections above must be done. `gold.dollar_bars` must be populated (Section 4).
- **Notes:** End-to-end: bars → CUSUM → labels → features → training → promotion → first inference. Validates entire system. Must pass all 11 verification checks from spec Section 11.
