# [TRD-39] Define ML Schema Contracts & Frozen Config Dataclasses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the foundational data contracts (`StructType` schemas) and frozen configuration dataclasses for the ML Pipeline (Phase 1A).

**Architecture:** Pure Python configuration and PySpark schema definitions. No Spark session is required to instantiate or test these module-level constants. `MLPipelineConfig` will compose other configuration dataclasses using `dataclasses.field(default_factory=...)`.

**Tech Stack:** Python 3.12, PySpark (`pyspark.sql.types`), `dataclasses`, `pytest`, `uv`

---

### Task 1: Branch Setup and Package Initialization

**Files:**
- Create: `src/trading_databricks/ml/__init__.py`
- Create: `tests/ml/__init__.py`

- [ ] **Step 1: Create a new Git branch**

```bash
git checkout main && git pull
git checkout -b feat/TRD-39-ml-schema-config-dataclasses
```
Expected output: Switched to a new branch 'feat/TRD-39-ml-schema-config-dataclasses'

- [ ] **Step 2: Create the `ml` package structure**

```bash
mkdir -p src/trading_databricks/ml
touch src/trading_databricks/ml/__init__.py
mkdir -p tests/ml
touch tests/ml/__init__.py
```

- [ ] **Step 3: Commit**

```bash
git add src/trading_databricks/ml/__init__.py tests/ml/__init__.py
git commit -m "TRD-39 chore: initialize ml package structure"
```

---

### Task 2: Implement Frozen Config Dataclasses

**Files:**
- Create: `tests/ml/test_config.py`
- Create: `src/trading_databricks/ml/config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/test_config.py
import pytest
from trading_databricks.ml.config import (
    CUSUMConfig, LabelConfig, FeatureConfig, CPCVConfig, 
    BetSizingConfig, TrainingConfig, InferenceConfig, MLPipelineConfig
)

def test_cusum_config_defaults():
    cfg = CUSUMConfig()
    assert cfg.ewma_lambda == 0.94
    assert cfg.threshold_multiplier == 1.0
    assert cfg.min_inter_event_bars == 5
    assert cfg.lookback_days == 30

def test_feature_config_defaults():
    cfg = FeatureConfig()
    assert cfg.technical_windows == (5, 10, 14, 20, 26, 50)
    assert cfg.news_lookback_hours == 24

def test_ml_pipeline_config_composition():
    cfg = MLPipelineConfig()
    assert cfg.symbol == "BTCUSDT"
    assert cfg.catalog == "trading-signal-gen-dev"
    assert isinstance(cfg.cusum, CUSUMConfig)
    assert isinstance(cfg.feature, FeatureConfig)

def test_configs_are_frozen():
    cfg = CUSUMConfig()
    with pytest.raises(Exception):
        cfg.ewma_lambda = 0.99  # type: ignore
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_databricks.ml.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/trading_databricks/ml/config.py
from dataclasses import dataclass, field

@dataclass(frozen=True)
class CUSUMConfig:
    ewma_lambda: float = 0.94
    threshold_multiplier: float = 1.0
    min_inter_event_bars: int = 5
    lookback_days: int = 30

@dataclass(frozen=True)
class LabelConfig:
    upper_barrier_multiplier: float = 2.0
    lower_barrier_multiplier: float = 1.0
    vertical_barrier_days: int = 5
    zero_label_threshold: float = 0.1

@dataclass(frozen=True)
class FeatureConfig:
    technical_windows: tuple[int, ...] = (5, 10, 14, 20, 26, 50)
    volatility_windows: tuple[int, ...] = (20, 50)
    volume_windows: tuple[int, ...] = (5, 20)
    news_lookback_hours: int = 24

@dataclass(frozen=True)
class CPCVConfig:
    n_groups: int = 6
    k_test: int = 2
    embargo_pct: float = 0.01

@dataclass(frozen=True)
class BetSizingConfig:
    meta_threshold: float = 0.50
    primary_threshold: float = 0.55
    tiers: tuple[tuple[float, float], ...] = (
        (0.10, 0.00), (0.25, 0.15), (0.50, 0.35), (0.75, 0.60), (1.01, 0.85)
    )

@dataclass(frozen=True)
class TrainingConfig:
    algorithm: str = "RandomForestClassifier"
    n_iter_search: int = 50
    random_state: int = 42

@dataclass(frozen=True)
class InferenceConfig:
    cadence_hours: int = 4

@dataclass(frozen=True)
class MLPipelineConfig:
    cusum: CUSUMConfig = field(default_factory=CUSUMConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    cpcv: CPCVConfig = field(default_factory=CPCVConfig)
    bet_sizing: BetSizingConfig = field(default_factory=BetSizingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    symbol: str = "BTCUSDT"
    catalog: str = "trading-signal-gen-dev"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/ml/config.py tests/ml/test_config.py
git commit -m "TRD-39 feat: add frozen config dataclasses for ML pipeline"
```

---

### Task 3: Implement PySpark Schema Contracts

**Files:**
- Create: `tests/ml/test_schemas.py`
- Create: `src/trading_databricks/ml/schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/test_schemas.py
from trading_databricks.ml.schemas import (
    CUSUM_EVENTS_SCHEMA, CUSUM_EVENT_STATS_SCHEMA, TRIPLE_BARRIER_LABELS_SCHEMA,
    FEATURES_TECHNICAL_SCHEMA, FEATURES_VOLATILITY_SCHEMA, FEATURES_VOLUME_SCHEMA,
    FEATURES_NEWS_SCHEMA, TRAINING_DATASET_SCHEMA, SIGNALS_SCHEMA,
    POSITIONS_SCHEMA, PROMOTION_EVIDENCE_SCHEMA, TRIAL_REGISTRY_SCHEMA,
    DATA_QUALITY_SCHEMA
)

def test_cusum_events_schema_count():
    assert len(CUSUM_EVENTS_SCHEMA.fields) == 11

def test_signals_schema_count():
    assert len(SIGNALS_SCHEMA.fields) == 13

def test_features_news_schema_count():
    # Key columns (4) + 35 news features
    assert len(FEATURES_NEWS_SCHEMA.fields) == 39
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_databricks.ml.schemas'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/trading_databricks/ml/schemas.py
from pyspark.sql.types import (
    StructType, StructField, StringType, TimestampType, DoubleType, 
    LongType, IntegerType, DateType, BooleanType
)

CUSUM_EVENTS_SCHEMA = StructType([
    StructField("event_id", StringType(), False),
    StructField("symbol", StringType(), False),
    StructField("event_ts", TimestampType(), False),
    StructField("bar_index", LongType(), False),
    StructField("log_return", DoubleType(), False),
    StructField("cusum_side", StringType(), False),
    StructField("cusum_value", DoubleType(), False),
    StructField("threshold_used", DoubleType(), False),
    StructField("ewma_vol", DoubleType(), False),
    StructField("avg_bars_per_day", DoubleType(), False),
    StructField("_ingested_at", TimestampType(), False),
])

CUSUM_EVENT_STATS_SCHEMA = StructType([
    StructField("symbol", StringType(), False),
    StructField("date", DateType(), False),
    StructField("event_count", IntegerType(), False),
    StructField("threshold_used", DoubleType(), False),
    StructField("ewma_vol", DoubleType(), False),
    StructField("avg_bars_per_day", DoubleType(), False),
])

TRIPLE_BARRIER_LABELS_SCHEMA = StructType([
    StructField("event_id", StringType(), False),
    StructField("symbol", StringType(), False),
    StructField("event_ts", TimestampType(), False),
    StructField("label", IntegerType(), False),
    StructField("barrier_hit", StringType(), False),
    StructField("return_pct", DoubleType(), False),
    StructField("upper_level", DoubleType(), False),
    StructField("lower_level", DoubleType(), False),
    StructField("vertical_bar", LongType(), False),
    StructField("matured_at", TimestampType(), False),
    StructField("avg_uniqueness", DoubleType(), False),
])

FEATURES_TECHNICAL_SCHEMA = StructType([
    StructField("event_id", StringType(), False),
    StructField("symbol", StringType(), False),
    StructField("event_ts", TimestampType(), False),
    StructField("computed_at", TimestampType(), False),
    StructField("rsi_14", DoubleType(), True),
    StructField("macd_signal", DoubleType(), True),
    StructField("bb_pct", DoubleType(), True),
    StructField("sma_ratio_5_20", DoubleType(), True),
    StructField("sma_ratio_10_50", DoubleType(), True),
    StructField("log_return_5", DoubleType(), True),
    StructField("log_return_20", DoubleType(), True),
])

FEATURES_VOLATILITY_SCHEMA = StructType([
    StructField("event_id", StringType(), False),
    StructField("symbol", StringType(), False),
    StructField("event_ts", TimestampType(), False),
    StructField("computed_at", TimestampType(), False),
    StructField("realized_vol_20", DoubleType(), True),
    StructField("realized_vol_50", DoubleType(), True),
    StructField("parkinson_vol_20", DoubleType(), True),
    StructField("garman_klass_20", DoubleType(), True),
    StructField("vol_ratio_20_50", DoubleType(), True),
])

FEATURES_VOLUME_SCHEMA = StructType([
    StructField("event_id", StringType(), False),
    StructField("symbol", StringType(), False),
    StructField("event_ts", TimestampType(), False),
    StructField("computed_at", TimestampType(), False),
    StructField("vwap_dev_20", DoubleType(), True),
    StructField("relative_volume_20", DoubleType(), True),
    StructField("obv_slope_20", DoubleType(), True),
    StructField("dollar_volume_ratio", DoubleType(), True),
])

# Dynamically generate News features (4 metrics x 8 L1 categories + 3 global)
_news_fields = [
    StructField("event_id", StringType(), False),
    StructField("symbol", StringType(), False),
    StructField("event_ts", TimestampType(), False),
    StructField("computed_at", TimestampType(), False),
]
for l1 in ["Sentiment", "DeFi", "Supply", "Exchange", "Macro", "Adoption", "Security", "Regulation"]:
    _news_fields.extend([
        StructField(f"news_count_{l1}", IntegerType(), True),
        StructField(f"news_max_conf_{l1}", DoubleType(), True),
        StructField(f"news_direction_{l1}", IntegerType(), True),
        StructField(f"news_accepted_ratio_{l1}", DoubleType(), True),
    ])
_news_fields.extend([
    StructField("news_volume_24h", IntegerType(), True),
    StructField("news_sentiment_score", DoubleType(), True),
    StructField("has_news_24h", BooleanType(), True),
])
FEATURES_NEWS_SCHEMA = StructType(_news_fields)

TRAINING_DATASET_SCHEMA = StructType([
    StructField("dataset_version", StringType(), False),
    StructField("event_id", StringType(), False),
    StructField("symbol", StringType(), False),
    StructField("event_ts", TimestampType(), False),
    StructField("label", IntegerType(), True),
    StructField("avg_uniqueness", DoubleType(), False),
    StructField("cpcv_group", IntegerType(), False),
    # ... (Technical 7)
    StructField("rsi_14", DoubleType(), True),
    StructField("macd_signal", DoubleType(), True),
    StructField("bb_pct", DoubleType(), True),
    StructField("sma_ratio_5_20", DoubleType(), True),
    StructField("sma_ratio_10_50", DoubleType(), True),
    StructField("log_return_5", DoubleType(), True),
    StructField("log_return_20", DoubleType(), True),
    # ... (Volatility 5)
    StructField("realized_vol_20", DoubleType(), True),
    StructField("realized_vol_50", DoubleType(), True),
    StructField("parkinson_vol_20", DoubleType(), True),
    StructField("garman_klass_20", DoubleType(), True),
    StructField("vol_ratio_20_50", DoubleType(), True),
    # ... (Volume 4)
    StructField("vwap_dev_20", DoubleType(), True),
    StructField("relative_volume_20", DoubleType(), True),
    StructField("obv_slope_20", DoubleType(), True),
    StructField("dollar_volume_ratio", DoubleType(), True),
    # ... (News 35)
    *[_f for _f in _news_fields if _f.name not in ("event_id", "symbol", "event_ts", "computed_at")],
    StructField("created_at", TimestampType(), False),
])

SIGNALS_SCHEMA = StructType([
    StructField("signal_id", StringType(), False),
    StructField("symbol", StringType(), False),
    StructField("event_id", StringType(), False),
    StructField("event_ts", TimestampType(), False),
    StructField("primary_side", IntegerType(), False),
    StructField("primary_confidence", DoubleType(), False),
    StructField("meta_label", IntegerType(), False),
    StructField("meta_confidence", DoubleType(), False),
    StructField("raw_bet_size", DoubleType(), False),
    StructField("discretized_weight", DoubleType(), False),
    StructField("primary_model_version", IntegerType(), False),
    StructField("meta_model_version", IntegerType(), False),
    StructField("created_at", TimestampType(), False),
])

POSITIONS_SCHEMA = StructType([
    StructField("symbol", StringType(), False),
    StructField("side", StringType(), False),
    StructField("weight", DoubleType(), False),
    StructField("confidence", DoubleType(), False),
    StructField("signal_id", StringType(), False),
    StructField("updated_at", TimestampType(), False),
])

PROMOTION_EVIDENCE_SCHEMA = StructType([
    StructField("run_id", StringType(), False),
    StructField("model_type", StringType(), False),
    StructField("model_version", IntegerType(), False),
    StructField("dataset_version", StringType(), False),
    StructField("psr", DoubleType(), False),
    StructField("dsr", DoubleType(), False),
    StructField("max_drawdown", DoubleType(), False),
    StructField("time_under_water", IntegerType(), False),
    StructField("cost_robust", BooleanType(), False),
    StructField("strategy_failure_prob", DoubleType(), False),
    StructField("meta_precision", DoubleType(), False),
    StructField("meta_recall", DoubleType(), False),
    StructField("meta_f1", DoubleType(), False),
    StructField("meta_logloss", DoubleType(), False),
    StructField("all_gates_pass", BooleanType(), False),
    StructField("failure_reasons", StringType(), True),
    StructField("evaluated_at", TimestampType(), False),
])

TRIAL_REGISTRY_SCHEMA = StructType([
    StructField("trial_id", StringType(), False),
    StructField("run_id", StringType(), False),
    StructField("model_type", StringType(), False),
    StructField("algorithm", StringType(), False),
    StructField("sharpe", DoubleType(), False),
    StructField("created_at", TimestampType(), False),
])

DATA_QUALITY_SCHEMA = StructType([
    StructField("check_id", StringType(), False),
    StructField("check_name", StringType(), False),
    StructField("symbol", StringType(), False),
    StructField("status", StringType(), False),
    StructField("metric_value", DoubleType(), False),
    StructField("threshold", DoubleType(), False),
    StructField("message", StringType(), False),
    StructField("checked_at", TimestampType(), False),
])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_schemas.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/ml/schemas.py tests/ml/test_schemas.py
git commit -m "TRD-39 feat: add all PySpark schema contracts for ML pipeline"
```

---

### Task 4: Create Parallel Coordination Document

**Files:**
- Create: `docs/coordination/TRD-38-ml-pipeline.md`

- [ ] **Step 1: Write coordination document**

Create `docs/coordination/TRD-38-ml-pipeline.md`:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add docs/coordination/TRD-38-ml-pipeline.md
git commit -m "TRD-39 docs: create TRD-38 parallel work coordination document"
```

---

### Task 5: Final Validation and Handoff

- [ ] **Step 1: Run complete test suite to ensure no regressions**

Run: `uv run pytest -v`
Expected: All tests pass.

- [ ] **Step 2: Push branch to GitHub**

```bash
git push -u origin feat/TRD-39-ml-schema-config-dataclasses
```