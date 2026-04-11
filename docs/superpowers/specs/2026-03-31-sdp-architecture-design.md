# SDP Architecture Design: trading-databricks

**Date:** 2026-03-31
**Ticket:** TRD-12
**Status:** Approved
**Supersedes:** `trading-databricks-design.md` (Section 3 repo structure, Section 4/5 pipeline architecture, Section 6 compute)

---

## Context

The original architecture used `spark_python_task` Databricks jobs with classic clusters and a flat `src/news/bronze/`, `src/news/silver/`, `src/news/gold/` structure. All 6 source files were stubs with zero implementation.

This redesign replaces that with Spark Declarative Pipelines (SDP) and scales to 4 domains before any implementation begins, ensuring the architecture is production-grade from the start.

---

## Key Decisions

### 1. Spark Declarative Pipelines (SDP)
Use the modern `from pyspark import pipelines as dp` API (not legacy `import dlt`). SDP pipelines are defined in `pipeline.py` files using `@dp.table`, `@dp.materialized_view`, and `dp.create_auto_cdc_flow`. The `pipeline.py` files are **not locally testable** — they run only inside a Databricks pipeline context.

### 2. Domain-First Structure with Flat Transform Layers
Each domain (`news/`, `ticks/`, `ml/`, `agents/`) has:
- `pipeline.py` — thin SDP wrappers only, no transform logic
- `bronze.py`, `silver.py`, `gold.py` — pure PySpark functions, fully testable with pytest

This ensures transform logic is testable locally while SDP-specific decorators stay isolated.

### 3. Separate Schemas Per Layer
Tables are written to separate Unity Catalog schemas: `{catalog}.01_bronze.*`, `{catalog}.silver.*`, `{catalog}.03_gold.*`. Schema-level access control lets downstream consumers query gold without touching bronze/silver.

### 4. Serverless Compute
All SDP pipelines use `serverless: true`. Classic cluster config (`clusters.yml`) is deleted. No node type, spot/on-demand, or runtime version to manage.

### 5. AUTO CDC for Silver Dedup (News)
News silver uses `dp.create_auto_cdc_flow()` with SCD Type 1 (keep latest by `scraped_at`) instead of manual MERGE. This is the idiomatic SDP pattern for deduplication.

### 6. Materialized Views for Gold Bars
Tick bar construction (Lopez de Prado AFML Chapter 2) requires stateful processing over ordered ticks. `@dp.materialized_view()` (full recompute) is the correct SDP construct — streaming tables can't handle these cumulative algorithms.

---

## Package Structure

```
src/trading_databricks/
├── __init__.py
├── news/
│   ├── __init__.py
│   ├── pipeline.py          # @dp.table / @dp.materialized_view / dp.create_auto_cdc_flow
│   ├── bronze.py            # Pure: enrich_bronze()
│   ├── silver.py            # Pure: cast_timestamps, normalize_text, flatten_provider, clean_news
│   ├── gold.py              # Pure: sentiment, entities, news_features (stubs — TRD-4)
│   ├── schemas.py           # StructType definitions for news NDJSON contract
│   └── expectations.py      # SDP quality rule strings (@dp.expect_or_drop)
├── ticks/
│   ├── __init__.py
│   ├── pipeline.py          # @dp.table / @dp.materialized_view
│   ├── bronze.py            # Pure: enrich_bronze()
│   ├── silver.py            # Pure: cast_types, compute_dollar_value, clean_ticks
│   ├── gold/
│   │   ├── __init__.py      # Re-exports all build_* functions
│   │   ├── dollar_bars.py   # build_dollar_bars()
│   │   ├── volume_bars.py   # build_volume_bars()
│   │   ├── tick_imbalance_bars.py
│   │   └── dollar_imbalance_bars.py
│   ├── schemas.py           # StructType for tick NDJSON + bar schemas
│   └── expectations.py
├── ml/
│   ├── __init__.py
│   ├── pipeline.py          # @dp.materialized_view: gold → feature tables
│   ├── features.py          # Pure feature engineering (stubs — TBD)
│   └── jobs/
│       ├── __init__.py
│       └── train.py         # Non-SDP: MLflow training job entry point
├── agents/
│   ├── __init__.py
│   ├── tools.py             # UC function tools, retrieval logic (stubs — TRD-6)
│   └── serve.py             # Agent endpoint configuration (stubs — TRD-6)
└── common/
    ├── __init__.py
    ├── config.py            # @dataclass(frozen=True): DollarBarConfig, VolumeBarConfig, …
    ├── quality.py           # Shared expectation helper functions: not_null, positive, …
    └── types.py             # Shared type aliases
```

---

## SDP Pipeline Architecture

### News Pipeline (`resources/news_pipeline.pipeline.yml`)

| Layer | SDP Construct | Table | Notes |
|-------|--------------|-------|-------|
| Bronze | `@dp.table()` streaming | `bronze.news_raw` | `cloudFiles` Auto Loader, `@dp.expect_or_drop` quality gate |
| Silver | `dp.create_auto_cdc_flow()` | `silver.news_cleaned` | SCD Type 1 on `story_id`, sequenced by `scraped_at` |
| Gold | `@dp.materialized_view()` | `gold.news_features` | Sentiment, entities (stubs — TRD-4) |
| Gold | `@dp.temporary_view()` | `_news_latest` | Last 24h — pipeline-internal for gen-AI agents |

Bronze uses `cloudFiles.schemaLocation` pointing to a Volume:
`/Volumes/{catalog}/pipeline_metadata/trading_databricks_metadata/schemas/bronze_news_raw`

Silver AUTO CDC pattern:
```python
dp.create_streaming_table(name=f"{silver_schema}.news_cleaned", ...)
dp.create_auto_cdc_flow(
    target=f"{silver_schema}.news_cleaned",
    source=f"{bronze_schema}.news_raw",
    keys=["story_id"],
    sequence_by=col("scraped_at"),
    stored_as_scd_type="1",
)
```

### Tick Pipeline (`resources/tick_pipeline.pipeline.yml`)

| Layer | SDP Construct | Table | Notes |
|-------|--------------|-------|-------|
| Bronze | `@dp.table()` streaming | `bronze.ticks_raw` | `cloudFiles` from S3 NDJSON |
| Silver | `@dp.table()` streaming | `silver.ticks_cleaned` | `@dp.expect_or_drop` for `price > 0`, `quantity > 0` |
| Gold | `@dp.materialized_view()` | `gold.dollar_bars` | Calls `build_dollar_bars()` — full recompute |
| Gold | `@dp.materialized_view()` | `gold.volume_bars` | Calls `build_volume_bars()` |
| Gold | `@dp.materialized_view()` | `gold.tick_imbalance_bars` | Calls `build_tick_imbalance_bars()` |
| Gold | `@dp.materialized_view()` | `gold.dollar_imbalance_bars` | Calls `build_dollar_imbalance_bars()` |

Gold bars use `@dp.materialized_view()` (full recompute) because Lopez de Prado cumulative algorithms require stateful ordered processing.

### ML Pipeline (`resources/ml_pipeline.pipeline.yml`)

| Layer | SDP Construct | Table | Notes |
|-------|--------------|-------|-------|
| Features | `@dp.materialized_view()` | `feature_*` | Reads gold tables, produces ML-ready features |

Training runs as a separate non-SDP job (`ml/jobs/train.py`) with MLflow tracking.

---

## DABs Configuration

### New Variables Added to `databricks.yml`

```yaml
variables:
  bronze_schema:
    default: bronze
  silver_schema:
    default: silver
  gold_schema:
    default: gold
  schema_location_base:
    default: /Volumes/${var.catalog}/pipeline_metadata/trading_databricks_metadata/schemas
```

### Pipeline Resource Pattern

```yaml
# resources/news_pipeline.pipeline.yml
resources:
  pipelines:
    news_pipeline:
      name: news-pipeline-${bundle.target}
      catalog: ${var.catalog}
      serverless: true
      libraries:
        - glob:
            include: ../src/trading_databricks/news/pipeline.py
      configuration:
        bronze_schema: ${var.bronze_schema}
        silver_schema: ${var.silver_schema}
        gold_schema: ${var.gold_schema}
        schema_location_base: ${var.schema_location_base}
        source_bucket: ${var.bucket}
```

---

## Deleted Files (TRD-12)

| Deleted | Replaced by |
|---------|------------|
| `src/news/bronze/ingest_news.py` | `src/trading_databricks/news/pipeline.py` + `bronze.py` |
| `src/news/silver/clean_news.py` | `src/trading_databricks/news/silver.py` |
| `src/news/gold/news_features.py` | `src/trading_databricks/news/gold.py` |
| `src/ticks/bronze/ingest_ticks.py` | `src/trading_databricks/ticks/pipeline.py` + `bronze.py` |
| `src/ticks/silver/clean_ticks.py` | `src/trading_databricks/ticks/silver.py` |
| `src/ticks/gold/bar_construction.py` | `src/trading_databricks/ticks/gold/` (4 modules) |
| `resources/clusters.yml` | Serverless — no cluster config needed |
| `resources/news_jobs.yml` | `resources/news_pipeline.pipeline.yml` |
| `resources/tick_jobs.yml` | `resources/tick_pipeline.pipeline.yml` |

---

## Testing Approach

- **Transform modules** (`bronze.py`, `silver.py`, `gold.py`) are pure PySpark functions — fully testable with pytest via `DatabricksSession.builder.serverless()`
- **`pipeline.py` files** are NOT locally testable — they use SDP-specific decorators that only work inside a pipeline context
- Test files mirror package structure under `tests/`
- `tests/conftest.py` uses `DatabricksSession.builder.serverless().getOrCreate()` — requires `~/.databrickscfg`

---

## Coding Standards (SDP-Specific)

- `pipeline.py` contains **only** SDP decorator calls — no transform logic
- All transform logic lives in sibling modules (`bronze.py`, `silver.py`, `gold.py`)
- Use `DataFrame.transform()` for chaining: `df.transform(cast_timestamps).transform(normalize_text).transform(add_ingested_at)`
- Config objects: `@dataclass(frozen=True)` — e.g. `DollarBarConfig(threshold=1_000_000.0)`
- Keyword-only params (`*`) for all public functions with more than 2 arguments
- `spark` is an injected global inside `pipeline.py` (standard SDP pattern) — use `# noqa: F821` to suppress undefined-name linting warnings
