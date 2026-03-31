# Project Specification: trading-databricks

**Version:** 1.0
**Last updated:** 2026-03-29
**Status:** Active

---

## 1. Project Objective and Scope

`trading-databricks` is the **Databricks transformation and feature engineering platform** within a larger algorithmic trading system. It is responsible for ingesting raw data landed in S3 by scraper repos, transforming it through a medallion architecture, and serving features to ML models and gen-AI agent pipelines.

**In scope:**
- Ingesting NDJSON files from S3 into Delta bronze tables via Auto Loader
- Transforming bronze → silver (dedup, type casting, normalization)
- Constructing alternative bars (dollar, volume, tick imbalance, dollar imbalance) from tick data
- Feature engineering for news data (sentiment, entities, classification)
- Serving gold layer Delta tables to ML pipelines and gen-AI agents

**Out of scope:**
- Data collection from external APIs (news-scraper-aws, tick-scraper-aws)
- Writing to S3 (this repo is read-only from S3)
- Frontend or API serving
- Model training or inference

---

## 2. System Overview

Three repos in the trading system:

```
news-scraper-aws                     tick-scraper-aws
       │                                    │
       ▼                                    ▼
  S3: news/tradingview/...            S3: ticks/binance/...
       │                                    │
       └──────────────┬─────────────────────┘
                      ▼
        ┌─────────────────────────────────────────┐
        │           trading-databricks             │
        │                                         │
        │   Auto Loader  ──►  bronze (Delta)       │
        │                         │               │
        │                         ▼               │
        │                      silver (Delta)      │
        │                         │               │
        │                         ▼               │
        │                      gold (Delta)        │
        │                       │       │         │
        │                       ▼       ▼         │
        │                 ML pipeline  Gen-AI      │
        │                              agents      │
        └─────────────────────────────────────────┘
```

| Repo | Responsibility | IaC | Deploys to |
|------|---------------|-----|-----------|
| `news-scraper-aws` | Scrape news → S3 NDJSON | SAM | AWS Lambda |
| `tick-scraper-aws` (planned) | Scrape BTC-USDT ticks → S3 | SAM | AWS Lambda |
| `trading-databricks` (this repo) | Ingest, transform, bars, features | DABs | Databricks |

---

## 3. Workspace + Unity Catalog Setup

**One Databricks workspace on AWS** (same region as S3 bucket), provisioned once manually.

**Environment isolation via catalogs:**

| Catalog | Purpose |
|---------|---------|
| `trading-signal-gen-dev` | Local development, experimentation |
| `trading-signal-gen-staging` | Pre-prod validation, CI/CD target |
| `trading-signal-prod` | Production pipelines |

Each catalog has identical schemas: `bronze`, `silver`, `gold`.

**Cross-account S3 access (provisioned, do not recreate):**
1. IAM role (`databricks-s3-ingest-5dfcc`) in the AWS account — Databricks workspace can assume it
2. Permissions: `s3:GetObject`, `s3:ListBucket` on `ai-trading-raw-ap-southeast-1`
3. Unity Catalog storage credential: `db_s3_credentials_databricks-s3-ingest-5dfcc-ai-trading-raw`
4. Unity Catalog external location: `db_s3_external_databricks-s3-ingest-5dfcc-ai-trading-raw` → `s3://ai-trading-raw-ap-southeast-1/`
5. Job owner/service principal must have `READ FILES` on the external location (workspace admins have implicit access)

---

## 4. Architecture

| Component | Technology | Notes |
|-----------|-----------|-------|
| Workspace | Databricks on AWS | Single workspace, catalog-per-environment |
| Governance | Unity Catalog | Fine-grained access, data lineage |
| Ingestion | Auto Loader (Structured Streaming) | availableNow=True trigger, scheduled |
| Storage | Delta Lake | Medallion: bronze / silver / gold |
| Compute | Classic clusters (m5.xlarge) | Spot workers, on-demand driver |
| IaC | Databricks Asset Bundles (DABs) | `databricks.yml` + `resources/*.yml` |
| CI/CD | GitHub Actions | validate → test → staging → prod (manual gate) |
| Package manager | uv | `pyproject.toml` + `uv.lock` |

---

## 5. News Pipeline

**Source:** `s3://ai-trading-raw-ap-southeast-1/news/tradingview/year=YYYY/month=MM/day=DD/*.ndjson`
**Schema contract:** defined in `news-scraper-aws/CLAUDE.md`

### Bronze — `bronze.news_raw`
- Auto Loader reads NDJSON, infers schema, appends to Delta table
- Trigger: `availableNow=True` — scheduled every 15 min
- Schema evolution: `addNewColumns` mode

### Silver — `silver.news_cleaned`
1. Dedup by `story_id` (keep latest `scraped_at`)
2. Cast `published_at`, `scraped_at` → `TIMESTAMP`
3. Normalize whitespace in `headline`, `body`
4. Flatten `provider` JSON into typed columns
5. Add `_ingested_at` metadata column

Write mode: MERGE on `story_id`.

### Gold — `gold.news_features`
- Sentiment scores (NLP library TBD — brainstorm before implementing)
- Named entity extraction: tickers, companies, people
- Topic/keyword classification flags
- `gold.news_latest` view: last 24h of cleaned articles for gen-AI agents

Write mode: overwrite.

---

## 6. Tick Pipeline

**Source:** `s3://ai-trading-raw-ap-southeast-1/ticks/binance/symbol={symbol}/year=YYYY/month=MM/day=DD/*.ndjson`
**Schema contract:** defined in `tick-scraper-aws/CLAUDE.md` (not yet built)

**Status:** Bronze and silver are stubs. Bar construction is scaffolded.

### Bronze — `bronze.ticks_raw`
Auto Loader reads tick NDJSON. Minimum expected fields: `timestamp`, `price`, `quantity`, `side`, `trade_id`, `symbol`, `exchange`, `batch_id`.

### Silver — `silver.ticks_cleaned`
1. Cast `timestamp` → `TIMESTAMP`
2. Cast `price`, `quantity` → `DECIMAL(18, 8)`
3. Compute `dollar_value = price * quantity`
4. Filter invalid ticks (`price <= 0` or `quantity <= 0`)
5. Dedup by `trade_id`
6. Add `_ingested_at` metadata

### Gold — Bar Construction

Four bar types (Lopez de Prado, AFML Chapter 2):

| Table | Sampling rule |
|-------|--------------|
| `gold.dollar_bars` | Sample every N dollars of `dollar_value` |
| `gold.volume_bars` | Sample every N units of `quantity` |
| `gold.tick_imbalance_bars` | Sample on cumulative tick direction imbalance threshold (adaptive) |
| `gold.dollar_imbalance_bars` | Sample on cumulative dollar imbalance threshold (adaptive) |

**Bar output schema (all bar types):**

| Column | Type | Description |
|--------|------|-------------|
| `bar_index` | `LONG` | Sequential bar number |
| `bar_start_time` | `TIMESTAMP` | Timestamp of first tick in bar |
| `bar_end_time` | `TIMESTAMP` | Timestamp of last tick in bar |
| `open` | `DOUBLE` | First price in bar |
| `high` | `DOUBLE` | Max price in bar |
| `low` | `DOUBLE` | Min price in bar |
| `close` | `DOUBLE` | Last price in bar |
| `volume` | `DOUBLE` | Sum of quantity in bar |
| `dollar_volume` | `DOUBLE` | Sum of dollar_value in bar |
| `tick_count` | `LONG` | Number of ticks in bar |

---

## 7. Compute Configuration

| Setting | Value |
|---------|-------|
| Node type | `m5.xlarge` |
| Workers | 1–2 (autoscale) |
| Spot instances | Enabled for workers |
| Driver | On-demand (`first_on_demand: 1`) |
| Runtime | Databricks Runtime 17.3 LTS ML (Scala 2.13) |
| S3 access | UC external location `db_s3_external_databricks-s3-ingest-5dfcc-ai-trading-raw` |

---

## 8. CI/CD

```
GitHub Actions:
  PR / push to feature branch  → databricks bundle validate + uv run pytest
  Push to main                 → deploy to trading-signal-gen-staging catalog
  Manual approval              → deploy to trading-signal-prod catalog
```

Required GitHub secrets:
- `DATABRICKS_HOST` — workspace URL
- `DATABRICKS_TOKEN` — service principal or PAT
- `DATABRICKS_BUNDLE_VAR_bucket` — S3 bucket name (e.g. `ai-trading-raw-ap-southeast-1`)
- `DATABRICKS_BUNDLE_VAR_checkpoint_base` — S3 checkpoint prefix (e.g. `s3://ai-trading-raw-ap-southeast-1/checkpoints`)

Create a `production` GitHub environment with required reviewers for the prod gate.

---

## 9. Testing Strategy

**Framework:** `pytest` with local `SparkSession`
**Runner:** `uv run pytest`
**Test location:** `tests/` — mirrors `src/` structure

### Principles
- Transformation functions are pure (`DataFrame` in, `DataFrame` out) — test with local Spark
- Job entry points (`run_*`) are not unit tested — integration tested via `databricks bundle run`
- Use `chispa` for DataFrame equality assertions
- Each domain module gets its own test file

### SparkSession fixture
Defined in `tests/conftest.py` — session-scoped, Delta Lake extensions enabled, `shuffle.partitions=4` for speed.

---

## 10. Code Style

Primary authority: `docs/python-programing-rules.md` (Fluent Python idioms).

Project-specific:
- `DataFrame.transform()` for chaining transformations
- Pure transformation functions separate from `run_*` entry points
- `@dataclass` for config objects (e.g. `ImbalanceBarConfig`)
- Keyword-only params for public functions with >2 args
- Bar construction: `.toPandas()` for stateful loop logic, then back to Spark

---

## 11. Git Workflow

- Feature branches off `main`
- CI triggers on push to `main` (validate → test → deploy to staging)
- Prod deployment requires manual approval in GitHub
- Commit convention: `type: description` (`feat`, `fix`, `chore`, `docs`, `test`, `refactor`)
- No direct pushes to `main`

---

## 12. Infrastructure Constraints

Do not modify without explicit user approval:
- `databricks.yml` targets — any change affects all environments
- `resources/*.yml` job schedules — changing schedule affects production run cadence
- Gold table schemas — downstream contracts for ML and gen-AI pipelines

---

## 13. Open Questions

| Question | Impact | When to Decide |
|----------|--------|----------------|
| NLP library for sentiment/entity extraction | gold news_features implementation | Brainstorm before implementing |
| Tick data NDJSON schema | Bronze ticks table definition | When building tick-scraper-aws |
| Bar construction thresholds (initial N values) | Dollar/volume/imbalance bar parameters | During tick pipeline implementation |
| Dashboard tool (Databricks SQL, Grafana, Streamlit?) | Frontend architecture | After first gold tables are populated |
| Gen-AI agent framework (LangChain, Databricks Agents?) | Gold layer schema design | Before building agent pipelines |
| Schema versioning (embed version in S3 key or record?) | Auto Loader schema evolution | Before adding 2nd news source |
