# Trading Databricks Platform — Design Spec

**Date:** 2026-03-29
**Status:** Draft
**Revised:** 2026-03-30 — Updated for early-stop dedup, S3 topology refactor, and corrected EventBridge schedule.
**New repo:** `trading-databricks`

---

## Context

The `news-scraper-aws` Lambda scrapes financial news and lands NDJSON in S3 every 2 minutes. It uses SSM-based watermark tracking (`/news-scraper/tv-watermark`) to early-stop pagination when already-seen articles are encountered, eliminating ~97–99% of duplicates at source. A planned `tick-scraper-aws` repo will do the same for BTC-USDT tick data from exchange APIs. Neither scraper repo handles transformation or feature engineering.

This spec defines a new `trading-databricks` repo that:
1. Ingests news and tick data from S3 into Delta tables (bronze)
2. Cleans and deduplicates (silver)
3. Constructs alternative bars (dollar, volume, tick imbalance, dollar imbalance) and news features (gold)
4. Serves a statistics dashboard (tool TBD)
5. Provides data for a gen-AI multi-agent financial news analysis system

The two data domains (news, ticks) are **independent pipelines** that share workspace infrastructure.

---

## 1. System Overview

Three repos in the trading system:

| Repo | Responsibility | IaC | Deploys to |
|------|---------------|-----|-----------|
| `news-scraper-aws` (existing) | Scrape news → S3 NDJSON | SAM | AWS Lambda |
| `tick-scraper-aws` (planned) | Scrape BTC-USDT ticks → S3 | SAM | AWS Lambda |
| `trading-databricks` (this spec) | Ingest, transform, bars, features | DABs | Databricks |

```
news-scraper-aws                   tick-scraper-aws
       │                                  │
       ▼                                  ▼
  S3: ai-trading-raw-.../            S3: ai-trading-raw-.../
      news/tradingview/...               ticks/binance/...
       │                                  │
       └──────────┬───────────────────────┘
                  ▼
        ┌─────────────────────────────────────┐
        │     trading-databricks              │
        │                                     │
        │  Auto Loader  ──►  bronze (Delta)   │
        │                      │              │
        │                      ▼              │
        │                   silver (Delta)    │
        │                      │              │
        │                      ▼              │
        │                   gold (Delta)      │
        │                    │       │        │
        │                    ▼       ▼        │
        │              ML pipeline  Gen-AI    │
        │                    agents           │
        │                                     │
        │              Dashboard (TBD)        │
        └─────────────────────────────────────┘
```

---

## 2. Workspace + Environment Setup

### 2.1 Workspace Provisioning (One-Time Manual)

- Create Databricks workspace via account console on **AWS, same region as S3 bucket**
- VPC: Databricks-managed (simplest)
- Create Unity Catalog metastore (one per region), attach to workspace

### 2.2 Environment Isolation via Catalogs

Single workspace, three catalogs:

| Catalog | Purpose |
|---------|---------|
| `trading-signal-gen-dev` | Local development, experimentation |
| `trading-signal-gen-staging` | Pre-prod validation, CI/CD target |
| `trading-signal-prod` | Production pipelines |

Each catalog has identical schemas: `bronze`, `silver`, `gold`.

### 2.3 Cross-Account S3 Access

The raw bucket `ai-trading-raw-ap-southeast-1` is owned by the `ai-trading-infra` stack (platform resource), not by any application stack. All scrapers write to it; Databricks reads from it.

1. Create IAM role in the AWS account that Databricks can assume
2. Grant: `s3:GetObject`, `s3:ListBucket` on `ai-trading-raw-ap-southeast-1`
3. Register **storage credential** in Unity Catalog → IAM role
4. Register **external location**: `s3://ai-trading-raw-ap-southeast-1/news/`
5. Create **instance profile** with the IAM role for classic cluster nodes

---

## 3. Repo Structure

```
trading-databricks/
  databricks.yml                        # DABs bundle config (targets: dev/staging/prod)
  resources/
    news_jobs.yml                       # News pipeline job definitions
    tick_jobs.yml                       # Tick pipeline job definitions
    clusters.yml                        # Shared cluster configs
  src/
    news/
      bronze/
        ingest_news.py                  # Auto Loader: S3 NDJSON → bronze.news_raw
      silver/
        clean_news.py                   # Dedup, type cast, normalize
      gold/
        news_features.py                # Sentiment, entities, classification
    ticks/
      bronze/
        ingest_ticks.py                 # Auto Loader: S3 → bronze.ticks_raw
      silver/
        clean_ticks.py                  # Validate, type cast, fill gaps
      gold/
        bar_construction.py             # Dollar, volume, imbalance bars
  tests/
    news/
      test_ingest_news.py
      test_clean_news.py
    ticks/
      test_ingest_ticks.py
      test_clean_ticks.py
      test_bar_construction.py
  docs/
    python-programing-rules.md          # Code style authority (shared)
    superpowers/
      specs/
        project-spec.md                 # Architecture, contracts, conventions
      plans/
        bootstrap-plan.md               # Bootstrap plan + feature workflow template
  CLAUDE.md                             # Agent orientation
```

### 3.1 DABs Environment Targets

```yaml
# databricks.yml
bundle:
  name: trading-databricks

targets:
  dev:
    default: true
    workspace:
      host: https://<workspace>.cloud.databricks.com
    variables:
      catalog: trading-signal-gen-dev
  staging:
    workspace:
      host: https://<workspace>.cloud.databricks.com
    variables:
      catalog: trading-signal-gen-staging
  prod:
    workspace:
      host: https://<workspace>.cloud.databricks.com
    variables:
      catalog: trading-signal-prod
```

All targets read from the same S3 bucket.

---

## 4. News Pipeline

### 4.1 Bronze — `bronze.news_raw`

Auto Loader (Structured Streaming) with `availableNow=True` trigger — processes all new files then terminates. Scheduled every 15 min via DABs job schedule.

```python
(spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", checkpoint_path)
    .option("cloudFiles.inferColumnTypes", "true")
    .load("s3://ai-trading-raw-ap-southeast-1/news/tradingview/year=*/month=*/day=*/")
    .writeStream
    .option("checkpointLocation", checkpoint_path)
    .trigger(availableNow=True)
    .toTable(f"{catalog}.bronze.news_raw"))
```

- **availableNow=True** — processes all files since last checkpoint then exits cleanly; no persistent streaming context needed
- **Scheduled every 15 min** — scraper runs every 2 min; 15-min window batches ~7 files per run
- **Auto Loader tracks processed files** — no reprocessing
- **Schema evolution** — new fields added automatically
- **Append-only** — no updates or deletes

Bronze table schema mirrors NDJSON record contract:

| Column | Type | Source |
|--------|------|--------|
| `story_id` | `STRING` | NDJSON field |
| `headline` | `STRING` | NDJSON field |
| `url` | `STRING` | NDJSON field |
| `published_at` | `STRING` | NDJSON field (cast in silver) |
| `source` | `STRING` | NDJSON field |
| `source_feed` | `STRING` | NDJSON field |
| `body` | `STRING` | NDJSON field |
| `ast_description_raw` | `STRING` | NDJSON field (JSON string) |
| `provider` | `STRING` | NDJSON field (JSON string) |
| `scraped_at` | `STRING` | NDJSON field (cast in silver) |
| `fetch_method` | `STRING` | NDJSON field |
| `batch_id` | `STRING` | NDJSON field. Format: `batch_{YYYYMMDDTHHMMSSZ}_{8-char-hex}` |
| `_rescued_data` | `STRING` | Auto Loader rescue column |
| `_metadata` | `STRUCT` | Auto Loader file metadata |

### 4.2 Silver — `silver.news_cleaned`

Scheduled job, triggered after bronze ingestion:

1. **Dedup** by `story_id` (keep latest `scraped_at`). Note: the scraper's SSM-based early-stop dedup (`/news-scraper/tv-watermark`) eliminates ~97–99% of duplicates at ingestion time. Silver dedup is a safety net for edge cases (Lambda cold-start races, watermark resets).
2. **Type cast** `published_at`, `scraped_at` → `TIMESTAMP`
3. **Flatten** `provider` JSON into typed columns
4. **Normalize** text (whitespace, unicode) in `headline`, `body`
5. **Retain** `ast_description_raw` for gen-AI agent use
6. **Add** `_ingested_at` metadata column

Write mode: MERGE on `story_id`.

### 4.3 Gold — `gold.news_features`

Scheduled job, triggered after silver completes:

- Sentiment scores (headline + body)
- Named entity extraction (tickers, companies, people)
- Keyword/topic classification flags
- `gold.news_latest` view for gen-AI agent queries

---

## 5. Tick Data Pipeline

### 5.1 Bronze — `bronze.ticks_raw`

Auto Loader reads tick NDJSON from S3 at `s3://ai-trading-raw-ap-southeast-1/ticks/binance/symbol=BTCUSDT/year=YYYY/month=MM/day=DD/`. Schema TBD when `tick-scraper-aws` is built.

Expected fields (minimum):
- `timestamp` — exchange timestamp (microsecond precision)
- `price` — trade price
- `quantity` — trade quantity
- `side` — buy/sell
- `trade_id` — exchange trade ID

### 5.2 Silver — `silver.ticks_cleaned`

- Validate timestamps, handle gaps/outliers
- Cast types, enforce constraints
- Compute `dollar_value` = `price * quantity`
- Add `_ingested_at` metadata

### 5.3 Gold — Bar Construction

Four bar types, each as a separate Delta table:

| Table | Sampling Rule |
|-------|--------------|
| `gold.dollar_bars` | Sample every N dollars traded |
| `gold.volume_bars` | Sample every N units of volume |
| `gold.tick_imbalance_bars` | Sample when cumulative tick imbalance exceeds threshold |
| `gold.dollar_imbalance_bars` | Sample when cumulative dollar imbalance exceeds threshold |

Each bar record contains:
- `bar_start_time`, `bar_end_time`
- `open`, `high`, `low`, `close`
- `volume`, `dollar_volume`
- `tick_count`
- `bar_index` (sequential)
- Bar-type-specific fields (threshold, cumulative imbalance, etc.)

---

## 6. Compute

Classic clusters with Databricks Jobs.

| Setting | Value |
|---------|-------|
| Node type | `m5.xlarge` (or equivalent) |
| Workers | 1–2 (autoscale) |
| Spot instances | Enabled for workers |
| Driver | On-demand |
| Runtime | Latest LTS |
| Instance profile | IAM role with S3 read access |

Bronze ingestion is lightweight. Bar construction is heavier — scale workers as tick volume grows.

---

## 7. CI/CD

```
GitHub Actions:
  push to feature branch  → databricks bundle validate
  push to main            → databricks bundle deploy -t staging
  manual approval         → databricks bundle deploy -t prod
```

Steps:
1. `databricks bundle validate` — schema check, syntax validation
2. `databricks bundle deploy -t staging` — deploy to staging catalog
3. Run integration tests against `trading-signal-gen-staging` catalog
4. Manual approval gate
5. `databricks bundle deploy -t prod` — deploy to production

---

## 8. Claude Code Project Setup

The repo is bootstrapped for AI-assisted development with Claude Code.

### 8.1 Files to Create

| File | Purpose | Pattern from |
|------|---------|-------------|
| `CLAUDE.md` | Agent behavior, boundaries, commands, navigation | `news-scraper-aws/CLAUDE.md` |
| `docs/superpowers/specs/project-spec.md` | Architecture, schema contracts, conventions | `news-scraper-aws` project-spec |
| `docs/superpowers/plans/bootstrap-plan.md` | Bootstrap plan + feature workflow template | `news-scraper-aws` bootstrap-plan |
| `docs/python-programing-rules.md` | Code style authority | Shared with scraper repo |

### 8.2 CLAUDE.md Scope

Non-negotiable rules:
- Follow `python-programing-rules.md` for all Python code
- No secrets in code — use Databricks secrets or Unity Catalog credentials
- Never modify `databricks.yml` targets without explicit approval
- Delta table schemas in `gold.*` are contracts — field changes require approval
- Run tests before claiming work is done

Boundaries table:

| Always | Ask First | Never |
|--------|-----------|-------|
| Follow `python-programing-rules.md` | Modify `databricks.yml` | Hardcode credentials |
| Use PySpark for transformations | Add new job definitions | Delete production Delta tables |
| Write to Delta tables via DABs jobs | Change cluster configs | Push directly to `main` |
| Keep domain separation (news/ticks) | Add new dependencies | Modify S3 bucket structure |

Commands:
```bash
databricks bundle validate              # Validate bundle config
databricks bundle deploy -t dev         # Deploy to dev
databricks bundle run <job> -t dev      # Run a specific job in dev
pytest tests/                           # Run unit tests locally
```

### 8.3 Feature Workflow

Same as `news-scraper-aws`:
1. Brainstorm → design doc → implementation plan → execute
2. Specs: `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`
3. Plans: `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`
4. Run tests after each task
5. Commit after each task

---

## 9. Dashboard

**Tool:** TBD — decide after pipelines are running. Candidates: Databricks SQL Dashboard, Grafana, Streamlit.

**Data source:** Gold layer Delta tables (`gold.news_features`, `gold.dollar_bars`, etc.).

---

## 10. Open Questions

| Question | Impact | When to Decide |
|----------|--------|----------------|
| Schema versioning for NDJSON (embed in S3 path or record?) | Affects Auto Loader schema evolution | Before adding 2nd news source |
| Same AWS account or cross-account for Databricks? | IAM trust policy complexity. Bucket name is now deterministic: `ai-trading-raw-ap-southeast-1` (no account ID). | During workspace provisioning |
| Gen-AI agent framework (LangChain, custom, Databricks Agents?) | Gold layer schema design | Before building agent pipelines |
| Tick data NDJSON schema | Bronze table definition for ticks | When building tick-scraper-aws |
| Bar construction thresholds (initial values for N) | Dollar/volume/imbalance bar parameters | During tick pipeline implementation |
| Dashboard tool selection | Frontend architecture | After first gold tables are populated |

---

## 11. Relationship to Other Repos

| Concern | news-scraper-aws | tick-scraper-aws | trading-databricks |
|---------|-----------------|-----------------|-------------------|
| Data collection | News articles | BTC-USDT ticks | No |
| S3 writes | Yes | Yes | No |
| Dedup watermark (SSM) | Yes (`/news-scraper/tv-watermark`) | TBD | No (relies on upstream) |
| S3 reads | No | No | Yes |
| Transformation | No | No | Yes |
| Bar construction | No | No | Yes |
| Feature engineering | No | No | Yes |
| Schema authority | News NDJSON | Tick NDJSON | Delta table schemas |
| IaC | SAM | SAM | DABs |
| CI/CD | GitHub Actions → SAM | GitHub Actions → SAM | GitHub Actions → DABs |
