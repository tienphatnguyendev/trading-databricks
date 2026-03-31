# Daily Development Workflow

This guide walks you through the full development cycle: writing code locally, testing it, deploying to dev, promoting through staging, and shipping to production.

---

## 1. S3 Data Model — What You're Ingesting

This repo **reads from S3** — it never writes to it. Two upstream scraper repos land raw data into S3 as NDJSON files. Understanding this file structure is essential before working on any bronze ingestion code.

### S3 bucket layout

```
s3://ai-trading-raw-ap-southeast-1/
  news/
    tradingview/                          ← news-scraper-aws writes here
      year=2026/
        month=03/
          day=28/
            batch_20260328T120000Z_a1b2c3d4.ndjson
            batch_20260328T123000Z_e5f6a7b8.ndjson
          day=29/
            batch_20260329T000000Z_c9d0e1f2.ndjson
            ...
  ticks/                                 ← tick-scraper-aws will write here (not yet built)
    binance/
      symbol=BTCUSDT/
        year=2026/
          month=03/
            day=28/
              batch_20260328T120000Z_1a2b3c4d.ndjson
              ...
```

### Path convention (this is a contract — do not change)

| Data source | S3 prefix | Pattern |
|-------------|-----------|---------|
| News | `news/tradingview/` | `news/{source}/year=YYYY/month=MM/day=DD/{batch_id}.ndjson` |
| Ticks | `ticks/binance/` | `ticks/binance/symbol={symbol}/year=YYYY/month=MM/day=DD/{batch_id}.ndjson` |

- **Batch ID format:** `batch_{YYYYMMDDTHHMMSSZ}_{8-char-hex}` — one file per Lambda invocation
- **File format:** NDJSON (newline-delimited JSON) — one JSON object per line, UTF-8
- **Partitioning:** Hive-style (`year=YYYY/month=MM/day=DD/`) — enables Auto Loader partition discovery

### News NDJSON record schema

Each line in a news NDJSON file is a JSON object with these fields (defined in `news-scraper-aws`):

| Field | Type | Description |
|-------|------|-------------|
| `story_id` | `str` | Source-assigned unique ID |
| `headline` | `str` | Article title |
| `url` | `str` | Full canonical URL |
| `published_at` | `str \| null` | ISO 8601 UTC datetime |
| `source` | `str` | Source identifier (e.g. `"tradingview"`) |
| `source_feed` | `str` | Provider/feed ID within the source |
| `body` | `str \| null` | Plain text article body |
| `ast_description_raw` | `dict \| null` | Raw rich-text AST from source |
| `provider` | `dict` | Original provider metadata object |
| `scraped_at` | `str` | ISO 8601 UTC datetime of scrape |
| `fetch_method` | `str` | How body was fetched (e.g. `"api"`) |
| `batch_id` | `str` | Batch identifier for this Lambda run |

**Example record:**

```json
{"story_id":"abc123","headline":"BTC breaks 100k","url":"https://...","published_at":"2026-03-28T12:00:00Z","source":"tradingview","source_feed":"crypto","body":"Bitcoin surged...","ast_description_raw":null,"provider":{"name":"CoinDesk","logo":"..."},"scraped_at":"2026-03-28T12:05:00Z","fetch_method":"api","batch_id":"batch_20260328T120500Z_a1b2c3d4"}
```

### Tick NDJSON record schema (planned — not yet built)

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | `str` | Exchange timestamp, ISO 8601, microsecond precision |
| `price` | `str` | Trade price as decimal string |
| `quantity` | `str` | Trade quantity as decimal string |
| `side` | `str` | `"buy"` or `"sell"` |
| `trade_id` | `str` | Exchange-assigned trade ID |
| `symbol` | `str` | e.g. `"BTCUSDT"` |
| `exchange` | `str` | e.g. `"binance"` |
| `batch_id` | `str` | Batch identifier |

### How Auto Loader uses this structure

Auto Loader in the bronze ingestion jobs is configured to watch an S3 prefix (e.g. `s3://ai-trading-raw-ap-southeast-1/news/tradingview/`). It:

1. **Discovers new files** — tracks which files it has already processed via a checkpoint directory
2. **Reads each new NDJSON file** — parses one JSON object per line
3. **Appends to the bronze Delta table** — `{catalog}.bronze.news_raw`
4. **Never re-reads old files** — unless you delete the checkpoint (dev only)

The `availableNow=True` trigger means: "process all new files that arrived since last run, then stop." This is scheduled every 15 minutes for news.

### How this maps to the medallion layers

```
S3 (raw NDJSON files)                    You cannot change this — it's written by other repos
       │
       ▼
bronze.news_raw (Delta)                  Append-only, schema matches NDJSON + metadata
       │
       ▼
silver.news_cleaned (Delta)              Deduplicated, typed, normalized
       │
       ▼
gold.news_features (Delta)               ML-ready features (sentiment, entities, etc.)
```

Each layer adds structure and quality. Bronze is a faithful copy of S3. Silver is where you clean. Gold is where you engineer features.

---

## 2. Environment Map

The project has three environments. They all run the **same code and job definitions** — the only difference is which Unity Catalog catalog they write to.

| Environment | Catalog | How you get there | Who triggers it |
|-------------|---------|-------------------|-----------------|
| **dev** | `trading-signal-gen-dev` | `databricks bundle deploy -t dev` from your laptop | You, manually |
| **staging** | `trading-signal-gen-staging` | Auto-deploys when a PR merges to `main` (GitHub Actions) | CI/CD |
| **prod** | `trading-signal-prod` | Manual approval in GitHub `production` environment | You or a reviewer |

The `${var.catalog}` variable in `databricks.yml` switches the target catalog. Jobs are automatically named per environment — `news-pipeline-dev`, `news-pipeline-staging`, `news-pipeline-prod` — via `${bundle.target}`.

All three environments read from the **same S3 bucket** (`ai-trading-raw-ap-southeast-1`). Isolation comes from writing to different catalogs: `trading-signal-gen-dev.bronze.news_raw` vs `trading-signal-gen-staging.bronze.news_raw` vs `trading-signal-prod.bronze.news_raw`.

---

## 3. Local Development Loop

This is the inner loop you run on your laptop. It costs $0 and needs no Databricks cluster.

```bash
# 1. Create a feature branch (never work directly on main)
git checkout -b feat/news-silver-dedup

# 2. Write tests first
uv run pytest tests/news/test_clean_news.py -x -v   # should FAIL (function not implemented)

# 3. Write the implementation (pure transform functions in src/)
#    ... edit src/news/silver/clean_news.py ...

# 4. Run tests again
uv run pytest tests/news/test_clean_news.py -x -v   # should PASS

# 5. Run the full test suite
uv run pytest -v

# 6. Validate bundle config (catches YAML/config errors)
databricks bundle validate
```

The test suite uses a **local SparkSession** defined in `tests/conftest.py` with Delta Lake extensions. No remote cluster is involved.

---

## 4. Dev Environment — Your Personal Sandbox

After local tests pass, deploy to dev to test with **real S3 data** on a **real Databricks cluster**.

### Deploy and run

```bash
# Deploy your branch's job definitions to the dev workspace
databricks bundle deploy -t dev

# Run the full news pipeline (bronze → silver → gold)
databricks bundle run news_pipeline -t dev

# Or run a single task within the pipeline
databricks bundle run news_pipeline -t dev --task bronze_ingest
```

### What happens under the hood

1. DABs creates (or updates) a job called `news-pipeline-dev` in your Databricks workspace
2. The job spins up an `m5.xlarge` cluster (spot workers, on-demand driver)
3. Tasks read NDJSON from `s3://ai-trading-raw-ap-southeast-1/news/tradingview/...`
4. Results land in `trading-signal-gen-dev.bronze.news_raw`, `trading-signal-gen-dev.silver.news_cleaned`, etc.

### Inspect dev results

Open the Databricks workspace UI and query your dev tables:

```sql
-- In Databricks SQL Editor or a notebook
SELECT * FROM `trading-signal-gen-dev`.bronze.news_raw LIMIT 10;
SELECT count(*) FROM `trading-signal-gen-dev`.silver.news_cleaned;
```

Or from the CLI:

```bash
databricks workspace list /
```

**Dev is your playground.** You can safely re-run jobs, overwrite tables, or even drop tables in `trading-signal-gen-dev` without affecting staging or production.

---

## 5. Staging — Pre-Production Validation

You do **not** deploy to staging manually. CI/CD handles it automatically.

### How it works

1. **Open a PR** against `main` — GitHub Actions runs:
   - `databricks bundle validate` (config check)
   - `uv run pytest` (unit tests)
2. **Merge the PR** into `main` — GitHub Actions automatically:
   - Deploys to staging (`trading-signal-gen-staging` catalog)
   - Creates/updates `news-pipeline-staging` in the workspace

### Monitor staging

From the CLI:

```bash
# List recent workflow runs
gh run list --workflow=deploy.yml

# View details of a specific run
gh run view <run-id>
```

From the Databricks UI:

- Go to **Jobs** > `news-pipeline-staging` > **Run history**
- Verify the latest run completed successfully
- Spot-check staging tables: `SELECT * FROM \`trading-signal-gen-staging\`.silver.news_cleaned LIMIT 10`

Staging uses the same S3 source data as dev and prod. The point is to verify that your deployed job works end-to-end with real data before promoting to production.

---

## 6. Production — Manual Approval Gate

After staging succeeds, production deployment requires human approval.

### Promotion steps

1. Go to the **GitHub Actions** run that deployed to staging
2. The `deploy-prod` job shows **"Waiting for review"** — it's gated by the `production` GitHub environment
3. A reviewer (or you, if you have access) clicks **Approve**
4. GitHub Actions deploys to `trading-signal-prod` catalog

### Production schedules

Once deployed, jobs run on their configured schedules:

| Job | Schedule | Status |
|-----|----------|--------|
| `news-pipeline-prod` | Every 15 minutes | Active |
| `tick-pipeline-prod` | Every 5 minutes | Not yet scheduled (awaiting tick data) |

### Monitor production

- **Databricks UI** > Jobs > `news-pipeline-prod` > Run history
- Watch for: failed runs, increasing run duration, data freshness gaps
- Query prod tables: `SELECT max(scraped_at) FROM \`trading-signal-prod\`.bronze.news_raw` to check latest data

---

## 7. Lifecycle of a Code Change (End-to-End Example)

Here's the complete journey of implementing the silver news deduplication:

```
LOCAL DEVELOPMENT
  1. git checkout -b feat/news-silver-dedup
  2. Write test_deduplicate_keeps_latest_scraped_at in tests/news/test_clean_news.py
  3. uv run pytest tests/news/test_clean_news.py -x -v          → RED (not implemented)
  4. Implement deduplicate() in src/news/silver/clean_news.py
  5. uv run pytest tests/news/test_clean_news.py -x -v          → GREEN
  6. uv run pytest -v                                            → all GREEN
  7. databricks bundle validate                                  → OK

DEV ENVIRONMENT
  8. databricks bundle deploy -t dev                             → deploy to dev
  9. databricks bundle run news_pipeline -t dev                  → verify with real S3 data
 10. Check trading-signal-gen-dev tables in Databricks UI        → data looks correct

STAGING (automated)
 11. git add + commit + push + open PR
 12. GitHub Actions runs validate + tests                        → CI passes
 13. Reviewer approves and merges PR to main
 14. GitHub Actions auto-deploys to trading-signal-gen-staging
 15. Verify news-pipeline-staging succeeds in Databricks UI

PRODUCTION (manual gate)
 16. Approve prod deploy in GitHub Actions
 17. news-pipeline-prod picks up the change on next scheduled run (within 15 min)
```

---

## 8. Quick Reference

### Commands by stage

| Stage | Command | Purpose |
|-------|---------|---------|
| **Local** | `uv sync --extra dev` | Install dependencies |
| **Local** | `uv run pytest -v` | Run all tests |
| **Local** | `uv run pytest -x -v` | Stop on first failure |
| **Local** | `uv run pytest tests/news/` | News tests only |
| **Local** | `uv run pytest tests/ticks/` | Tick tests only |
| **Local** | `databricks bundle validate` | Validate config (no deploy) |
| **Dev** | `databricks bundle deploy -t dev` | Deploy to dev catalog |
| **Dev** | `databricks bundle run news_pipeline -t dev` | Run full news pipeline |
| **Dev** | `databricks bundle run news_pipeline -t dev --task bronze_ingest` | Run single task |
| **Dev** | `databricks bundle run tick_pipeline -t dev` | Run tick pipeline |
| **Staging** | `gh run list --workflow=deploy.yml` | Check CI/CD status |
| **Staging** | `gh run view <run-id>` | View run details |

### Environment quick-check queries

```sql
-- Which environment am I looking at?
SELECT current_catalog();

-- Latest data in bronze
SELECT max(scraped_at) FROM `trading-signal-gen-dev`.bronze.news_raw;

-- Row count in silver
SELECT count(*) FROM `trading-signal-gen-dev`.silver.news_cleaned;
```

Replace `` `trading-signal-gen-dev` `` with `` `trading-signal-gen-staging` `` or `trading-signal-prod` as needed.

---

## 9. Common Gotchas for New Databricks Developers

**Cluster spin-up time.** Dev job runs take 3-5 minutes just for cluster startup before your code runs. This is normal — don't assume the job is stuck.

**Spot instance preemption.** Workers use spot instances to save cost. If a run fails with "instance lost" or "spot interruption", just re-run it. The driver node is on-demand, so job state is preserved.

**Bundle deploy is idempotent.** Running `databricks bundle deploy -t dev` multiple times is safe. It creates the job on first deploy and updates it on subsequent deploys.

**Checkpoint state.** Auto Loader tracks which S3 files have already been processed via checkpoint files. If you need to reprocess all files in dev, delete the checkpoint directory for that job (dev only — never do this in staging or prod).

**Schema evolution.** Auto Loader uses `addNewColumns` mode. If a new field appears in the NDJSON source, it's automatically added to the bronze table. However, removed fields stay in the schema — they just become null.

**Delta MERGE is safe to re-run.** Silver uses MERGE (upsert) by `story_id`. Running the same data through twice won't create duplicates.

**Same S3 bucket, different catalogs.** All three environments read from the same S3 source data. Isolation is on the write side only (different catalogs). This means staging and prod process the same raw data — which is intentional.

**Never deploy to staging or prod manually.** Always go through the CI/CD pipeline (merge to main → staging auto-deploy → manual prod approval). The only environment you deploy to directly is dev.
