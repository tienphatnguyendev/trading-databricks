# S3 Topology Refactor: Dedicated Raw Bucket (Infra-Owned)

**Status:** Accepted
**Date:** 2026-03-30
**Decision:** Dedicated Bucket Per Project — raw bucket only, owned by `ai-trading-infra` stack

---

## 1. Bucket Naming Convention

### Pattern

```
{org}-{project}-{layer}-{region}
```

### Segment Definitions

| Segment | Value | Purpose |
|---------|-------|---------|
| `{org}` | `ai` | Organization/owner namespace. Prevents global S3 name collisions. Immutable. |
| `{project}` | `trading` | Logical project boundary. Maps 1:1 to a codebase or domain. IAM policies, cost allocation tags, and CloudWatch alarms scope to this. |
| `{layer}` | `raw` | Medallion-equivalent processing stage. Silver/Gold use Databricks Unity Catalog managed storage (`databricks-storage-*` buckets) — no separate curated/enriched S3 buckets needed. |
| `{region}` | `ap-southeast-1` | AWS region. Required for S3 global uniqueness. Must co-locate with compute (Databricks workspace, Lambda). |

### Concrete Bucket Name

```
ai-trading-raw-ap-southeast-1     # Landing zone — scrapers write, Databricks Auto Loader reads
```

> **Curated and enriched buckets are deferred (YAGNI).** Unity Catalog managed storage handles Silver/Gold layers. Dedicated curated/enriched S3 buckets will be created only if a non-Databricks consumer requires them.

### Infrastructure Ownership

The raw bucket is a **platform resource**, not an application resource. It is owned by the `ai-trading-infra` CloudFormation/SAM stack (separate repo), not by `news-scraper-aws` or any other application stack.

| Resource | Owner | Why |
|----------|-------|-----|
| `ai-trading-raw-ap-southeast-1` | `ai-trading-infra` | Platform resource. Multiple producers (news-scraper, tick-scraper) and consumers (Databricks) share it. Two SAM stacks cannot co-own the same bucket. |
| Lambda (TradingViewScraper) | `news-scraper-aws` | Application resource. Scoped to news collection. |

Application stacks reference the bucket by its deterministic name — no CloudFormation exports needed.

### Layer Mapping

| S3 Layer Name | Medallion Equivalent | Format on S3 | Written By | Read By |
|---------------|---------------------|--------------|------------|---------|
| `raw` | Bronze landing zone | NDJSON | Lambda scrapers | Databricks Auto Loader |
| `curated` | Silver | Delta (Parquet + `_delta_log/`) — **deferred** | — | — |
| `enriched` | Gold | Delta (Parquet + `_delta_log/`) — **deferred** | — | — |

### Why Layer in the Bucket Name (Not a Prefix)

| Concern | Layer-as-prefix (1 bucket) | Layer-as-bucket (3 buckets) |
|---------|---------------------------|----------------------------|
| **IAM** | Prefix conditions on `s3:ListBucket` are clunky and error-prone. Lambda needs `PutObject` on `raw/*` only, but `ListBucket` is bucket-scoped. | Hard boundary. Lambda gets `PutObject` on the raw bucket. Databricks gets `GetObject` on raw, `ReadWrite` on curated/enriched. No condition hacks. |
| **Lifecycle** | Single set of rules with prefix filters. Adding a new layer requires editing existing rules. | Independent lifecycle per bucket. Raw: IA at 30d, expire at 365d. |
| **Encryption** | One encryption config for all layers. | Raw: SSE-S3 (sufficient). |
| **Cost attribution** | S3 billing is per-bucket. Prefix-level cost requires Storage Lens custom dashboards. | Native per-bucket billing in AWS Cost Explorer. |

**Verdict:** Layer-as-bucket. IAM and lifecycle separation alone justify the overhead.

### Why No `{env}` in the Bucket Name

| Layer | Env isolation needed? | Reason |
|-------|----------------------|--------|
| `raw` | **No** | Scrapers produce one copy of data. All Databricks environments (dev/staging/prod) read the same raw files. Duplicating raw data per env wastes storage and complicates scraper deployment. |
| `curated` | **No** | Unity Catalog provides env isolation via catalogs (`dev_trading.silver.*` vs `prod_trading.silver.*`). Each catalog maps to a separate managed storage root. Databricks manages this. |
| `enriched` | **No** | Same as curated. Unity Catalog handles it. |

Environment isolation lives in Databricks Unity Catalog (catalog-per-env), not in S3 bucket-per-env.

---

## 2. Directory Tree

### Raw Bucket (Bronze Landing Zone)

```
s3://ai-trading-raw-ap-southeast-1/
│
├── news/
│   └── tradingview/
│       ├── year=2026/
│       │   └── month=03/
│       │       ├── day=28/
│       │       │   ├── batch_20260328T120000Z_a1b2c3d4.ndjson
│       │       │   ├── batch_20260328T121500Z_e5f6a7b8.ndjson
│       │       │   └── batch_20260328T123000Z_c9d0e1f2.ndjson
│       │       └── day=29/
│       │           └── batch_20260329T000000Z_1a2b3c4d.ndjson
│       └── year=2027/
│           └── ...
│
├── ohlcv/                  # future — tick-scraper
│   └── tradingview/
│       └── symbol=BTCUSDT/
│           └── year=2026/
│               └── ...
│
└── ticks/                  # future — tick-scraper
    └── binance/
        └── symbol=BTCUSDT/
            └── year=2026/
                └── ...
```

### Key Structural Decisions

| Decision | Rationale |
|----------|-----------|
| `symbol=` partition only for tick/OHLCV tables | News is not per-symbol. Adding an empty partition dimension wastes prefix listings. |
| No nested schema directory | Domain (`news/`, `ohlcv/`, `ticks/`) replaces layer prefix. The bucket name IS the layer. |

---

## 3. Path Convention Contract

### Raw (Bronze Landing Zone)

```
s3://ai-trading-raw-{region}/{domain}/{source}/[symbol={symbol}/]year={YYYY}/month={MM}/day={DD}/{batch_id}.ndjson
```

### Segment Specification

| Segment | Allowed Values | Owner | Mutability |
|---------|---------------|-------|------------|
| `{domain}` | `news`, `ohlcv`, `ticks` | Data architect | Add-only. Adding a new domain requires architecture review. Never rename. |
| `{source}` | `tradingview`, `binance`, `reuters`, ... | Ingest pipeline (scraper repo) | Add-only per domain. Each source gets its own scraper repo. |
| `symbol={symbol}` | `BTCUSDT`, `ETHUSDT`, ... Uppercase, no delimiters. | Ingest pipeline | Present only for symbol-scoped data. Omitted for news. Hive-style key=value. |
| `year={YYYY}` | 4-digit year | Ingest pipeline | Hive-style. UTC-based. |
| `month={MM}` | 01–12, zero-padded | Ingest pipeline | Hive-style. UTC-based. |
| `day={DD}` | 01–31, zero-padded | Ingest pipeline | Hive-style. UTC-based. |
| `{batch_id}` | `batch_{YYYYMMDDTHHMMSSZ}_{8-char-hex}` | Ingest pipeline | One file per Lambda invocation. |

### Ownership Boundaries

```
                    ┌─────────────────────────────────────────────────────┐
                    │                     S3 Path                         │
                    ├──────────┬──────────┬────────────┬──────────────────┤
                    │  domain  │  source  │ partitions │    filename      │
                    ├──────────┼──────────┼────────────┼──────────────────┤
  Raw bucket:       │ Architect│ Scraper  │  Scraper   │    Scraper       │
                    └──────────┴──────────┴────────────┴──────────────────┘
```

---

## 4. Standards Compliance

### AWS Prescriptive Guidance for Data Lakes

| Guidance | This Design | Status |
|----------|-------------|--------|
| Separate buckets for raw, curated, enriched layers | Raw bucket active; curated/enriched deferred to Unity Catalog managed storage | **PASS** |
| Bucket name: `{org}-{purpose}-{account}-{region}` | `ai-trading-{layer}-{region}`. Account ID omitted (deterministic single-account deployment). | **PASS** |
| Hive-style partitioning (`key=value/`) | `year=YYYY/month=MM/day=DD/` | **PASS** |
| Raw layer: immutable, append-only | NDJSON files are never modified or deleted (lifecycle expiry only) | **PASS** |
| Landing zone uses `raw/` terminology, not `bronze/` | Bucket named `raw`, domain prefix is `news/` | **PASS** |
| No secrets, credentials, or PII in S3 paths | Paths contain only structural metadata | **PASS** |

### Databricks Medallion Architecture

| Principle | This Design | Status |
|-----------|-------------|--------|
| Bronze = first ingestion layer | `raw` bucket is pre-bronze (NDJSON files). Actual bronze is the `bronze.news_raw` Delta table in Unity Catalog. No naming collision. | **PASS** |
| Silver/Gold = Databricks-managed | Unity Catalog managed storage handles Silver/Gold. No curated/enriched S3 buckets yet. | **PASS** |
| Unity Catalog governs table access | External/managed tables in Unity Catalog. Catalog-per-env isolation (dev/staging/prod). | **PASS** |

### Apache Hive Partition Spec

| Requirement | This Design | Status |
|-------------|-------------|--------|
| Partition columns as `key=value/` directory names | `year=2026/month=03/day=28/`, `symbol=BTCUSDT/` | **PASS** |
| Consistent partition column ordering | `[symbol/]year/month/day` — symbol (coarse) before date (fine) | **PASS** |
| Partition discovery via Auto Loader | Hive-style paths enable automatic partition discovery | **PASS** |

**Overall: Full compliance with all standards.**

---

## 5. Migration

### Repos Requiring Changes

| Repo | Change | Scope |
|------|--------|-------|
| `news-scraper-aws` | ~~Remove bucket resource, update `S3_BUCKET`/`S3_PREFIX` env vars, Hive-style date format~~ **DONE** (merged via `feat/s3-topology-refactor`) | `s3_writer.py`, `template.yaml`, `handler.py`, tests |
| `ai-trading-infra` (new) | Create SAM stack that owns `ai-trading-raw-ap-southeast-1` | `template.yaml` |
| `trading-databricks` | Update Auto Loader source path to new bucket/prefix convention | `databricks.yml`, `src/news/bronze/ingest_news.py` |
| `tick-scraper-aws` (future) | Build with new convention from the start | N/A — not yet built |

### Migration Strategy: Fresh Start

No existing data migration required. Databricks ingestion of the old raw bucket does not yet exist. Old bucket (`news-scraper-bronze-{account}-{region}`) is decommissioned after the new stack deploys.

| Step | Action | Owner |
|------|--------|-------|
| 1 | Deploy `ai-trading-infra` stack → `ai-trading-raw-ap-southeast-1` created | Infra |
| 2 | Deploy updated `news-scraper-aws` → Lambda writes to new bucket/prefix | App |
| 3 | Verify NDJSON lands at `news/tradingview/year=.../month=.../day=.../` | App |
| 4 | Delete contents of old bucket, then delete bucket | Infra |

Old bucket data can be deleted immediately — no Databricks job has ingested it yet.
