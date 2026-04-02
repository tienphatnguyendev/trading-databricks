# Update Project Spec with Kinesis Tick Streamer Details Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `docs/superpowers/specs/project-spec.md` to reflect the completed `kinesis-tick-streamer` upstream provider, replacing outdated references to the planned `tick-scraper-aws`.

**Architecture:** We will replace all references to `tick-scraper-aws` with the actual implemented `kinesis-tick-streamer` (ECS Fargate, CDK). This includes updating the System Overview diagram, the repository responsibility table, the Tick Pipeline section with the exact NDJSON schema contract, and removing the resolved open question about the tick schema.

**Tech Stack:** Markdown, Git.

---

### Task 1: Update System Overview and Repository References

**Files:**
- Modify: `docs/superpowers/specs/project-spec.md`

- [ ] **Step 1: Update Out-of-Scope list**

Use the `replace` tool to update the Out of scope list.

**Old String:**
```text
**Out of scope:**
- Data collection from external APIs (news-scraper-aws, tick-scraper-aws)
```

**New String:**
```text
**Out of scope:**
- Data collection from external APIs (news-scraper-aws, kinesis-tick-streamer)
```

- [ ] **Step 2: Update the Architecture Diagram**

Use the `replace` tool to update the system overview diagram.

**Old String:**
```text
news-scraper-aws                     tick-scraper-aws
       │                                    │
       ▼                                    ▼
  S3: news/tradingview/...            S3: ticks/binance/...
       │                                    │
```

**New String:**
```text
news-scraper-aws                 kinesis-tick-streamer
       │                                    │
       ▼                                    ▼
  S3: news/tradingview/...            S3: ticks/binance/...
       │                                    │
```

- [ ] **Step 3: Update the Repository Table**

Use the `replace` tool to update the responsibilities and tech stack for the tick streamer.

**Old String:**
```text
| Repo | Responsibility | IaC | Deploys to |
|------|---------------|-----|-----------|
| `news-scraper-aws` | Scrape news → S3 NDJSON | SAM | AWS Lambda |
| `tick-scraper-aws` (planned) | Scrape BTC-USDT ticks → S3 | SAM | AWS Lambda |
| `trading-databricks` (this repo) | Ingest, transform, bars, features | DABs | Databricks |
```

**New String:**
```text
| Repo | Responsibility | IaC | Deploys to |
|------|---------------|-----|-----------|
| `news-scraper-aws` | Scrape news → S3 NDJSON | SAM | AWS Lambda |
| `kinesis-tick-streamer` | Stream BTC & ETH ticks → S3 | AWS CDK | ECS Fargate |
| `trading-databricks` (this repo) | Ingest, transform, bars, features | DABs | Databricks |
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/project-spec.md
git commit -m "docs: update system overview to kinesis-tick-streamer"
```

---

### Task 2: Update Tick Pipeline Schema Contract

**Files:**
- Modify: `docs/superpowers/specs/project-spec.md`

- [ ] **Step 1: Replace the Tick Pipeline introduction and schema**

Use the `replace` tool to update the source path and schema contract based on the latest specification.

**Old String:**
```text
## 6. Tick Pipeline

**Source:** `s3://ai-trading-raw-ap-southeast-1/ticks/binance/symbol={symbol}/year=YYYY/month=MM/day=DD/*.ndjson`
**Schema contract:** defined in `tick-scraper-aws/CLAUDE.md` (not yet built)

**Status:** Bronze and silver are stubs. Bar construction is scaffolded.
```

**New String:**
```text
## 6. Tick Pipeline

**Source:** `s3://ai-trading-raw-ap-southeast-1/ticks/binance/symbol={SYMBOL}/year=YYYY/month=MM/day=DD/*.ndjson`
**Schema contract:**

```json
{
  "timestamp": "2026-03-31T12:00:00.123456Z",
  "price": "67234.56000000",
  "quantity": "0.00150000",
  "side": "buy",
  "trade_id": "123456789",
  "symbol": "BTCUSDT",
  "exchange": "binance",
  "batch_id": "batch_20260331T120000Z_a1b2c3d4"
}
```

**Status:** Bronze and silver are stubs. Bar construction is scaffolded.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/project-spec.md
git commit -m "docs: add exact NDJSON schema contract for tick pipeline"
```

---

### Task 3: Remove Outdated Open Question

**Files:**
- Modify: `docs/superpowers/specs/project-spec.md`

- [ ] **Step 1: Remove the Tick data NDJSON schema row**

Use the `replace` tool to remove the now-resolved open question.

**Old String:**
```text
| NLP library for sentiment/entity extraction | gold news_features implementation | Brainstorm before implementing |
| Tick data NDJSON schema | Bronze ticks table definition | When building tick-scraper-aws |
| Bar construction thresholds (initial N values) | Dollar/volume/imbalance bar parameters | During tick pipeline implementation |
```

**New String:**
```text
| NLP library for sentiment/entity extraction | gold news_features implementation | Brainstorm before implementing |
| Bar construction thresholds (initial N values) | Dollar/volume/imbalance bar parameters | During tick pipeline implementation |
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/project-spec.md
git commit -m "docs: remove resolved open question about tick schema"
```
