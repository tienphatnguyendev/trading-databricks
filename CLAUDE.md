# trading-databricks

Databricks transformation & feature engineering for the algorithmic trading system.

**Tick data flow:** Binance WebSocket → AWS Kinesis → two paths:
- **Fast path:** Kinesis → Databricks streaming (real-time services)
- **Slow path:** Kinesis → S3 → Auto Loader → Databricks (batch experiments & research)

**News data** is ingested from S3 via Auto Loader.

**This repo reads from S3 — it never writes to it.**

**Current stage sample datasets** (must use these for ML & MAS development):
- `trading-signal-gen-dev.bronze.news_agent_eval_dataset`
- `trading-signal-gen-dev.bronze.raw_aggtrades_samples`

## 🔍 First Thing Every Session

Before writing any code, orient yourself. This takes 2 minutes and prevents hours of wrong assumptions:

1. **Databricks MCP** — use `execute_sql` to inspect catalog contents (`` SHOW SCHEMAS IN `trading-signal-gen-dev` ``), check pipeline states, and verify table freshness.
2. **Read the latest spec** — `docs/superpowers/specs/` (most recent date wins).
3. **Check coordination** — `ls docs/coordination/` for your ticket's file. If one exists, **read it fully** before coding — it tells you what other agents have built, what APIs they export, and what files they own. Update your section during/after work. See `docs/coordination/_TEMPLATE.md` for the format.

Trust the tools. Verify state before acting.

## Non-Negotiable Rules

- **Code style:** Read `docs/python-programing-rules.md` **before writing any code** — it is the authority on all Python idioms. Every implementation must conform.
- **Polars, not pandas:** All single-node DataFrame work in the `ml/` domain **must** use Polars (`import polars as pl`). Never use pandas for ML pipeline code. This applies to CUSUM detection, labeling, feature engineering, training data prep, bet sizing, and any other non-Spark transforms. When bridging Spark ↔ Polars, use Arrow: `pl.from_arrow(spark_df.collect().to_arrow())` or `spark.createDataFrame(polars_df.to_arrow())`. The only exception is interfacing with libraries that require pandas (e.g., scikit-learn `.fit()` — convert at the boundary with `.to_pandas()`).
- **Research before implementing:** For any new, uncommon, or unfamiliar framework/technology, use web search or available MCP tools to read the **latest official documentation** before writing code. Never guess at APIs.
- **No secrets in code.** Credentials go in Databricks secrets or Unity Catalog credentials only.
- **Never modify `databricks.yml` targets without explicit user approval.**
- **Gold Delta table schemas are contracts.** Field renames or removals require explicit approval.
- **Tests must pass before claiming work is done.** Run `uv run pytest` and verify.
- **Run `databricks bundle validate` before any deployment.**
- **Local execution:** Databricks Connect configured in `~/.databrickscfg`. Use dev cluster `0331-043814-ghylcr28` for MCP testing.
- **Every feature branch must reference a Linear ticket.**

## Boundaries

| Always | Ask First | Never |
|--------|-----------|-------|
| Follow `python-programing-rules.md` | Modify `databricks.yml` targets | Hardcode secrets |
| Keep domains isolated (news, ticks, ml, agents) | Add new job/pipeline definitions | Delete production Delta tables |
| Write to Delta via SDP pipelines | Change bar construction thresholds | Push directly to `main` |
| Run all tests before marking done | Add new pyproject.toml deps | Write to S3 from this repo |
| Preserve gold layer Delta schemas | Change cluster node types | Mix news and tick domain code |
| Verify Linear ticket before branching | Create ticket for unplanned work | Branch without a Linear ticket |

## Databricks Development Workflow

1. **Pure transforms** — test locally: `uv run pytest`. No cluster needed.
2. **Cluster-dependent code** — write a local script, then execute via Databricks MCP:
   - `run_python_file_on_databricks(file_path="<local_path>", cluster_id="0331-043814-ghylcr28")`
   - Reuse the returned `context_id` in follow-up calls for faster iteration.
3. **SQL inspection** — use `execute_sql` (SQL warehouse, no cluster needed):
   - Verify table contents, schemas, row counts, pipeline states.
4. **SDP pipeline testing** — deploy and trigger via terminal:
   - `databricks bundle validate` → `databricks bundle deploy -t dev` → `databricks bundle run <pipeline_name> -t dev`

**Parallel coordination:** When working alongside other agents, you **must** follow session step 3 above. Read and update `docs/coordination/<TICKET-ID>-<description>.md` from the **main worktree** path (not your worktree). Never edit another agent's section — only update your own with status, modified files, and exported APIs.

## Linear Integration

Project: **Trading Databricks** — ID: `bd63748f-7b85-4aa4-b67f-f26430a65a7f`, team: `data-sci-bro`, prefix: `TRD`.

Use the `linear-server` MCP to look up or create a ticket before branching.

### Branch naming (CRITICAL FOR CI)

```
type/TICKET-ID-short-description
```
- `type`: `feat` | `fix` | `chore` | `docs` | `test` | `refactor`
- `TICKET-ID`: uppercase (e.g., `TRD-1`, `SOLO-175`)
- Examples: `feat/SOLO-175-v2-mastery-brief-pipeline`, `fix/TRD-2-tick-timestamp-cast`

### Commit format

```
TRD-NNN type: short imperative description
```

### Workflow order

`brainstorming → writing-plans → [Linear ticket] → using-git-worktrees → subagent-driven-development → finishing-a-development-branch` — one ticket per branch, never mix IDs.

## S3 Data Contracts (incoming — do not change)

- News: `s3://trading-signal-gen-data/news/{source}/year=YYYY/month=MM/day=DD/{batch_id}.ndjson`
- Ticks: `s3://trading-signal-gen-data/ticks/binance/symbol={symbol}/year=YYYY/month=MM/day=DD/{batch_id}.ndjson`

Schemas defined in the upstream repos (`news-scraper-aws`, `tick-scraper-aws`).

## Architecture Decisions (not in code — must know)

- **SDP pattern:** `pipeline.py` = only `@dp.table` / `@dp.materialized_view` wrappers. Zero logic. All transforms in `bronze.py`, `silver.py`, `gold.py`.
- **Bar construction:** Lopez de Prado stateful algorithms → `@dp.materialized_view` (full recompute), not streaming.
- **Serverless:** All SDP pipelines use `serverless: true`. No classic cluster config.
- **Specs & plans:** Read `docs/superpowers/specs/` before architectural decisions. Check `docs/superpowers/plans/` before implementing. Plans are dated — use most recent.
