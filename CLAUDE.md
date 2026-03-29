# trading-databricks

This is the **Databricks transformation and feature engineering platform** for the algorithmic trading system. It ingests financial news and BTC-USDT tick data from S3 (written by `news-scraper-aws` and `tick-scraper-aws`), transforms to medallion layers (bronze → silver → gold), constructs alternative bars, and serves features for ML models and gen-AI agent pipelines.

**This repo reads from S3 — it never writes to it.**

## Non-Negotiable Rules

- **Code style:** Follow `docs/python-programing-rules.md`. It is the authority on all Python idioms.
- **No secrets in code:** Credentials go in Databricks secrets or Unity Catalog credentials only.
- **Never modify `databricks.yml` targets without explicit user approval.**
- **Gold Delta table schemas are contracts.** Field renames or removals require explicit approval.
- **Tests must pass before claiming work is done.** Run `uv run pytest` and verify output.
- **Run `databricks bundle validate` before any deployment.**

## Boundaries

| Always | Ask First | Never |
|--------|-----------|-------|
| Follow `python-programing-rules.md` | Modify `databricks.yml` targets | Hardcode secrets or credentials |
| Keep news and tick domains separate | Add new job definitions | Delete production Delta tables |
| Write to Delta via DABs jobs | Change cluster node types | Push directly to `main` |
| Run all tests before marking done | Add new pyproject.toml dependencies | Write to S3 from this repo |
| Preserve gold layer Delta schemas | Change bar construction thresholds | Mix news and tick domain code |

## Commands

```bash
uv sync --extra dev                              # Install / sync dependencies
uv run pytest                                    # Run full test suite
uv run pytest -x -v                              # Stop on first failure, verbose
uv run pytest tests/news/                        # Run news domain tests only
uv run pytest tests/ticks/                       # Run tick domain tests only
databricks bundle validate                       # Validate DABs bundle config
databricks bundle deploy -t dev                  # Deploy to dev_trading catalog
databricks bundle deploy -t staging              # Deploy to staging_trading catalog
databricks bundle run news_pipeline -t dev       # Run news pipeline job in dev
databricks bundle run tick_pipeline -t dev       # Run tick pipeline job in dev
```

## Repo Navigation

```
src/                          # PySpark job source code
  news/                       # News domain (from news-scraper-aws S3 output)
    bronze/ingest_news.py     # Auto Loader: S3 NDJSON → bronze.news_raw
    silver/clean_news.py      # Dedup, cast, normalize → silver.news_cleaned
    gold/news_features.py     # Sentiment, entities → gold.news_features
  ticks/                      # Tick domain (from tick-scraper-aws S3 output)
    bronze/ingest_ticks.py    # Auto Loader: S3 NDJSON → bronze.ticks_raw
    silver/clean_ticks.py     # Validate, cast → silver.ticks_cleaned
    gold/bar_construction.py  # Dollar/volume/imbalance bars → gold.*_bars
tests/                        # pytest suite — mirrors src/ structure
  conftest.py                 # SparkSession fixture (session-scoped)
resources/                    # DABs job and cluster definitions
databricks.yml                # DABs bundle config — do not modify targets without approval
docs/superpowers/
  specs/                      # Design docs — read before architectural decisions
  plans/                      # Implementation plans — use most recent for each feature
```

## Working with Specs and Plans

- Before architectural decisions, read `docs/superpowers/specs/project-spec.md`.
- Before implementing a feature, check `docs/superpowers/plans/` for the relevant plan.
- Plans are dated (`YYYY-MM-DD-<topic>.md`) — use the most recent one.
- New features: brainstorm first → design doc → implementation plan → execute.

## Coding Standards

Follow `python-programing-rules.md` (Fluent Python idioms). Project-specific:

- PySpark `DataFrame.transform()` for chaining transformations
- Separate pure transformation functions from job entry points (`run_*`)
- Bar construction operates on pandas partitions via `.toPandas()` for stateful logic
- `@dataclass` for configuration objects (e.g. `ImbalanceBarConfig`)
- Keyword-only params (`*`) for all public functions with more than 2 arguments

## Data Contracts

**Incoming (from S3 — do not change):**
- News: `bronze/{source}/YYYY/MM/DD/{batch_id}.ndjson` — schema defined in `news-scraper-aws/CLAUDE.md`
- Ticks: `bronze/ticks/{symbol}/YYYY/MM/DD/{batch_id}.ndjson` — schema defined in `tick-scraper-aws/CLAUDE.md`

**Outgoing Delta tables (do not rename or remove fields without explicit approval):**

| Table | Key field(s) | Write mode |
|-------|-------------|-----------|
| `bronze.news_raw` | story_id, scraped_at | Append-only |
| `silver.news_cleaned` | story_id | MERGE |
| `gold.news_features` | story_id | Overwrite |
| `bronze.ticks_raw` | trade_id, timestamp | Append-only |
| `silver.ticks_cleaned` | trade_id | Overwrite |
| `gold.dollar_bars` | bar_index | Overwrite |
| `gold.volume_bars` | bar_index | Overwrite |
| `gold.tick_imbalance_bars` | bar_index | Overwrite |
| `gold.dollar_imbalance_bars` | bar_index | Overwrite |
