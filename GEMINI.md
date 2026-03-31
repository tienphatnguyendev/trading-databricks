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
| Keep each domain isolated (news, ticks, ml, agents) | Add new job/pipeline definitions | Delete production Delta tables |
| Write to Delta via SDP pipelines | Change bar construction thresholds | Push directly to `main` |
| Run all tests before marking done | Add new pyproject.toml dependencies | Write to S3 from this repo |
| Preserve gold layer Delta schemas | Change cluster node types | Mix news and tick domain code |

## Commands

```bash
uv sync --extra dev                              # Install / sync dependencies
uv run pytest                                    # Run full test suite
uv run pytest -x -v                              # Stop on first failure, verbose
uv run pytest tests/news/                        # Run news domain tests only
uv run pytest tests/ticks/                       # Run tick domain tests only
uv run pytest tests/ml/                          # Run ML domain tests only
uv run pytest tests/common/                      # Run shared utilities tests only
databricks bundle validate                       # Validate DABs bundle config
databricks bundle deploy -t dev                  # Deploy to trading-signal-gen-dev catalog
databricks bundle deploy -t staging              # Deploy to trading-signal-gen-staging catalog
databricks bundle run news_pipeline -t dev       # Run news SDP pipeline in dev
databricks bundle run tick_pipeline -t dev       # Run tick SDP pipeline in dev
databricks bundle run ml_pipeline -t dev         # Run ML feature SDP pipeline in dev
```

## Repo Navigation

```
src/trading_databricks/        # Python package — all production code
  news/                        # News domain (news-scraper-aws S3 output)
    pipeline.py                # SDP: @dp.table / @dp.materialized_view wrappers (NOT locally testable)
    bronze.py                  # Pure transforms: enrich_bronze()
    silver.py                  # Pure transforms: cast_timestamps(), normalize_text(), clean_news(), …
    gold.py                    # Pure transforms: sentiment, entities (stubs — TRD-4)
    schemas.py                 # StructType definitions for news NDJSON contract
    expectations.py            # SDP quality rule strings (@dp.expect_or_drop)
  ticks/                       # Tick domain (tick-scraper-aws S3 output)
    pipeline.py                # SDP: @dp.table / @dp.materialized_view wrappers
    bronze.py                  # Pure transforms: enrich_bronze()
    silver.py                  # Pure transforms: cast_types(), compute_dollar_value(), clean_ticks()
    gold/                      # Sub-package: 4 bar algorithms (stubs — TRD-5)
      dollar_bars.py           # build_dollar_bars()
      volume_bars.py           # build_volume_bars()
      tick_imbalance_bars.py   # build_tick_imbalance_bars()
      dollar_imbalance_bars.py # build_dollar_imbalance_bars()
    schemas.py                 # StructType definitions for tick NDJSON + bar schemas
    expectations.py            # SDP quality rule strings
  ml/                          # ML domain (stubs — TBD)
    pipeline.py                # SDP: @dp.materialized_view for feature tables
    features.py                # Pure feature engineering transforms
    jobs/train.py              # Non-SDP: MLflow model training job entry point
  agents/                      # Gen-AI domain (stubs — TRD-6)
    tools.py                   # UC function tools, retrieval logic
    serve.py                   # Agent endpoint configuration
  common/                      # Shared utilities
    config.py                  # @dataclass: DollarBarConfig, VolumeBarConfig, …
    quality.py                 # Shared SDP expectation helper functions
    types.py                   # Shared type aliases
tests/                         # pytest suite — mirrors src/trading_databricks/
  conftest.py                  # DatabricksSession.builder.serverless() fixture
  news/                        # News domain tests
  ticks/                       # Tick domain tests
  ml/                          # ML domain tests
  common/                      # Shared utilities tests
resources/                     # DABs pipeline and job definitions
  news_pipeline.pipeline.yml   # News SDP pipeline resource
  tick_pipeline.pipeline.yml   # Tick SDP pipeline resource
  ml_pipeline.pipeline.yml     # ML feature SDP pipeline resource
databricks.yml                 # DABs bundle config — do not modify targets without approval
docs/superpowers/
  specs/                       # Design docs — read before architectural decisions
  plans/                       # Implementation plans — use most recent for each feature
```

## Working with Specs and Plans

- Before architectural decisions, read `docs/superpowers/specs/project-spec.md` and `docs/superpowers/specs/2026-03-31-sdp-architecture-design.md`.
- Before implementing a feature, check `docs/superpowers/plans/` for the relevant plan.
- Plans are dated (`YYYY-MM-DD-<topic>.md`) — use the most recent one.
- New features: brainstorm first → design doc → implementation plan → execute.

## Coding Standards

Follow `python-programing-rules.md` (Fluent Python idioms). Project-specific:

- **SDP pipelines:** `from pyspark import pipelines as dp`. `pipeline.py` files contain **only** `@dp.table` / `@dp.materialized_view` / `dp.create_auto_cdc_flow` calls. No transformation logic in `pipeline.py`.
- **Pure transforms:** All business logic lives in `bronze.py`, `silver.py`, `gold.py` (or `gold/`). These are locally testable with `pytest`.
- **DataFrame chaining:** Use `DataFrame.transform()` for chaining multiple transformations.
- **Config objects:** `@dataclass(frozen=True)` for all configuration (e.g. `DollarBarConfig`).
- **Keyword-only params:** `*` for all public functions with more than 2 arguments.
- **Bar construction:** Stateful algorithms (Lopez de Prado) operate on ordered ticks; use `@dp.materialized_view` (full recompute) not streaming tables.
- **Serverless compute:** All SDP pipelines use `serverless: true`. No classic cluster config.

## Data Contracts

**Incoming (from S3 — do not change):**
- News: `news/{source}/year=YYYY/month=MM/day=DD/{batch_id}.ndjson` — schema defined in `news-scraper-aws/CLAUDE.md`
- Ticks: `ticks/binance/symbol={symbol}/year=YYYY/month=MM/day=DD/{batch_id}.ndjson` — schema defined in `tick-scraper-aws/CLAUDE.md`

**Outgoing Delta tables (do not rename or remove fields without explicit approval):**

| Table | Key field(s) | SDP construct | Write mode |
|-------|-------------|---------------|-----------|
| `bronze.news_raw` | story_id, scraped_at | Streaming table | Append-only |
| `silver.news_cleaned` | story_id | Streaming table (AUTO CDC SCD1) | Upsert on story_id |
| `gold.news_features` | story_id | Materialized view | Full recompute |
| `bronze.ticks_raw` | trade_id, timestamp | Streaming table | Append-only |
| `silver.ticks_cleaned` | trade_id | Streaming table | Append-only (streaming) |
| `gold.dollar_bars` | bar_index | Materialized view | Full recompute |
| `gold.volume_bars` | bar_index | Materialized view | Full recompute |
| `gold.tick_imbalance_bars` | bar_index | Materialized view | Full recompute |
| `gold.dollar_imbalance_bars` | bar_index | Materialized view | Full recompute |
