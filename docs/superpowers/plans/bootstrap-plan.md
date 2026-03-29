# Bootstrap Plan

**Status:** Complete
**Date:** 2026-03-29
**Design doc:** `docs/superpowers/specs/2026-03-29-trading-databricks-design.md` (in news-scraper-aws)

## What Was Created

| File / Directory | Purpose |
|-----------------|---------|
| `CLAUDE.md` | Agent behavioral rules, boundaries, commands, repo navigation |
| `docs/superpowers/specs/project-spec.md` | Architecture, schema contracts, conventions |
| `databricks.yml` | DABs bundle (dev/staging/prod catalog targets) |
| `resources/news_jobs.yml` | News pipeline job definitions (bronze → silver → gold) |
| `resources/tick_jobs.yml` | Tick pipeline job definitions (stub, schedule commented out) |
| `src/news/` | News domain source stubs (bronze, silver, gold) |
| `src/ticks/` | Tick domain source stubs (bronze, silver, gold + bar construction) |
| `tests/` | Test stubs with planned test descriptions |
| `.github/workflows/deploy.yml` | CI/CD: validate → test → staging → prod |

## How to Start a New Feature

1. Brainstorm with Claude Code → approve design
2. Claude saves design doc to `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`
3. Claude writes implementation plan to `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`
4. Execute plan task by task, run `uv run pytest` after each task
5. Commit after each task

## Pending Implementation (in priority order)

- [ ] `src/news/bronze/ingest_news.py` — Auto Loader streaming job
- [ ] `src/news/silver/clean_news.py` — dedup, cast, normalize, MERGE
- [ ] `src/news/gold/news_features.py` — NLP features (brainstorm NLP stack first)
- [ ] `src/ticks/` — entire tick pipeline (after tick-scraper-aws is built)
- [ ] Statistics dashboard (brainstorm tool selection first)
- [ ] Gen-AI agent access layer (brainstorm framework first)
