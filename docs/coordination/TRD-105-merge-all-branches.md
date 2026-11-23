# Parallel Work Coordination — TRD-105: Merge All Branches

## Integration — root
- **Status:** verified; pull-request merge pending
- **Branch:** chore/TRD-105-merge-all-branches
- **Modified files:** Integration merge resolutions and this coordination record.
- **Exports (public APIs):** Preserve current main contracts while integrating outstanding branch history.
- **Needs from others:** Read-only audits of eight outstanding remote heads.
- **Notes:** Source main is 946839f. Preserve original untracked files. Merge through a pull request; no direct push to main and no Databricks deployment. Databricks MCP authentication failed during orientation.


- **Validation:** `CI=1 UV_PROJECT_ENVIRONMENT=/Users/aaronng/repos/trading-databricks/.venv uv run pytest -o addopts='' -q`: 362 passed, 43 skipped (Databricks-dependent tests use the existing shared CI skip). All 76 original branch refs are ancestors of the integration head. All five gold schema definitions are unchanged; `databricks.yml`, dependency files, production ML/ticks code, and CI settings are unchanged. Bundle validation could not authenticate because the saved OAuth refresh token expired.
- **Resolution:** Preserved current workflow, Spark fixture, tick schema/bar behavior, and tick pipeline when historical branches conflicted. Retained the separate trades research pipeline and both inspection queries. Removed generated `build/lib` copies, repaired agent helper compatibility and pipeline ownership, and changed research defaults to the approved sample tables.
- **Integrated remote heads:** `Khoi_EDA`; `feat/TRD-32-mas-integration`; `feat/TRD-321-enhance-qa-notebook-11563299545450088678`; `feat/TRD-5-implement-volume-bars-sampling-251850208203927795`; `feat/build-mas-ml-research-notebooks`; `henrypham-dev`; `perf/compute-news-join-where-14986401307351773553`; `refactor/TRD-news-optimization-2114600337191037903`; `test/mechanism-validator-fallback-4477052559269222859`.
- **Inventory correction:** The first audit counted eight outstanding branches; the complete inventory contains nine distinct remote heads (the local TRD-32 ref duplicates one). All nine are integrated.
- **Limits:** No deployment or live cluster validation. Imported research Spark threshold bars still need algorithm verification against the Polars baseline before production use; historical maintenance/backfill notebooks are not executed by this integration.

## Agent compatibility — audit_research
- **Status:** done
- **Modified files:** `src/trading_databricks/agents/gold.py`, `src/trading_databricks/agents/nodes/rag_retriever.py` (integration worktree).
- **Exports (public APIs):** Restore `_make_gatekeeper_udf(config)` and `_query_firehose(query_text, *, index_name, num_results, timeout_s)`; preserve gold history and add `news_firehose_transform(df)` for the pipeline wrapper.
- **Notes:** Restored legacy helper contracts, kept SQL gatekeeping/structured ai_query/chunk retrieval additions, removed seven-day causal history truncation, and restored explicit unavailable/no-match outcomes when market joins fail. Fixed pandas UDF type resolution by making pandas available at module scope. Existing gold tests: 3 passed; existing RAG, graph, and causality sub-pipeline tests: 18 passed. No existing tests changed. Full integration suite remains root-owned; no dependency changes or commits.

## Research test compatibility — audit_mas
- **Status:** done
- **Modified files:** `notebooks/btcusdt_utils/__init__.py`, `tests/test_bars_spark.py` (integration worktree).
- **Exports (public APIs):** Preserve research root exports and sampling aliases with lazy imports; Spark bar tests reuse the existing shared Databricks session fixture.
- **Notes:** Reproduced collection failure from eager `quality` import requiring undeclared `statsmodels`. Root exports now load dependencies only when requested and preserve real dependency errors. Added a subprocess regression test that constructs a real Polars bar with `statsmodels` unavailable. Removed the unsupported local Spark fixture. Verification: owned target with `CI=1` passed 2 local tests and skipped 18 live Spark tests through the unchanged shared fixture; `git diff --check` passed. No dependencies, bar algorithms, or stationarity implementation changed.

## Independent review — audit_pipelines
- **Status:** done
- **Modified files:** `src/trading_databricks/agents/pipeline.py`, new `src/trading_databricks/agents/silver.py`, `resources/agents_pipeline.pipeline.yml` (integration worktree).
- **Exports (public APIs):** `news_chunks_transform(cleaned: DataFrame) -> DataFrame`; thin wrappers call this and `news_firehose_transform` from the gold owner.
- **Notes:** Moved existing chunk UDF unchanged into a pure transform, removed duplicate ownership of `news_cleaned`, retained SQL gatekeeper mode. Official Databricks bundle substitutions documentation explicitly supports `${workspace.file_path}/src` in pipeline configuration. Verified original/extracted transform AST equivalence, seven original/new/serialized UDF cases, three thin wrappers, YAML source path, and `git diff --check`; existing gold tests pass (3/3) using the main virtual environment. No new dependencies/jobs, target edits, commits, or deployments. Earlier read-only audit recommends retaining current main behavior for stale volume/news branches.
