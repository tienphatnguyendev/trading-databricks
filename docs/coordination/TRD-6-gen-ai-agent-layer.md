# Parallel Work Coordination — TRD-6: Build Gen-AI Agent Layer

> **Parent issue:** [TRD-6](https://linear.app/aaron-solo/issue/TRD-6/build-gen-ai-agent-layer)
>
> Each agent owns ONE section below. Read the full file at session start.
> Update your section during/after work. Never edit another agent's section.
>
> **Cross-worktree access:** Always read/write from the **main worktree** path, not your worktree.
>
> **Status values:** `not_started` → `in_progress` → `done` (or `blocked:<reason>`)

---

## Section 1: Schema & Config — TRD-23, TRD-24 (Wave 1)
- **Status:** done
- **Branch:** feat/TRD-24-mas-config-dataclasses (merged PR #13), feat/TRD-23-mas-schemas-and-taxonomy (merged PR #14)
- **Modified files:**
  - `src/trading_databricks/agents/config.py` (NEW)
  - `src/trading_databricks/agents/schemas.py` (NEW)
  - `src/trading_databricks/agents/taxonomy.py` (NEW)
  - `scripts/seed_taxonomy.py` (NEW)
  - `tests/agents/__init__.py` (NEW)
  - `tests/agents/test_config.py` (NEW)
  - `tests/agents/test_schemas.py` (NEW)
  - `tests/agents/test_taxonomy.py` (NEW)
- **Exports (public APIs):**
  - `config.py`: `GatekeeperConfig`, `UncertaintyRouterConfig`, `MechanismValidatorConfig`, `MASConfig`
  - `schemas.py`: `CAUSAL_FEATURES_SCHEMA`, `TREATMENT_TAXONOMY_SCHEMA`, `MECHANISM_WHITELIST_SCHEMA`, `DAGNESS_AUDIT_SCHEMA`, `CONFOUNDING_AUDIT_SCHEMA`
  - `taxonomy.py`: `seed_taxonomy()`, `load_taxonomy()`, `load_whitelist()`, `TREATMENT_TAXONOMY_SEED`, `MECHANISM_WHITELIST_SEED`
- **Needs from others:** None (foundation — no dependencies)
- **Notes:** Both merged. Run `uv run python scripts/seed_taxonomy.py` on cluster 0331-043814-ghylcr28 to seed gold tables (34 taxonomy rows, 40 whitelist rows). Wave 2 can begin.

---

## Section 2: Gatekeeper — TRD-25, TRD-26 (Wave 2–3)
- **Status:** done (TRD-25), not_started (TRD-26)
- **Branch:** feat/TRD-25-gatekeeper-stage1-spacy-ner (merged)
- **Modified files:**
  - `src/trading_databricks/agents/gatekeeper.py` (NEW)
  - `tests/agents/test_gatekeeper.py` (NEW)
  - `pyproject.toml` (`spacy==3.8.4` added)
- **Exports (public APIs):**
  - `gatekeeper.py`: `CRYPTO_ENTITY_PATTERNS`, `build_nlp`, `extract_ner_features`, `is_crypto_relevant`
- **Needs from others:** `GatekeeperConfig` from Section 1 ✅
- **Notes:** Stage 1 NER only (ner_only_mode=True, V1). TRD-26 adds Stage 2 pandas UDF +
  sklearn LogReg — blocked on TRD-25 merge. UDF callers must call build_nlp() inside
  the UDF body (spec §5.4); cache is per-executor-process, not serialized.

---

## Section 3: RAG + Extractor — TRD-28, TRD-29 (Wave 2)
- **Status:** in_progress (TRD-28 done, TRD-29 pending)
- **Branch:** main (TRD-28 merged)
- **Modified files:**
  - `src/trading_databricks/agents/nodes/__init__.py` (NEW)
  - `src/trading_databricks/agents/nodes/few_shot_retriever.py` (NEW)
  - `src/trading_databricks/agents/tools.py` (MODIFIED)
  - `scripts/provision_vector_search.py` (NEW)
  - `tests/agents/test_few_shot_retriever.py` (NEW)
- **Exports (public APIs):**
  - `nodes/few_shot_retriever.py`: `few_shot_retriever(state, config)` — LangGraph node
  - `nodes/few_shot_retriever.py`: `STATIC_FEW_SHOT_EXAMPLES` — 5 hardcoded examples
  - `tools.py`: `retrieve_few_shot_chains(headline)` — callable tool
- **Needs from others:** ✅ All Wave 1 exports available
- **Provides to others:**
  - `few_shot_examples` state field for TRD-29 (Extractor)
  - `retrieval_story_ids` for lineage tracking
  - Static examples for TRD-31 (Router) similarity fallback
- **Notes:** VS infrastructure provisioned via `scripts/provision_vector_search.py --env dev`. Cold-start guard triggers at < 50 accepted chains. TRD-29 extractor implementation pending.

---

## Section 4: Validator + Router + Context — TRD-27, TRD-30, TRD-31 (Wave 2–3)
- **Status:** done (TRD-27, TRD-30, TRD-31 complete)
- **Branch:** feat/TRD-30-mechanism-validator, aaronngdev/trd-27-w2-implement-context_retriever-node-returns-null-market (merged PR #17), feat/TRD-31-w3-implement-uncertainty_router-node-module-a-abci-inspired (merged PR #19)
- **Modified files:**
  - `src/trading_databricks/agents/nodes/__init__.py` (MODIFIED)
  - `src/trading_databricks/agents/nodes/mechanism_validator.py` (NEW)
  - `src/trading_databricks/agents/nodes/context_retriever.py` (NEW)
  - `src/trading_databricks/agents/nodes/uncertainty_router.py` (NEW)
  - `src/trading_databricks/agents/state.py` (MODIFIED)
  - `src/trading_databricks/agents/tools.py` (MODIFIED)
  - `tests/agents/test_mechanism_validator.py` (NEW)
  - `tests/agents/test_context_retriever.py` (NEW)
  - `tests/agents/test_state.py` (MODIFIED)
  - `tests/agents/test_uncertainty_router.py` (NEW)
- **Exports (public APIs):**
  - `nodes.validate_extraction`
  - `nodes.context_retriever`
  - `nodes.uncertainty_router`
  - `state.MASState` (TypedDict)
  - `tools.fetch_market_context`
- **Needs from others:** `MechanismValidatorConfig`, `UncertaintyRouterConfig`, `MECHANISM_WHITELIST_SCHEMA`, `load_whitelist()` from Section 1
- **Notes:** TRD-30 complete. TRD-27 (context retriever) is complete as a no-op stub returning `None` until TRD-5. TRD-31 is complete and merged (PR #19). Ready for LangGraph integration in TRD-32.

---

## Section 5: Integration + Deploy + Eval — TRD-32, TRD-33, TRD-34 (Wave 4–6)
- **Status:** not_started
- **Branch:** —
- **Modified files:** —
- **Exports (public APIs):** —
- **Needs from others:** All exports from Sections 1–4
- **Notes:** Strictly sequential. Cannot start until ALL sections above are done.
