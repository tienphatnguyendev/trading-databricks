# MAS Agent Evaluation Harness — Design

**Date:** 2026-04-17
**Status:** Awaiting Review
**Owner:** aaronng
**Related:** `2026-04-14-llm-judge-validator-design.md` (supersedes §3.2 prompt format; see Addendum at end)

## 1. Prompt management — YAML + MLflow Prompt Registry (hybrid)

All MAS prompts — causality extractor variants AND the 3 production judges — move out of `src/trading_databricks/agents/prompts.py` Python string constants into a versioned hybrid:

- **Source of truth in repo:** YAML files under `src/trading_databricks/agents/prompts/`, one per prompt.
- **Runtime source:** MLflow Prompt Registry. Runtime code calls `load_prompt(name, version="production")`; YAML is never read at runtime.
- **Promotion:** `_sync.py` CLI reads each YAML, calls `mlflow.register_prompt()`, and sets stage. YAML commit to `main` is the production promotion gate.
- **Round-trip with GEPA:** `optimize_prompts()` writes new `stage=candidate` versions to the Registry. Accepted candidates are dumped back to YAML (`_sync.py --pull`) and committed.

### 1.1 YAML schema

```yaml
name: causality_extractor_base
description: Full prompt for standard and careful extraction paths.
variables: [taxonomy_injection, whitelist_injection, few_shot_injection]
messages:
  - role: system
    content: |
      You are a financial news causality extractor for BTC/ETH markets.
      ...
metadata:
  used_by: [causality_extractor]
  paths: [standard, careful]
```

### 1.2 Module layout

```
src/trading_databricks/agents/prompts/
├── __init__.py                     # load_prompt(name, version) → list[dict]
├── causality_extractor_base.yaml   # standard + careful paths
├── causality_extractor_fast.yaml
├── judge_direction.yaml
├── judge_mechanism.yaml
├── judge_grounding.yaml
├── _formatters.py                  # lifted from prompts.py
└── _sync.py                        # CLI: YAML ↔ mlflow.register_prompt()
```

Existing format helpers (`format_compact_taxonomy`, `format_full_taxonomy`, `format_whitelist`, `format_few_shot`) move to `_formatters.py`. `build_system_prompt(path)` becomes a thin wrapper that loads the right prompt from Registry and applies formatters.

### 1.3 Why not pure Registry, or pure ChatPromptTemplate?

- Pure Registry = no source control, hard to diff in PRs, no offline dev.
- `ChatPromptTemplate` (proposed in old spec §3.2) = drags LangChain runnable surface into a codebase that only uses LangGraph topology + raw `mlflow.deployments.predict()`. YAML keeps the prompt format framework-agnostic.
- Hybrid = PR-reviewable, versioned, GEPA-compatible, and framework-free.

## 2. Primary evaluation dataset + labeled calibration slices

### 2.1 Primary dataset — `smart-trading-dev.04_eval.mas_eval_v1`

- **Source:** `smart-trading-dev.01_bronze.news_agent_eval_dataset` (500 rows, unlabeled).
- **Managed via:** `mlflow.genai.datasets.create_dataset()`, UC-backed.
- **Stratification:** composite `source × token` with proportional allocation. Cells with fewer than 5 rows fall back to marginal sampling on the smaller-population axis. Stratification dims captured as record `tags`.
- **Record schema:** `{inputs: {headline, body, symbol}, expectations: {}, tags: {source, token, corpus: "primary"}}`.
- **Expectations left empty** so reference-based scorers auto-skip; populated later via labeling session.

### 2.2 Labeled calibration slices (all UC-managed MLflow datasets)

| Dataset | Corpus | Labels | Scorer targets |
|---|---|---|---|
| `mas_eval_direction_cryptolin_v1` | CryptoLin (2,683) + FinRL_BTC (279) | `direction_label` (up/down/neutral) | direction judge calibration |
| `mas_eval_grounding_fincausal_v1` | FinCausal 2022/2025 | `causal_span_present` (pos/neg pairs) | grounding judge calibration |
| `mas_eval_grounding_cnc_v1` | Causal News Corpus (3,767) | `causal_span_present` (cross-genre) | grounding judge generalization |
| `mas_eval_mechanism_bootstrap_v1` | Option-B bootstrap (200 articles, Claude Opus 4.6 weak labels) | `mechanism_label` | mechanism judge calibration |

All four slices register in the same MLflow experiment as the primary set. One experiment calibrates thresholds from §5.3 of the 2026-04-14 spec AND runs the unlabeled MAS eval.

### 2.3 Dataset loaders — `eval/dataset.py`

One public builder `build_eval_dataset(corpus: Literal["primary","cryptolin","fincausal","cnc","mechanism_bootstrap"], version=...)` returns a pandas DataFrame ready for `mlflow.genai.evaluate(data=...)`. Internals use Polars; `.to_pandas()` at the MLflow boundary only.

## 3. Scorer catalog — 3 tiers

`eval/scorers.py` registers all scorers at import. Three tiers, all reusable across `predict_target` values.

### 3.1 Tier 1 — Deterministic, reference-free

- `schema_valid` — extraction conforms to Pydantic `CausalChain` schema.
- `treatment_in_taxonomy` — `treatment_class` ∈ 34-class taxonomy.
- `mechanism_in_whitelist` — `(L1, mechanism)` pair ∈ whitelist for extracted L1.
- `evidence_spans_verbatim` — every evidence span appears verbatim in `body`.
- `validator_accepted` — `state.validation_status == "accepted"`.
- `extraction_latency_ms`, `tokens_in`, `tokens_out` — pulled from trace spans.

### 3.2 Tier 2 — LLM judges (reference-free, MemAlign-ready)

Three `mlflow.genai.judges.make_judge()` instances — **same rubrics, prompts, and model as the production judges in the 2026-04-14 spec**:

- `judge_direction` — categorical (up/down/neutral).
- `judge_mechanism` — categorical (whitelist entry).
- `judge_grounding` — float [0.0, 1.0].

All three:
- Load instructions from Prompt Registry (§1).
- Model = Gemini 3.1 Pro via Databricks external-model endpoint (fallback Claude Sonnet 4).
- Registered at module import with idempotency guard (§6.3).
- Judge `name` matches the future SME label-schema name — required by MemAlign `align()`.

### 3.3 Tier 3 — Reference-based (only on labeled slices)

- `direction_match` — extracted direction == `expectations.direction_label`.
- `mechanism_match` — extracted mechanism == `expectations.mechanism_label`.
- `grounding_alignment` — judge score agreement with `expectations.causal_span_present`.

`run_evaluation()` auto-filters these out when the dataset has empty expectations.

### 3.4 Public surface

`eval/__init__.py` exports:
```python
ALL_SCORERS, DETERMINISTIC_SCORERS, JUDGE_SCORERS, REFERENCE_SCORERS
```

## 4. Unified runner API — `eval/runner.py`

```python
def run_evaluation(
    *,
    dataset_name: str,
    dataset_version: str | int = "latest",
    scorers: list[str] | Literal["all","deterministic","judges","reference"] = "all",
    predict_target: Literal["e2e","causality_extractor","uncertainty_router","mechanism_validator"] = "e2e",
    mas_config_overrides: dict | None = None,
    experiment_id: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> mlflow.models.EvaluationResult:
```

### 4.1 `predict_target` dispatch

- `e2e` — builds full graph via `build_mas_graph(config=...)`, invokes on `{headline, body, symbol}`.
- `causality_extractor` / `uncertainty_router` / `mechanism_validator` — builds a partial state with the upstream node outputs pre-computed (cached from a prior e2e run or computed inline), runs only the target node.

### 4.2 Auto-behaviors

- Auto-loads `MASConfig` from repo default, applies `mas_config_overrides`.
- Auto-tags run with `prompt_versions`, `mas_config_hash`, `git_sha`, `git_dirty`, `dataset_name@version`, `predict_target`, `run_kind`.
- Auto-filters scorers: drops reference-based when `expectations` is empty across the dataset.
- Drives `mlflow.genai.evaluate(data=df, predict_fn=wrapper, scorers=resolved_scorers)`.

### 4.3 Notebook drivers

`notebooks/qa/mas/e2e_mas_evaluation.py` — thin (~30 lines): imports `run_evaluation`, sets params, calls it.

## 5. Per-node prompt tuning notebooks

Under `notebooks/qa/mas/tuning/`:

- `causality_extractor_tuning.py`
- `judge_direction_tuning.py`
- `judge_mechanism_tuning.py`
- `judge_grounding_tuning.py`

Each notebook:
1. Loads current YAML prompt.
2. User edits prompt inline (scratch variable).
3. Calls `mlflow.register_prompt(name=..., template=..., stage="candidate")` — **never writes `production`**.
4. Calls `run_evaluation(predict_target=<target>, ...)` with `tuning_hypothesis` tag.
5. Prints diff vs. baseline run.

**Promotion gate:** user copies accepted candidate back into YAML, commits, PR merges to `main`, CI runs `_sync.py` to push to Registry at `stage=production`. No notebook ever promotes directly.

## 6. Experiment layout, run naming, MemAlign readiness

### 6.1 Experiment — one experiment, faceted by tags

Single experiment `/Shared/mas-evaluation` (overridable via `MLFLOW_EXPERIMENT_ID`). Rationale: scorers, judges, and datasets live in the experiment's registry; splitting per node fragments Registry lookups and breaks MemAlign (judges must co-locate with scored runs).

Faceting via tags:
- `predict_target`, `dataset_name`, `dataset_version`
- `prompt_versions` — JSON `{causality_extractor_base: "3", judge_direction: "2", ...}`
- `mas_config_hash` — sha256[:12] of serialized `MASConfig`
- `git_sha`, `git_dirty`
- `tuning_hypothesis` — free-text from tuning notebooks
- `run_kind` — `baseline` | `tuning` | `calibration` | `regression`

### 6.2 Run naming

`{run_kind}__{predict_target}__{dataset_name}@{dataset_version}__{git_sha7}`

Tuning notebooks append `__{hypothesis_slug}`.

### 6.3 Judge registration (MemAlign-ready)

```python
def _register_if_absent(scorer, experiment_id):
    existing = {s.name for s in mlflow.genai.scorers.list_scorers(experiment_id=experiment_id)}
    if scorer.name not in existing:
        scorer.register(experiment_id=experiment_id)
    return scorer
```

**Contract for `align()`:** judge `name` (`judge_direction`, `judge_mechanism`, `judge_grounding`) MUST equal the label-schema name in any future SME labeling session.

### 6.4 Dataset records scaffolded for labeling

Every record carries an `expectations` slot — empty for unlabeled primary, populated for calibration slices. Labeling sessions can populate without schema migration. Reference-based scorers (§3.3) auto-skip missing expectations.

### 6.5 Labeling stub

`eval/labeling.py` — empty placeholder with docstring pointing at Workflow 7. Actual labeling session creation deferred until SME is scheduled.

---

## Addendum — Reconciliation with 2026-04-14 LLM Judge Validator spec

The 2026-04-14 spec stands as the production design for the 3 LLM judges running as a LangGraph node after `mechanism_validator`. This harness does NOT change that production design. Only the following are updated:

- **§3.2 Prompt format** — Python modules exporting `SYSTEM_PROMPT: ChatPromptTemplate` + `PROMPT_HASH` are **superseded** by the YAML + Registry hybrid in §1 of this spec. The production LangGraph judge node loads its prompts from the same Registry entries used by offline scorers — single source of truth.
- **Open Question #3** (prompt file location) — **resolved**: `src/trading_databricks/agents/prompts/*.yaml` synced to Registry via `_sync.py`.

All other sections of the 2026-04-14 spec (strict-AND aggregation, thresholds, calibration corpora, judge model, rollout plan) remain authoritative.
