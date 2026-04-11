# MAS E2E Deep-Dive Research Notebook — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `notebooks/qa/mas/e2e_mas_deep_dive_research.py`, a deep-dive Databricks research notebook that runs the compiled MAS LangGraph end-to-end against live Databricks endpoints, produces per-node real-span observability, evaluates with custom scorers + LLM-as-judge via `mlflow.genai.evaluate()`, and exposes an interactive failure-autopsy sandbox.

**Architecture:** Evaluate-first with interactive sandbox (Approach 3). A notebook-local `build_instrumented_mas_graph` wraps every node in `mlflow.trace(span_type=SpanType.CHAIN)` so per-node spans are guaranteed. A sibling helper `wire_graph_environment` monkey-patches the `NotImplementedError` placeholders in the agents package without touching production code. Offline mode replaces only the LLM/embedding/VS clients — the real deterministic `_mechanism_validator_node` always runs.

**Tech Stack:** MLflow 3 GenAI eval API (`mlflow.genai.evaluate`, `make_judge`, `mlflow.search_traces`, `mlflow.log_feedback`), LangGraph `StateGraph`, Databricks Vector Search, Databricks Connect, Polars (single-node ML convention), Plotly, `pytest`, `uv`.

**Spec:** `docs/superpowers/specs/2026-04-11-mas-e2e-deep-dive-research-notebook-design.md`

---

## File Structure

### Created files

- `notebooks/qa/mas/e2e_mas_deep_dive_research.py` — the deep-dive research notebook in Databricks source format
- `tests/notebooks/test_e2e_mas_deep_dive_research.py` — structural + offline-mode tests for the new notebook and its helpers

### Modified files

- `notebooks/qa/utils.py` — extended with new helpers (`_extract_headline`, `_display`, `_get_dbutils`, `_endpoint_ready`, `_table_has_rows`, `_vs_index_exists`, `_stratified_sample`, `build_instrumented_mas_graph`, `wire_graph_environment`)

### Untouched (explicit non-goals)

- `src/trading_databricks/agents/**` — production code is read-only for this plan
- `notebooks/qa/mas/99_e2e_graph.py` — smoke test stays as-is
- `notebooks/qa/utils.py:_prepare_local_graph_environment` — the legacy patcher stays in place for the smoke test; we add new helpers alongside it, we do not remove it
- `notebooks/qa/mas/01_gatekeeper.py`…`06_mechanism_validator.py` — retrofits are a follow-up, not this plan

---

## Ordering Overview

1. **Linear ticket + branch** — CLAUDE.md requirement
2. **Shared helpers in `utils.py`** (TDD: write failing test → implement → commit). These land first so the notebook can `from notebooks.qa.utils import ...` them.
3. **Notebook scaffolding** — sections built incrementally with an offline-mode AST test guarding structure between every commit.
4. **Offline run** — confirm the whole pipeline runs green under `MAS_RESEARCH_OFFLINE=1`.
5. **Live run (gated on user approval)** — real endpoints, N=50, verify artifacts, commit final state.

---

## Task 0: Linear ticket + feature branch

**Files:** none (metadata only)

- [ ] **Step 1: Create a Linear ticket**

Use the `linear-server` MCP `save_issue` tool to create a ticket in project **Trading Databricks** (`bd63748f-7b85-4aa4-b67f-f26430a65a7f`), team `data-sci-bro`:

- Title: `MAS E2E Deep-Dive Research Notebook`
- Description: link to `docs/superpowers/specs/2026-04-11-mas-e2e-deep-dive-research-notebook-design.md`; 1-paragraph summary copied from the spec Objective.

Record the returned identifier (e.g. `TRD-NN`) — every commit in this plan must begin with it.

- [ ] **Step 2: Create feature branch**

```bash
git checkout -b feat/TRD-NN-mas-e2e-deep-dive-research-notebook
```

Replace `TRD-NN` with the real identifier from Step 1.

- [ ] **Step 3: Confirm clean working tree**

```bash
git status
```

Expected: `nothing to commit, working tree clean` and branch shown as `feat/TRD-NN-...`.

---

## Task 1: Add `_extract_headline` utility

**Files:**
- Modify: `notebooks/qa/utils.py` (append a new section)
- Create: `tests/notebooks/test_e2e_mas_deep_dive_research.py`

`_extract_headline` strips the `mas_training_dataset` request prefix to derive a clean headline. Used by §4 dataset ingestion.

- [ ] **Step 1: Write the failing test**

Create `tests/notebooks/test_e2e_mas_deep_dive_research.py` with:

```python
"""Tests for the MAS E2E deep-dive research notebook + its helpers."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from notebooks.qa.utils import _extract_headline


class TestExtractHeadline:
    def test_strips_standard_prefix(self):
        request = "Can you summarize the market impact of the following news: BTC rallies past 120k"
        assert _extract_headline(request) == "BTC rallies past 120k"

    def test_handles_extra_whitespace_after_colon(self):
        request = "Can you summarize the market impact of the following news:   Fed cuts rates"
        assert _extract_headline(request) == "Fed cuts rates"

    def test_fallback_to_full_request_when_prefix_missing(self):
        request = "Completely unrelated phrasing about ETH"
        assert _extract_headline(request) == "Completely unrelated phrasing about ETH"

    def test_empty_string_returns_empty(self):
        assert _extract_headline("") == ""

    def test_none_input_returns_empty(self):
        # Defensive against null Spark rows
        assert _extract_headline(None) == ""
```

- [ ] **Step 2: Run test and verify it fails**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestExtractHeadline -v
```

Expected: `ImportError: cannot import name '_extract_headline' from 'notebooks.qa.utils'`.

- [ ] **Step 3: Implement `_extract_headline` in `utils.py`**

Append to `notebooks/qa/utils.py` (after the existing `log_phase_metrics` helper, before the `summarize` function — group it with other text utilities):

```python
import re as _re

_HEADLINE_PREFIX_RE = _re.compile(
    r"^\s*Can you summarize the market impact of the following news:\s*",
    flags=_re.IGNORECASE,
)


def _extract_headline(request: str | None) -> str:
    """Strip the mas_training_dataset request prefix to derive a headline.

    Falls back to the full request on no match. Returns empty string
    on None input so downstream state-building is safe.
    """
    if not request:
        return ""
    return _HEADLINE_PREFIX_RE.sub("", request).strip()
```

- [ ] **Step 4: Run test and verify it passes**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestExtractHeadline -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add notebooks/qa/utils.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN feat: add _extract_headline utility for MAS deep-dive notebook"
```

---

## Task 2: Add `_display` + `_get_dbutils` dual-run guards

**Files:**
- Modify: `notebooks/qa/utils.py`
- Modify: `tests/notebooks/test_e2e_mas_deep_dive_research.py`

These helpers let the same `.py` file run under both the Databricks workspace (where `display` and `dbutils` are builtins) and `uv run python` (where they are not).

- [ ] **Step 1: Write the failing test**

Append to `tests/notebooks/test_e2e_mas_deep_dive_research.py`:

```python
from notebooks.qa.utils import _display, _get_dbutils


class TestDualRunGuards:
    def test_display_falls_back_to_print_when_builtin_absent(self, capsys):
        # In a uv run python context, `display` is not a builtin.
        _display("hello-mas")
        captured = capsys.readouterr()
        assert "hello-mas" in captured.out

    def test_get_dbutils_returns_none_outside_databricks(self):
        # uv run python has no dbutils — helper must return None, not raise.
        assert _get_dbutils() is None
```

- [ ] **Step 2: Run test and verify it fails**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestDualRunGuards -v
```

Expected: `ImportError: cannot import name '_display'`.

- [ ] **Step 3: Implement both helpers in `utils.py`**

Append to `notebooks/qa/utils.py` near `_extract_headline`:

```python
def _display(obj) -> None:
    """Dual-run display: uses Databricks `display` builtin when available,
    falls back to `print` under `uv run python`.
    """
    try:
        _display_fn = display  # type: ignore[name-defined]
    except NameError:
        print(obj)
        return
    _display_fn(obj)


def _get_dbutils():
    """Return a DBUtils instance when available (Databricks workspace),
    otherwise None. Never raises.
    """
    try:
        from pyspark.dbutils import DBUtils  # type: ignore
    except Exception:
        return None
    try:
        spark = get_spark()
        return DBUtils(spark)
    except Exception:
        return None
```

- [ ] **Step 4: Run test and verify it passes**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestDualRunGuards -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add notebooks/qa/utils.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN feat: add _display and _get_dbutils dual-run guards"
```

---

## Task 3: Add preflight probes `_endpoint_ready`, `_table_has_rows`, `_vs_index_exists`

**Files:**
- Modify: `notebooks/qa/utils.py`
- Modify: `tests/notebooks/test_e2e_mas_deep_dive_research.py`

These probe functions back the §3 preflight checks. Each returns `(ok: bool, detail: str)` and never raises — failure is surfaced as structured data so the preflight cell can log per-check metrics uniformly.

- [ ] **Step 1: Write the failing test**

Append to `tests/notebooks/test_e2e_mas_deep_dive_research.py`:

```python
from unittest.mock import MagicMock, patch

from notebooks.qa.utils import _endpoint_ready, _table_has_rows, _vs_index_exists


class TestPreflightProbes:
    def test_endpoint_ready_returns_true_when_state_is_ready(self):
        fake_client = MagicMock()
        fake_client.serving_endpoints.get.return_value.state.ready = "READY"
        with patch("notebooks.qa.utils.WorkspaceClient", return_value=fake_client):
            ok, detail = _endpoint_ready("databricks-claude-sonnet")
        assert ok is True
        assert "READY" in detail

    def test_endpoint_ready_returns_false_when_not_ready(self):
        fake_client = MagicMock()
        fake_client.serving_endpoints.get.return_value.state.ready = "NOT_READY"
        with patch("notebooks.qa.utils.WorkspaceClient", return_value=fake_client):
            ok, detail = _endpoint_ready("databricks-claude-sonnet")
        assert ok is False

    def test_endpoint_ready_catches_sdk_exception(self):
        with patch("notebooks.qa.utils.WorkspaceClient", side_effect=RuntimeError("network")):
            ok, detail = _endpoint_ready("databricks-claude-sonnet")
        assert ok is False
        assert "network" in detail

    def test_table_has_rows_true_when_count_positive(self):
        fake_spark = MagicMock()
        fake_spark.table.return_value.limit.return_value.count.return_value = 1
        with patch("notebooks.qa.utils.get_spark", return_value=fake_spark):
            ok, detail = _table_has_rows("catalog.schema.table")
        assert ok is True

    def test_table_has_rows_false_on_empty(self):
        fake_spark = MagicMock()
        fake_spark.table.return_value.limit.return_value.count.return_value = 0
        with patch("notebooks.qa.utils.get_spark", return_value=fake_spark):
            ok, detail = _table_has_rows("catalog.schema.table")
        assert ok is False

    def test_table_has_rows_catches_exception(self):
        with patch("notebooks.qa.utils.get_spark", side_effect=RuntimeError("no spark")):
            ok, detail = _table_has_rows("catalog.schema.table")
        assert ok is False
        assert "no spark" in detail

    def test_vs_index_exists_true_when_describe_succeeds(self):
        fake_vsc = MagicMock()
        fake_vsc.get_index.return_value.describe.return_value = {"status": {"ready": True}}
        with patch("notebooks.qa.utils.VectorSearchClient", return_value=fake_vsc):
            ok, detail = _vs_index_exists(endpoint_name="mas-few-shot", index_name="c.s.i")
        assert ok is True

    def test_vs_index_exists_false_on_exception(self):
        with patch("notebooks.qa.utils.VectorSearchClient", side_effect=RuntimeError("offline")):
            ok, detail = _vs_index_exists(endpoint_name="mas-few-shot", index_name="c.s.i")
        assert ok is False
        assert "offline" in detail
```

- [ ] **Step 2: Run test and verify it fails**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestPreflightProbes -v
```

Expected: `ImportError: cannot import name '_endpoint_ready'`.

- [ ] **Step 3: Implement the three probes in `utils.py`**

Append to `notebooks/qa/utils.py`:

```python
# Lazy-imported SDK references so unit tests can patch at the module level.
try:
    from databricks.sdk import WorkspaceClient  # noqa: F401
except Exception:  # pragma: no cover
    WorkspaceClient = None  # type: ignore

try:
    from databricks.vector_search.client import VectorSearchClient  # noqa: F401
except Exception:  # pragma: no cover
    VectorSearchClient = None  # type: ignore


def _endpoint_ready(endpoint_name: str) -> tuple[bool, str]:
    """Return (ok, detail) for a Databricks Model Serving endpoint."""
    try:
        if WorkspaceClient is None:
            return False, "databricks-sdk not installed"
        client = WorkspaceClient()
        ep = client.serving_endpoints.get(endpoint_name)
        ready = getattr(getattr(ep, "state", None), "ready", None)
        ok = str(ready).upper() == "READY"
        return ok, f"state.ready={ready}"
    except Exception as e:
        return False, repr(e)


def _table_has_rows(fqn: str) -> tuple[bool, str]:
    """Return (ok, detail) for a Unity Catalog table; ok iff ≥ 1 row visible."""
    try:
        spark = get_spark()
        count = spark.table(fqn).limit(1).count()
        ok = count > 0
        return ok, f"visible_rows>=1 ({count})"
    except Exception as e:
        return False, repr(e)


def _vs_index_exists(*, endpoint_name: str, index_name: str) -> tuple[bool, str]:
    """Return (ok, detail) for a Databricks Vector Search index."""
    try:
        if VectorSearchClient is None:
            return False, "databricks-vectorsearch not installed"
        vsc = VectorSearchClient()
        idx = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)
        desc = idx.describe()
        return True, f"describe_ok ({type(desc).__name__})"
    except Exception as e:
        return False, repr(e)
```

- [ ] **Step 4: Run test and verify it passes**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestPreflightProbes -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add notebooks/qa/utils.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN feat: add preflight probes for endpoints, tables, VS indexes"
```

---

## Task 4: Add `_stratified_sample` helper

**Files:**
- Modify: `notebooks/qa/utils.py`
- Modify: `tests/notebooks/test_e2e_mas_deep_dive_research.py`

Seeded stratified sampling by a bucket column — used by §4 to sample by published-date bucket.

- [ ] **Step 1: Write the failing test**

Append to `tests/notebooks/test_e2e_mas_deep_dive_research.py`:

```python
import polars as pl

from notebooks.qa.utils import _stratified_sample


class TestStratifiedSample:
    def _frame(self) -> pl.DataFrame:
        return pl.DataFrame({
            "story_id": [str(i) for i in range(20)],
            "_bucket": ["a"] * 10 + ["b"] * 10,
            "request": [f"r{i}" for i in range(20)],
        })

    def test_returns_requested_total_size(self):
        df = self._frame()
        out = _stratified_sample(df, n=8, by="_bucket", seed=42)
        assert out.height == 8

    def test_distributes_across_buckets(self):
        df = self._frame()
        out = _stratified_sample(df, n=8, by="_bucket", seed=42)
        counts = out.group_by("_bucket").len().sort("_bucket")
        # stratified: both buckets must be represented
        assert counts.height == 2
        assert counts["len"].min() >= 1

    def test_seed_is_deterministic(self):
        df = self._frame()
        a = _stratified_sample(df, n=8, by="_bucket", seed=42)
        b = _stratified_sample(df, n=8, by="_bucket", seed=42)
        assert a["story_id"].to_list() == b["story_id"].to_list()

    def test_n_larger_than_frame_returns_full_frame(self):
        df = self._frame()
        out = _stratified_sample(df, n=50, by="_bucket", seed=42)
        assert out.height == df.height
```

- [ ] **Step 2: Run test and verify it fails**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestStratifiedSample -v
```

Expected: `ImportError: cannot import name '_stratified_sample'`.

- [ ] **Step 3: Implement `_stratified_sample` in `utils.py`**

Append to `notebooks/qa/utils.py`:

```python
def _stratified_sample(
    df: pl.DataFrame,
    *,
    n: int,
    by: str,
    seed: int,
) -> pl.DataFrame:
    """Seeded stratified sample of `df` by column `by`, returning up to `n` rows.

    Allocates proportionally to each bucket's share, with a floor of 1 per
    bucket. If `n >= df.height`, returns the full frame untouched (still
    seeded for deterministic downstream operations).
    """
    if n >= df.height:
        return df

    bucket_counts = df.group_by(by).len().sort(by)
    total = df.height
    parts: list[pl.DataFrame] = []
    remaining = n
    buckets = bucket_counts.to_dicts()
    for i, row in enumerate(buckets):
        bucket_value = row[by]
        bucket_size = int(row["len"])
        if i == len(buckets) - 1:
            take = remaining
        else:
            take = max(1, round(n * bucket_size / total))
            take = min(take, bucket_size, remaining - (len(buckets) - i - 1))
            take = max(take, 0)
        if take <= 0:
            continue
        slice_df = df.filter(pl.col(by) == bucket_value).sample(
            n=min(take, bucket_size), seed=seed, shuffle=True,
        )
        parts.append(slice_df)
        remaining -= slice_df.height
    return pl.concat(parts) if parts else df.head(0)
```

- [ ] **Step 4: Run test and verify it passes**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestStratifiedSample -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add notebooks/qa/utils.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN feat: add seeded stratified sampling helper"
```

---

## Task 5: Add `build_instrumented_mas_graph` helper

**Files:**
- Modify: `notebooks/qa/utils.py`
- Modify: `tests/notebooks/test_e2e_mas_deep_dive_research.py`

Mirrors `build_mas_graph` topology but wraps every node in `mlflow.trace(name=..., span_type=SpanType.CHAIN)` so per-node spans are guaranteed regardless of LangGraph autolog coverage or MLflow version drift. Uses production node functions unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/notebooks/test_e2e_mas_deep_dive_research.py`:

```python
class TestBuildInstrumentedMasGraph:
    def test_compiles_with_five_nodes(self):
        from notebooks.qa.utils import build_instrumented_mas_graph
        from trading_databricks.agents.config import MASConfig

        compiled = build_instrumented_mas_graph(MASConfig())
        mermaid = compiled.get_graph().draw_mermaid()
        for node in (
            "context_retriever",
            "few_shot_retriever",
            "uncertainty_router",
            "causality_extractor",
            "mechanism_validator",
        ):
            assert node in mermaid, f"{node} missing from compiled graph topology"

    def test_entry_and_finish_points(self):
        from notebooks.qa.utils import build_instrumented_mas_graph
        from trading_databricks.agents.config import MASConfig

        compiled = build_instrumented_mas_graph(MASConfig())
        g = compiled.get_graph()
        # LangGraph exposes nodes dict + edges list on get_graph() result
        assert "context_retriever" in g.nodes
        assert "mechanism_validator" in g.nodes
```

- [ ] **Step 2: Run test and verify it fails**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestBuildInstrumentedMasGraph -v
```

Expected: `ImportError: cannot import name 'build_instrumented_mas_graph'`.

- [ ] **Step 3: Implement `build_instrumented_mas_graph` in `utils.py`**

Append to `notebooks/qa/utils.py`:

```python
def build_instrumented_mas_graph(config):
    """Compile the 5-node MAS graph with every node wrapped in mlflow.trace.

    Topology mirrors trading_databricks.agents.graph.build_mas_graph exactly
    but guarantees per-node CHAIN spans for §8 real-latency analysis.
    Production node code is unchanged.
    """
    from langgraph.graph import StateGraph
    from mlflow.entities import SpanType

    from trading_databricks.agents.graph import (
        _mechanism_validator_node,
        route_article,
    )
    from trading_databricks.agents.nodes import (
        causality_extractor,
        context_retriever,
        few_shot_retriever,
        uncertainty_router,
    )
    from trading_databricks.agents.state import MASState

    def _traced(name: str, fn):
        @mlflow.trace(name=name, span_type=SpanType.CHAIN)
        def _wrapper(state):
            return fn(state)
        _wrapper.__name__ = f"traced_{name}"
        return _wrapper

    graph = StateGraph(MASState)
    graph.add_node("context_retriever",   _traced("context_retriever",   lambda s: context_retriever(s, config=config)))
    graph.add_node("few_shot_retriever",  _traced("few_shot_retriever",  lambda s: few_shot_retriever(s, config=config)))
    graph.add_node("uncertainty_router",  _traced("uncertainty_router",  lambda s: uncertainty_router(s, config=config)))
    graph.add_node("causality_extractor", _traced("causality_extractor", lambda s: causality_extractor(s, config=config)))
    graph.add_node("mechanism_validator", _traced("mechanism_validator", lambda s: _mechanism_validator_node(s, config)))

    graph.set_entry_point("context_retriever")
    graph.add_edge("context_retriever", "few_shot_retriever")
    graph.add_edge("few_shot_retriever", "uncertainty_router")
    graph.add_conditional_edges(
        "uncertainty_router",
        route_article,
        {"causality_extractor": "causality_extractor"},
    )
    graph.add_edge("causality_extractor", "mechanism_validator")
    graph.set_finish_point("mechanism_validator")
    return graph.compile()
```

- [ ] **Step 4: Run test and verify it passes**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestBuildInstrumentedMasGraph -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add notebooks/qa/utils.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN feat: add build_instrumented_mas_graph with per-node trace spans"
```

---

## Task 6: Add `wire_graph_environment` helper (offline + live modes)

**Files:**
- Modify: `notebooks/qa/utils.py`
- Modify: `tests/notebooks/test_e2e_mas_deep_dive_research.py`

Monkey-patches the runtime placeholders (`get_spark_session`, `query_vs_index`) with real or mock implementations depending on `offline`. Offline mode additionally patches the deploy client's `predict` method with a deterministic context-sensitive mock. **`_mechanism_validator_node` is NEVER patched** in any mode — this is the critical regression fix vs `99_e2e_graph.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/notebooks/test_e2e_mas_deep_dive_research.py`:

```python
class TestWireGraphEnvironment:
    def test_offline_mode_replaces_query_vs_index(self):
        from notebooks.qa.utils import build_instrumented_mas_graph, wire_graph_environment
        from trading_databricks.agents.config import MASConfig
        from trading_databricks.agents.nodes import few_shot_retriever as fsr

        config = MASConfig()
        original = fsr.query_vs_index
        try:
            with wire_graph_environment(config, offline=True):
                # Placeholder must be replaced with a callable that does not raise
                result = fsr.query_vs_index(
                    index_name="x",
                    columns=[],
                    query_text="Fed raises rates",
                    num_results=5,
                    filters_json="{}",
                )
                assert isinstance(result, list)
        finally:
            assert fsr.query_vs_index is original, "wire_graph_environment must restore on exit"

    def test_offline_mode_never_patches_mechanism_validator(self):
        """Regression guard: _mechanism_validator_node must remain the real function."""
        from notebooks.qa.utils import wire_graph_environment
        from trading_databricks.agents import graph as agents_graph
        from trading_databricks.agents.config import MASConfig

        real = agents_graph._mechanism_validator_node
        with wire_graph_environment(MASConfig(), offline=True):
            assert agents_graph._mechanism_validator_node is real

    def test_offline_mode_predict_returns_deterministic_llm_shape(self):
        """Offline mock predict returns a taxonomy-compliant JSON response."""
        import json
        import mlflow.deployments

        from notebooks.qa.utils import wire_graph_environment
        from trading_databricks.agents.config import MASConfig

        config = MASConfig()
        with wire_graph_environment(config, offline=True):
            client = mlflow.deployments.get_deploy_client("databricks")
            resp = client.predict(
                endpoint=config.llm_endpoint,
                inputs={"messages": [{"role": "user", "content": "x"}], "temperature": 0.0, "n": 1, "max_tokens": 100},
            )
            assert "choices" in resp
            content = resp["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            assert "treatment_class" in parsed
            assert "mechanism" in parsed
            assert "expected_direction" in parsed

    def test_offline_mode_predict_returns_embedding_shape(self):
        import mlflow.deployments

        from notebooks.qa.utils import wire_graph_environment
        from trading_databricks.agents.config import MASConfig

        config = MASConfig()
        with wire_graph_environment(config, offline=True):
            client = mlflow.deployments.get_deploy_client("databricks")
            resp = client.predict(
                endpoint=config.validator.embedding_model,
                inputs={"input": ["hello"]},
            )
            assert "data" in resp
            assert "embedding" in resp["data"][0]
            assert isinstance(resp["data"][0]["embedding"], list)
```

- [ ] **Step 2: Run test and verify it fails**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestWireGraphEnvironment -v
```

Expected: `ImportError: cannot import name 'wire_graph_environment'`.

- [ ] **Step 3: Implement `wire_graph_environment` in `utils.py`**

Append to `notebooks/qa/utils.py`:

```python
@contextmanager
def wire_graph_environment(config, *, offline: bool):
    """Patch runtime hooks so the MAS graph can execute from a notebook.

    Live mode (offline=False):
      - Replaces `get_spark_session` placeholders with a Databricks Connect session
      - Replaces `query_vs_index` placeholder with a real VectorSearchClient call

    Offline mode (offline=True):
      - All of the above with deterministic mocks
      - Additionally patches mlflow.deployments get_deploy_client.predict to return
        taxonomy-compliant JSON (LLM) or a fixed unit vector (embeddings)

    The mechanism_validator_node is NEVER patched in any mode — the real
    deterministic validation logic runs on every row.
    """
    import json as _json
    from unittest.mock import patch as _patch

    from trading_databricks.agents.nodes import causality_extractor as _ce
    from trading_databricks.agents.nodes import few_shot_retriever as _fsr

    patches: list = []

    patches.append(_patch.object(_fsr, "get_spark_session", get_spark))
    patches.append(_patch.object(_ce, "get_spark_session", get_spark))

    if offline:
        def _mock_vs_query(**kwargs):
            return [
                {"story_id": "mock-1", "treatment_class": "Regulation.etf_ruling",
                 "mechanism": "institutional_access", "expected_direction": "bullish",
                 "confidence": 0.9, "trigger_event": "mock trigger",
                 "causal_chain_json": "{}", "similarity_score": 0.9},
                {"story_id": "mock-2", "treatment_class": "Macro.rate_decision",
                 "mechanism": "liquidity_contraction", "expected_direction": "bearish",
                 "confidence": 0.8, "trigger_event": "mock trigger 2",
                 "causal_chain_json": "{}", "similarity_score": 0.7},
            ]
        patches.append(_patch.object(_fsr, "query_vs_index", _mock_vs_query))

        llm_endpoint = config.llm_endpoint
        embed_endpoint = config.validator.embedding_model

        _mock_json = _json.dumps({
            "treatment_class": "Macro.rate_decision",
            "mechanism": "liquidity_contraction",
            "expected_direction": "bearish",
            "confidence": 0.8,
            "trigger_event": "mock trigger event",
            "evidence_spans": ["mock evidence"],
            "reasoning": "deterministic offline mock",
        })

        class _MockDeployClient:
            def predict(self, *, endpoint, inputs):
                if endpoint == embed_endpoint:
                    return {"data": [{"embedding": [1.0, 0.0, 0.0]}]}
                if endpoint == llm_endpoint:
                    n = int(inputs.get("n", 1))
                    return {
                        "choices": [
                            {"message": {"content": _mock_json}} for _ in range(n)
                        ]
                    }
                raise RuntimeError(f"Unexpected offline endpoint: {endpoint}")

        patches.append(
            _patch("mlflow.deployments.get_deploy_client", return_value=_MockDeployClient())
        )
    else:
        import json as _json2

        def _real_vs_query(**kwargs):
            vsc = VectorSearchClient()
            idx = vsc.get_index(
                endpoint_name=config.vs_endpoint_name,
                index_name=kwargs["index_name"],
            )
            raw = idx.similarity_search(
                query_text=kwargs["query_text"],
                columns=kwargs["columns"],
                num_results=kwargs["num_results"],
                filters=_json2.loads(kwargs["filters_json"]),
            )
            rows = raw.get("result", {}).get("data_array", []) or []
            cols = kwargs["columns"]
            out: list[dict] = []
            for row in rows:
                rec = {col: row[i] for i, col in enumerate(cols)}
                if len(row) > len(cols):
                    rec["similarity_score"] = row[len(cols)]
                out.append(rec)
            return out

        patches.append(_patch.object(_fsr, "query_vs_index", _real_vs_query))

    started = [p.start() for p in patches]
    try:
        yield
    finally:
        for p in patches:
            p.stop()
```

- [ ] **Step 4: Run test and verify it passes**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestWireGraphEnvironment -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add notebooks/qa/utils.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN feat: add wire_graph_environment helper for live and offline modes"
```

---

## Task 7: Create notebook scaffold (§0 header + §1 setup)

**Files:**
- Create: `notebooks/qa/mas/e2e_mas_deep_dive_research.py`
- Modify: `tests/notebooks/test_e2e_mas_deep_dive_research.py`

From this task forward every notebook cell is paired with an AST-level structural test so the notebook's shape is locked in as it grows.

- [ ] **Step 1: Write the failing structural test**

Append to `tests/notebooks/test_e2e_mas_deep_dive_research.py`:

```python
NOTEBOOK_PATH = Path("notebooks/qa/mas/e2e_mas_deep_dive_research.py")


def _notebook_text() -> str:
    assert NOTEBOOK_PATH.exists(), f"notebook not found at {NOTEBOOK_PATH}"
    return NOTEBOOK_PATH.read_text()


class TestNotebookStructure:
    def test_first_line_is_databricks_source_marker(self):
        content = _notebook_text()
        assert content.splitlines()[0] == "# Databricks notebook source"

    def test_has_setup_section(self):
        content = _notebook_text()
        assert "## §1 Setup" in content
        assert "MAS_RESEARCH_OFFLINE" in content
        assert "MAS_RESEARCH_SAMPLE_SIZE" in content
        assert "mlflow_component_run" in content
        assert "mlflow.langchain.autolog()" in content

    def test_notebook_is_valid_python(self):
        """The file must parse as Python — MAGIC lines are comments, COMMAND markers are comments."""
        content = _notebook_text()
        ast.parse(content)
```

- [ ] **Step 2: Run test and verify it fails**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure -v
```

Expected: `AssertionError: notebook not found at notebooks/qa/mas/e2e_mas_deep_dive_research.py`.

- [ ] **Step 3: Create the notebook scaffold**

Create `notebooks/qa/mas/e2e_mas_deep_dive_research.py` with exactly:

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # MAS E2E Deep-Dive Research Notebook
# MAGIC
# MAGIC **Objective:** Run the compiled MAS LangGraph end-to-end on real news from
# MAGIC `smart-trading-dev.others.mas_training_dataset` against live
# MAGIC `databricks-claude-sonnet` + `databricks-gte-large-en` endpoints, produce
# MAGIC per-node real-span observability, and evaluate with custom scorers and an
# MAGIC LLM-as-judge via `mlflow.genai.evaluate()`. Spec:
# MAGIC `docs/superpowers/specs/2026-04-11-mas-e2e-deep-dive-research-notebook-design.md`
# MAGIC
# MAGIC **How to run:**
# MAGIC - Databricks workspace: open and Run All; widgets for the autopsy sandbox
# MAGIC   appear in §11.
# MAGIC - Local CLI: `uv run python notebooks/qa/mas/e2e_mas_deep_dive_research.py`
# MAGIC
# MAGIC **Env vars (see also §1 Setup):**
# MAGIC - `MAS_RESEARCH_OFFLINE=1` — bypass preflight hard-fails, inject mocks
# MAGIC - `MAS_RESEARCH_SAMPLE_SIZE` (default 50)
# MAGIC - `MAS_RESEARCH_NEG_CONTROL_SIZE` (default 5)
# MAGIC - `MAS_RESEARCH_SEED` (default 42)
# MAGIC
# MAGIC **⚠ Live mode spends real tokens.** Estimated ~535k tokens at N=50.
# MAGIC
# MAGIC **Paired smoke test:** `notebooks/qa/mas/99_e2e_graph.py` stays as-is — this
# MAGIC notebook is the research instrument, not a CI gate.

# COMMAND ----------

# MAGIC %md
# MAGIC ## §1 Setup

# COMMAND ----------

import os

import mlflow
import polars as pl

from notebooks.qa.utils import (
    _display,
    _endpoint_ready,
    _extract_headline,
    _get_dbutils,
    _stratified_sample,
    _table_has_rows,
    _vs_index_exists,
    build_instrumented_mas_graph,
    get_spark,
    log_diag,
    log_plotly,
    mlflow_component_run,
    wire_graph_environment,
)
from trading_databricks.agents.config import MASConfig

OFFLINE = os.environ.get("MAS_RESEARCH_OFFLINE") == "1"
SAMPLE_SIZE = int(os.environ.get("MAS_RESEARCH_SAMPLE_SIZE", "50"))
NEG_CONTROL_SIZE = int(os.environ.get("MAS_RESEARCH_NEG_CONTROL_SIZE", "5"))
SEED = int(os.environ.get("MAS_RESEARCH_SEED", "42"))

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/trading-databricks-qa/mas/e2e_deep_dive_research")
mlflow.langchain.autolog()
config = MASConfig()

params = {
    "mode": "offline" if OFFLINE else "live",
    "sample_size": SAMPLE_SIZE,
    "neg_control_size": NEG_CONTROL_SIZE,
    "seed": SEED,
    "llm_endpoint": config.llm_endpoint,
    "embedding_endpoint": config.validator.embedding_model,
    "vs_index": config.vs_index_name,
    "vs_endpoint": config.vs_endpoint_name,
    "dataset_fqn": "smart-trading-dev.others.mas_training_dataset",
}

with mlflow_component_run(
    layer="mas", component="e2e_deep_dive_research", params=params,
):
    mlflow.set_tag("mode", "offline" if OFFLINE else "live")
    mlflow.log_metric("tokens.estimate_pre_run", 535_000)
    _display(f"Mode: {params['mode']}  |  Sample: {SAMPLE_SIZE}  |  Seed: {SEED}")
```

**Important notebook-file rules** — the rest of this plan appends cells to this file. Every code cell is preceded by `# COMMAND ----------` with one blank line before and after the marker. Every markdown cell prefixes each line with `# MAGIC %md` for the first line and `# MAGIC` for subsequent lines.

The `with mlflow_component_run(...):` block is the single top-level run context for the entire notebook. All subsequent code cells that want to log to this run must be indented inside this block. This plan will show the additional cells with the correct indentation.

- [ ] **Step 4: Run structural test and verify it passes**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure -v
```

Expected: 3 passed.

- [ ] **Step 5: Verify the file parses under `uv run python` dry mode**

```bash
uv run python -c "import ast; ast.parse(open('notebooks/qa/mas/e2e_mas_deep_dive_research.py').read()); print('ok')"
```

Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add notebooks/qa/mas/e2e_mas_deep_dive_research.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN feat: scaffold MAS e2e deep-dive research notebook with §1 Setup"
```

---

## Task 8: Add §2 system topology diagram cell

**Files:**
- Modify: `notebooks/qa/mas/e2e_mas_deep_dive_research.py`
- Modify: `tests/notebooks/test_e2e_mas_deep_dive_research.py`

- [ ] **Step 1: Add the structural test**

Add to `TestNotebookStructure` in `tests/notebooks/test_e2e_mas_deep_dive_research.py`:

```python
    def test_has_topology_diagram_section(self):
        content = _notebook_text()
        assert "## §2 System topology diagram" in content
        assert "build_instrumented_mas_graph(config)" in content
        assert "draw_mermaid" in content
        assert 'mlflow.log_text(mermaid, "topology/graph.mmd")' in content
```

- [ ] **Step 2: Run test and verify it fails**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure::test_has_topology_diagram_section -v
```

Expected: AssertionError.

- [ ] **Step 3: Append the §2 cell to the notebook**

Append to `notebooks/qa/mas/e2e_mas_deep_dive_research.py` (inside the `with mlflow_component_run(...):` block — maintain indentation):

```python

    # COMMAND ----------

    # MAGIC %md
    # MAGIC ## §2 System topology diagram
    # MAGIC The compiled MAS graph rendered as Mermaid. In the Databricks workspace,
    # MAGIC the fenced ```mermaid``` block below the code cell renders as a diagram.

    # COMMAND ----------

    graph = build_instrumented_mas_graph(config)
    mermaid = graph.get_graph().draw_mermaid()
    mlflow.log_text(mermaid, "topology/graph.mmd")
    _display(mermaid)

    # COMMAND ----------

    # MAGIC %md
    # MAGIC ```mermaid
    # MAGIC graph TD
    # MAGIC   context_retriever --> few_shot_retriever
    # MAGIC   few_shot_retriever --> uncertainty_router
    # MAGIC   uncertainty_router --> causality_extractor
    # MAGIC   causality_extractor --> mechanism_validator
    # MAGIC ```
```

Note: the static Mermaid block below the code cell is a rendering fallback for the Databricks markdown renderer; the code cell logs the authoritative mermaid to MLflow.

- [ ] **Step 4: Run the structural test and parsing check**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure -v
uv run python -c "import ast; ast.parse(open('notebooks/qa/mas/e2e_mas_deep_dive_research.py').read()); print('ok')"
```

Expected: all green; `ok`.

- [ ] **Step 5: Commit**

```bash
git add notebooks/qa/mas/e2e_mas_deep_dive_research.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN feat: add §2 topology diagram cell to deep-dive notebook"
```

---

## Task 9: Add §3 preflight checks cell

**Files:**
- Modify: `notebooks/qa/mas/e2e_mas_deep_dive_research.py`
- Modify: `tests/notebooks/test_e2e_mas_deep_dive_research.py`

- [ ] **Step 1: Add the structural test**

Add to `TestNotebookStructure`:

```python
    def test_has_preflight_section(self):
        content = _notebook_text()
        assert "## §3 Pre-flight checks" in content
        assert "preflight_rows" in content
        assert "_endpoint_ready(config.llm_endpoint)" in content
        assert "_endpoint_ready(config.validator.embedding_model)" in content
        assert "_table_has_rows" in content
        assert "_vs_index_exists" in content
        assert "cold_start_mode" in content
        assert "accepted_chain_count" in content
        # hard-fail enforcement
        assert 'raise RuntimeError("preflight critical check failed' in content
```

- [ ] **Step 2: Run test and verify it fails**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure::test_has_preflight_section -v
```

Expected: AssertionError.

- [ ] **Step 3: Append the §3 cell**

Append to the notebook (still inside the `with mlflow_component_run(...):` block):

```python

    # COMMAND ----------

    # MAGIC %md
    # MAGIC ## §3 Pre-flight checks
    # MAGIC Probes every external dependency. Critical failures hard-fail the run
    # MAGIC unless `MAS_RESEARCH_OFFLINE=1`. Cold-start is detected by querying the
    # MAGIC accepted-chain count and tagged on the run so routing distributions are
    # MAGIC interpreted correctly downstream.

    # COMMAND ----------

    preflight_rows: list[dict] = []

    def _check(name: str, ok: bool, detail: str, critical: bool) -> None:
        preflight_rows.append(
            {"name": name, "ok": ok, "detail": detail, "critical": critical}
        )
        mlflow.log_metric(f"preflight.{name}.ok", int(ok))

    ok, detail = _endpoint_ready(config.llm_endpoint)
    _check("endpoint.claude_sonnet", ok, detail, critical=True)

    ok, detail = _endpoint_ready(config.validator.embedding_model)
    _check("endpoint.gte_large_en", ok, detail, critical=False)

    ok, detail = _table_has_rows(f"{config.catalog}.{config.schema}.treatment_taxonomy")
    _check("table.treatment_taxonomy", ok, detail, critical=True)

    ok, detail = _table_has_rows(f"{config.catalog}.{config.schema}.mechanism_whitelist")
    _check("table.mechanism_whitelist", ok, detail, critical=True)

    ok, detail = _table_has_rows(f"{config.catalog}.{config.schema}.news_causal_features")
    _check("table.news_causal_features", ok, detail, critical=False)

    ok, detail = _vs_index_exists(
        endpoint_name=config.vs_endpoint_name, index_name=config.vs_index_name,
    )
    _check("vs.causal_chains_idx", ok, detail, critical=False)

    ok, detail = _table_has_rows("smart-trading-dev.others.mas_training_dataset")
    _check("table.mas_training_dataset", ok, detail, critical=True)

    preflight_df = pl.DataFrame(preflight_rows)
    log_diag("preflight", preflight_df)
    _display(preflight_df)

    # Cold-start detection
    accepted_count = 0
    try:
        spark = get_spark()
        accepted_count = spark.sql(
            f"SELECT COUNT(*) AS n FROM `{config.catalog}`.`{config.schema}`.news_causal_features "
            f"WHERE validation_status = 'accepted'"
        ).collect()[0]["n"]
    except Exception as e:
        _display(f"accepted_count query failed ({e!r}); treating as cold-start")

    cold_start_mode = accepted_count < config.router.min_index_size
    mlflow.set_tag("cold_start", str(cold_start_mode).lower())
    mlflow.log_metric("cold_start_mode", int(cold_start_mode))
    mlflow.log_metric("accepted_chain_count", accepted_count)

    if cold_start_mode:
        _display(
            "⚠ COLD START MODE — accepted_chain_count < router.min_index_size. "
            "All rows will route to 'careful' path. routing_legality scorer will "
            "accept only 'careful' as legal."
        )

    # Hard-fail on critical preflight failures (live mode only)
    critical_failures = [r for r in preflight_rows if r["critical"] and not r["ok"]]
    if critical_failures and not OFFLINE:
        for r in critical_failures:
            _display(f"CRITICAL: {r['name']} — {r['detail']}")
        raise RuntimeError(
            "preflight critical check failed; set MAS_RESEARCH_OFFLINE=1 to bypass "
            f"(failed: {[r['name'] for r in critical_failures]})"
        )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add notebooks/qa/mas/e2e_mas_deep_dive_research.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN feat: add §3 pre-flight checks with cold-start detection"
```

---

## Task 10: Add §4 dataset ingestion + sampling + lineage cell

**Files:**
- Modify: `notebooks/qa/mas/e2e_mas_deep_dive_research.py`
- Modify: `tests/notebooks/test_e2e_mas_deep_dive_research.py`

- [ ] **Step 1: Add the structural test**

Add to `TestNotebookStructure`:

```python
    def test_has_dataset_ingestion_section(self):
        content = _notebook_text()
        assert "## §4 Dataset ingestion" in content
        assert "smart-trading-dev.others.mas_training_dataset" in content
        assert 'expected_route' in content
        assert "_extract_headline" in content
        assert "_stratified_sample" in content
        assert "DeltaDatasetSource" in content
        assert "mlflow.log_input" in content
        assert "eval_dataset" in content
        assert "negative_control_dataset" in content
```

- [ ] **Step 2: Run test and verify it fails**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure::test_has_dataset_ingestion_section -v
```

Expected: AssertionError.

- [ ] **Step 3: Append the §4 cell**

Append to the notebook (inside the `with mlflow_component_run(...):` block):

```python

    # COMMAND ----------

    # MAGIC %md
    # MAGIC ## §4 Dataset ingestion, stratified sampling, lineage
    # MAGIC Loads `smart-trading-dev.others.mas_training_dataset`, derives
    # MAGIC `headline`/`body` for MASState, filters to `expected_route='news_agent'`,
    # MAGIC stratified-samples by published-date bucket, and logs a dataset-lineage
    # MAGIC record pointing back to the Delta source.

    # COMMAND ----------

    spark_session = get_spark()

    raw = pl.from_arrow(
        spark_session.table("smart-trading-dev.others.mas_training_dataset").toArrow()
    )

    if "published_at" in raw.columns:
        raw = raw.with_columns(
            pl.col("published_at").cast(pl.Datetime, strict=False).alias("published_at")
        )

    news = raw.filter(pl.col("expected_route") == "news_agent")
    news = news.with_columns(
        pl.col("request").map_elements(_extract_headline, return_dtype=pl.Utf8).alias("headline"),
        pl.col("context").fill_null("").alias("body"),
    )
    news = news.with_columns(
        pl.col("published_at").dt.date().alias("_bucket")
    )

    sample = _stratified_sample(news, n=SAMPLE_SIZE, by="_bucket", seed=SEED)
    mlflow.log_metric("dataset.news_rows_pre_sample", news.height)
    mlflow.log_metric("dataset.news_rows_sampled", sample.height)

    eval_dataset = [
        {"inputs": {
            "headline": r["headline"],
            "body": r["body"],
            "published_at": str(r["published_at"]),
            "story_id": r["story_id"],
        }}
        for r in sample.to_dicts()
    ]

    # Negative control slice — non-news_agent
    non_news = raw.filter(pl.col("expected_route") != "news_agent")
    neg_n = min(NEG_CONTROL_SIZE, non_news.height)
    if neg_n > 0:
        negative = non_news.sample(n=neg_n, seed=SEED, shuffle=True)
        negative = negative.with_columns(
            pl.col("request").map_elements(_extract_headline, return_dtype=pl.Utf8).alias("headline"),
            pl.col("context").fill_null("").alias("body"),
        )
        negative_control_dataset = [
            {"inputs": {
                "headline": r["headline"],
                "body": r["body"],
                "published_at": str(r["published_at"]),
                "story_id": r["story_id"],
            }}
            for r in negative.to_dicts()
        ]
    else:
        negative_control_dataset = []
    mlflow.log_metric("dataset.negative_control_size", len(negative_control_dataset))

    # Dataset lineage
    try:
        from mlflow.data.delta_dataset_source import DeltaDatasetSource
        ds = mlflow.data.from_pandas(
            sample.drop("_bucket").to_pandas(),
            source=DeltaDatasetSource(delta_table_name="smart-trading-dev.others.mas_training_dataset"),
            name="mas_deep_dive_research_sample",
        )
        mlflow.log_input(ds, context="research")
    except Exception as e:
        _display(f"dataset lineage log skipped: {e!r}")

    _display(sample.drop("_bucket").head(10))
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add notebooks/qa/mas/e2e_mas_deep_dive_research.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN feat: add §4 dataset ingestion with stratified sampling and lineage"
```

---

## Task 11: Add §5 graph wire-up cell

**Files:**
- Modify: `notebooks/qa/mas/e2e_mas_deep_dive_research.py`
- Modify: `tests/notebooks/test_e2e_mas_deep_dive_research.py`

- [ ] **Step 1: Add the structural test**

Add to `TestNotebookStructure`:

```python
    def test_has_graph_wireup_section(self):
        content = _notebook_text()
        assert "## §5 Graph wiring" in content
        assert "wire_graph_environment(config, offline=OFFLINE)" in content
```

- [ ] **Step 2: Run test and verify it fails**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure::test_has_graph_wireup_section -v
```

Expected: AssertionError.

- [ ] **Step 3: Append the §5 cell**

Append:

```python

    # COMMAND ----------

    # MAGIC %md
    # MAGIC ## §5 Graph wiring + instrumentation
    # MAGIC Opens `wire_graph_environment` for the duration of evaluation. In live
    # MAGIC mode this replaces the `NotImplementedError` placeholders for
    # MAGIC `get_spark_session`/`query_vs_index` with real implementations. In
    # MAGIC offline mode all external clients are replaced with deterministic mocks
    # MAGIC — the real deterministic `_mechanism_validator_node` always runs.

    # COMMAND ----------

    graph_env_ctx = wire_graph_environment(config, offline=OFFLINE)
    graph_env_ctx.__enter__()
```

This explicit `__enter__()` pattern lets us keep the notebook linear (cell-by-cell) rather than nesting everything inside a second `with` block. §14 calls `graph_env_ctx.__exit__(None, None, None)` to release patches.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add notebooks/qa/mas/e2e_mas_deep_dive_research.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN feat: add §5 graph wire-up with environment patching"
```

---

## Task 12: Add `predict_fn` and the 5 custom scorers

**Files:**
- Modify: `notebooks/qa/mas/e2e_mas_deep_dive_research.py`
- Modify: `tests/notebooks/test_e2e_mas_deep_dive_research.py`

The scorers are notebook-local research artifacts. We test them as callable units by importing them directly from the notebook source via `runpy`-free AST/exec — but since execing the whole notebook brings in Databricks deps, we instead factor each scorer's **pure predicate** into a plain-Python helper block and test that by copy-import.

This task uses a different test strategy: **extract the scorer predicates into a standalone Python module** `notebooks/qa/mas/_scorer_helpers.py` that the notebook imports. That module is cheap to unit-test without running the notebook.

- [ ] **Step 1: Write failing tests for scorer helper functions**

Append to `tests/notebooks/test_e2e_mas_deep_dive_research.py`:

```python
class TestScorerHelpers:
    def test_check_schema_conformance_all_valid(self):
        from notebooks.qa.mas._scorer_helpers import check_schema_conformance

        extraction = {
            "treatment_class": "Macro.rate_decision",
            "mechanism": "liquidity_contraction",
            "expected_direction": "bearish",
            "confidence": 0.8,
            "trigger_event": "Fed raises rates",
            "evidence_spans": ["Fed raises rates 75 bps"],
        }
        assert check_schema_conformance(extraction) == 1.0

    def test_check_schema_conformance_missing_field(self):
        from notebooks.qa.mas._scorer_helpers import check_schema_conformance

        extraction = {
            "treatment_class": "Macro.rate_decision",
            "mechanism": "liquidity_contraction",
            "expected_direction": "bearish",
            "confidence": 0.8,
            # missing trigger_event
            "evidence_spans": ["x"],
        }
        result = check_schema_conformance(extraction)
        assert 0.0 <= result < 1.0

    def test_check_schema_conformance_none_returns_zero(self):
        from notebooks.qa.mas._scorer_helpers import check_schema_conformance

        assert check_schema_conformance(None) == 0.0

    def test_check_schema_conformance_invalid_direction(self):
        from notebooks.qa.mas._scorer_helpers import check_schema_conformance

        extraction = {
            "treatment_class": "Macro.rate_decision",
            "mechanism": "liquidity_contraction",
            "expected_direction": "sideways",  # invalid
            "confidence": 0.8,
            "trigger_event": "Fed raises rates",
            "evidence_spans": ["x"],
        }
        result = check_schema_conformance(extraction)
        assert result < 1.0

    def test_check_schema_conformance_bad_confidence(self):
        from notebooks.qa.mas._scorer_helpers import check_schema_conformance

        extraction = {
            "treatment_class": "Macro.rate_decision",
            "mechanism": "liquidity_contraction",
            "expected_direction": "bearish",
            "confidence": 1.7,  # out of [0,1]
            "trigger_event": "x",
            "evidence_spans": ["x"],
        }
        assert check_schema_conformance(extraction) < 1.0

    def test_check_taxonomy_whitelist_compliance_valid(self):
        from notebooks.qa.mas._scorer_helpers import check_taxonomy_whitelist_compliance

        whitelist = {("Macro", "liquidity_contraction"): ["bearish", "volatile"]}
        extraction = {
            "treatment_class": "Macro.rate_decision",
            "mechanism": "liquidity_contraction",
            "expected_direction": "bearish",
        }
        assert check_taxonomy_whitelist_compliance(extraction, whitelist) is True

    def test_check_taxonomy_whitelist_compliance_wrong_direction(self):
        from notebooks.qa.mas._scorer_helpers import check_taxonomy_whitelist_compliance

        whitelist = {("Macro", "liquidity_contraction"): ["bearish"]}
        extraction = {
            "treatment_class": "Macro.rate_decision",
            "mechanism": "liquidity_contraction",
            "expected_direction": "bullish",
        }
        assert check_taxonomy_whitelist_compliance(extraction, whitelist) is False

    def test_check_taxonomy_whitelist_compliance_unknown_pair(self):
        from notebooks.qa.mas._scorer_helpers import check_taxonomy_whitelist_compliance

        whitelist = {("Macro", "liquidity_contraction"): ["bearish"]}
        extraction = {
            "treatment_class": "Unknown.x",
            "mechanism": "mystery_mechanism",
            "expected_direction": "bearish",
        }
        assert check_taxonomy_whitelist_compliance(extraction, whitelist) is False

    def test_check_taxonomy_whitelist_compliance_none_extraction(self):
        from notebooks.qa.mas._scorer_helpers import check_taxonomy_whitelist_compliance

        assert check_taxonomy_whitelist_compliance(None, {}) is False

    def test_check_routing_legality_cold_start_only_careful_is_legal(self):
        from notebooks.qa.mas._scorer_helpers import check_routing_legality

        assert check_routing_legality("careful", cold_start_mode=True,
                                      top_similarity=0.0, n_examples=5) is True
        assert check_routing_legality("fast", cold_start_mode=True,
                                      top_similarity=0.95, n_examples=5) is False

    def test_check_routing_legality_warm_start_rules(self):
        from notebooks.qa.mas._scorer_helpers import check_routing_legality

        # fast: high sim, >= 2 examples
        assert check_routing_legality("fast", cold_start_mode=False,
                                      top_similarity=0.9, n_examples=3) is True
        # fast illegal when sim too low
        assert check_routing_legality("fast", cold_start_mode=False,
                                      top_similarity=0.7, n_examples=3) is False
        # careful legal when sim low
        assert check_routing_legality("careful", cold_start_mode=False,
                                      top_similarity=0.3, n_examples=3) is True
        # standard in the middle zone
        assert check_routing_legality("standard", cold_start_mode=False,
                                      top_similarity=0.7, n_examples=3) is True

    def test_evidence_grounding_ratio(self):
        from notebooks.qa.mas._scorer_helpers import evidence_grounding_ratio

        body = "The Fed raised rates by 75 bps today, citing inflation."
        spans = ["Fed raised rates by 75 bps", "citing inflation"]
        assert evidence_grounding_ratio(spans, body) == 1.0

    def test_evidence_grounding_ratio_partial(self):
        from notebooks.qa.mas._scorer_helpers import evidence_grounding_ratio

        body = "The Fed raised rates by 75 bps today."
        spans = ["Fed raised rates by 75 bps", "imagined text that is not there"]
        assert evidence_grounding_ratio(spans, body) == 0.5

    def test_evidence_grounding_ratio_no_spans_is_zero(self):
        from notebooks.qa.mas._scorer_helpers import evidence_grounding_ratio

        assert evidence_grounding_ratio([], "body") == 0.0

    def test_evidence_grounding_ratio_casefold_match(self):
        from notebooks.qa.mas._scorer_helpers import evidence_grounding_ratio

        body = "The FED RAISED rates by 75 bps."
        spans = ["fed raised"]
        assert evidence_grounding_ratio(spans, body) == 1.0
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestScorerHelpers -v
```

Expected: `ModuleNotFoundError: No module named 'notebooks.qa.mas._scorer_helpers'`.

- [ ] **Step 3: Implement the scorer helper module**

Create `notebooks/qa/mas/_scorer_helpers.py`:

```python
"""Pure-Python scorer predicates for the MAS deep-dive research notebook.

Extracted from the notebook so they can be unit-tested without loading
Databricks dependencies. The notebook wraps each of these in an
`mlflow.genai.scorer` decorator.
"""
from __future__ import annotations


_REQUIRED_FIELDS = (
    "treatment_class",
    "mechanism",
    "expected_direction",
    "confidence",
    "trigger_event",
    "evidence_spans",
)
_VALID_DIRECTIONS = {"bullish", "bearish", "volatile"}


def check_schema_conformance(extraction: dict | None) -> float:
    """Fraction of schema conformance checks that pass, in [0, 1]."""
    if not isinstance(extraction, dict):
        return 0.0

    checks: list[bool] = []
    for field in _REQUIRED_FIELDS:
        checks.append(field in extraction and extraction[field] is not None)

    direction = extraction.get("expected_direction")
    checks.append(isinstance(direction, str) and direction in _VALID_DIRECTIONS)

    confidence = extraction.get("confidence")
    checks.append(isinstance(confidence, (int, float)) and 0.0 <= float(confidence) <= 1.0)

    spans = extraction.get("evidence_spans")
    checks.append(isinstance(spans, list) and all(isinstance(s, str) for s in spans))

    treatment_class = extraction.get("treatment_class")
    checks.append(isinstance(treatment_class, str) and len(treatment_class) > 0)

    return sum(1 for c in checks if c) / len(checks)


def check_taxonomy_whitelist_compliance(
    extraction: dict | None,
    whitelist: dict[tuple[str, str], list[str]],
) -> bool:
    """True iff (L1, mechanism) is in whitelist AND direction is in allowed list."""
    if not isinstance(extraction, dict):
        return False
    treatment_class = extraction.get("treatment_class") or ""
    l1 = treatment_class.split(".", maxsplit=1)[0] if "." in treatment_class else treatment_class
    mechanism = extraction.get("mechanism")
    direction = extraction.get("expected_direction")
    if not l1 or not mechanism or not direction:
        return False
    allowed = whitelist.get((l1, mechanism))
    if allowed is None:
        return False
    return direction in allowed


def check_routing_legality(
    path: str | None,
    *,
    cold_start_mode: bool,
    top_similarity: float,
    n_examples: int,
    high_threshold: float = 0.85,
    low_threshold: float = 0.50,
    min_examples_for_fast: int = 2,
) -> bool:
    """True iff the observed extraction_path matches the spec routing rule.

    - cold_start: only `careful` is legal
    - warm start:
        * fast iff top_sim >= high AND n_examples >= min_examples_for_fast
        * careful iff top_sim < low OR n_examples < min_examples_for_fast
        * standard otherwise
    """
    if cold_start_mode:
        return path == "careful"

    can_fast = top_similarity >= high_threshold and n_examples >= min_examples_for_fast
    must_careful = top_similarity < low_threshold or n_examples < min_examples_for_fast
    if can_fast:
        return path == "fast"
    if must_careful:
        return path == "careful"
    return path == "standard"


def evidence_grounding_ratio(evidence_spans: list[str] | None, body: str | None) -> float:
    """Fraction of evidence spans that appear (casefold) in the article body."""
    if not evidence_spans:
        return 0.0
    if not body:
        return 0.0
    body_cf = body.casefold()
    hits = sum(1 for span in evidence_spans if isinstance(span, str) and span.strip() and span.casefold() in body_cf)
    return hits / len(evidence_spans)
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestScorerHelpers -v
```

Expected: 14 passed.

- [ ] **Step 5: Add structural test for the notebook's predict_fn + scorer cell**

Add to `TestNotebookStructure`:

```python
    def test_has_predict_fn_and_scorers_section(self):
        content = _notebook_text()
        assert "## §6 predict_fn + custom scorers" in content
        assert "def predict_fn(" in content
        assert "from notebooks.qa.mas._scorer_helpers import" in content
        assert "@scorer" in content
        assert "def schema_conformance(" in content
        assert "def taxonomy_whitelist_compliance(" in content
        assert "make_routing_legality_scorer" in content
        assert "def evidence_grounding_substring(" in content
        assert "def extraction_path_label(" in content
        assert "make_judge" in content
        assert 'name="groundedness"' in content
        assert "load_whitelist" in content
```

- [ ] **Step 6: Run the structural test and verify it fails**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure::test_has_predict_fn_and_scorers_section -v
```

Expected: AssertionError.

- [ ] **Step 7: Append the §6 cell to the notebook**

Append (inside the `with mlflow_component_run(...):` block):

```python

    # COMMAND ----------

    # MAGIC %md
    # MAGIC ## §6 predict_fn + custom scorers
    # MAGIC `predict_fn` never raises — row-level failures return structured error
    # MAGIC dicts so `evaluate()` completes the full sample. Six scorers are
    # MAGIC defined: 4 structural, 1 categorical path label, 1 LLM-as-judge.

    # COMMAND ----------

    from mlflow.genai import scorer
    from mlflow.genai.judges import make_judge

    from notebooks.qa.mas._scorer_helpers import (
        check_routing_legality,
        check_schema_conformance,
        check_taxonomy_whitelist_compliance,
        evidence_grounding_ratio,
    )
    from trading_databricks.agents.taxonomy import load_whitelist

    # Whitelist loaded once, reused by every scorer row
    try:
        _whitelist = load_whitelist(get_spark(), catalog=config.catalog, schema=config.schema)
    except Exception as e:
        _display(f"load_whitelist failed ({e!r}); using empty whitelist")
        _whitelist = {}

    def predict_fn(*, headline, body, published_at, story_id):
        state = {"headline": headline, "body": body, "published_at": published_at}
        try:
            out = graph.invoke(state)
        except Exception as e:
            return {
                "error": repr(e),
                "story_id": story_id,
                "extraction": None,
                "extraction_path": None,
                "validation_status": None,
                "evidence_relevance_score": None,
                "few_shot_examples": [],
            }
        return {
            "error": None,
            "story_id": story_id,
            "extraction": out.get("extraction"),
            "extraction_path": out.get("extraction_path"),
            "validation_status": out.get("validation_status"),
            "evidence_relevance_score": out.get("evidence_relevance_score"),
            "few_shot_examples": out.get("few_shot_examples", []),
        }

    @scorer
    def schema_conformance(outputs):
        return check_schema_conformance(outputs.get("extraction"))

    @scorer
    def taxonomy_whitelist_compliance(outputs):
        return check_taxonomy_whitelist_compliance(outputs.get("extraction"), _whitelist)

    def make_routing_legality_scorer(cold_start_mode_flag: bool, cfg):
        @scorer
        def routing_legality(outputs):
            examples = outputs.get("few_shot_examples") or []
            sims = [ex.get("similarity_score") or 0.0 for ex in examples]
            top_sim = max(sims) if sims else 0.0
            return check_routing_legality(
                outputs.get("extraction_path"),
                cold_start_mode=cold_start_mode_flag,
                top_similarity=top_sim,
                n_examples=len(examples),
                high_threshold=cfg.router.high_similarity_threshold,
                low_threshold=cfg.router.low_similarity_threshold,
                min_examples_for_fast=cfg.router.min_vs_results,
            )
        return routing_legality

    @scorer
    def evidence_grounding_substring(inputs, outputs):
        extraction = outputs.get("extraction") or {}
        spans = extraction.get("evidence_spans") or []
        return evidence_grounding_ratio(spans, inputs.get("body", ""))

    @scorer
    def extraction_path_label(outputs):
        return outputs.get("extraction_path") or "unknown"

    groundedness_judge = make_judge(
        name="groundedness",
        instructions=(
            "Given the article body (in {{ inputs }}) and the extraction "
            "(in {{ outputs }}), rate whether the causal chain is supported by "
            "the body. Return 'faithful' if every claim is clearly backed by "
            "evidence; 'partial' if some claims are supported but others drift; "
            "'hallucinated' if the extraction introduces facts not present in "
            "the body."
        ),
        feedback_value_type="categorical",
        model=f"endpoints:/{config.llm_endpoint}",
        embedding_model=f"endpoints:/{config.validator.embedding_model}",
    )
```

- [ ] **Step 8: Run all tests**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py -v
```

Expected: all green (structural + scorer helpers).

- [ ] **Step 9: Commit**

```bash
git add notebooks/qa/mas/e2e_mas_deep_dive_research.py notebooks/qa/mas/_scorer_helpers.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN feat: add §6 predict_fn, 5 custom scorers, and groundedness judge"
```

---

## Task 13: Add §7 `mlflow.genai.evaluate()` call

**Files:**
- Modify: `notebooks/qa/mas/e2e_mas_deep_dive_research.py`
- Modify: `tests/notebooks/test_e2e_mas_deep_dive_research.py`

- [ ] **Step 1: Add the structural test**

Add to `TestNotebookStructure`:

```python
    def test_has_evaluate_call_section(self):
        content = _notebook_text()
        assert "## §7 mlflow.genai.evaluate()" in content
        assert "mlflow.genai.evaluate(" in content
        assert "data=eval_dataset" in content
        assert "predict_fn=predict_fn" in content
        assert "scorers=[" in content
        assert "schema_conformance" in content
        assert "groundedness_judge" in content
        assert "negative_control" in content
```

- [ ] **Step 2: Run test and verify it fails**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure::test_has_evaluate_call_section -v
```

Expected: AssertionError.

- [ ] **Step 3: Append the §7 cell**

Append:

```python

    # COMMAND ----------

    # MAGIC %md
    # MAGIC ## §7 mlflow.genai.evaluate()
    # MAGIC One call drives the whole cohort through the compiled graph. A second
    # MAGIC nested `evaluate()` handles the off-domain negative control.

    # COMMAND ----------

    scorer_list = [
        schema_conformance,
        taxonomy_whitelist_compliance,
        make_routing_legality_scorer(cold_start_mode, config),
        evidence_grounding_substring,
        extraction_path_label,
        groundedness_judge,
    ]

    eval_result = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=predict_fn,
        scorers=scorer_list,
    )

    mlflow.log_metric("eval.main.n_rows", len(eval_dataset))

    # Negative-control nested run
    if negative_control_dataset:
        with mlflow.start_run(run_name="negative_control", nested=True):
            mlflow.set_tag("slice", "negative_control")
            neg_eval_result = mlflow.genai.evaluate(
                data=negative_control_dataset,
                predict_fn=predict_fn,
                scorers=scorer_list,
            )
            mlflow.log_metric("eval.negative_control.n_rows", len(negative_control_dataset))
    else:
        neg_eval_result = None
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add notebooks/qa/mas/e2e_mas_deep_dive_research.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN feat: add §7 mlflow.genai.evaluate() call with negative control"
```

---

## Task 14: Add §8 trace-derived latency analysis cell

**Files:**
- Modify: `notebooks/qa/mas/e2e_mas_deep_dive_research.py`
- Modify: `tests/notebooks/test_e2e_mas_deep_dive_research.py`

- [ ] **Step 1: Add the structural test**

```python
    def test_has_trace_analysis_section(self):
        content = _notebook_text()
        assert "## §8 Trace-derived analysis" in content
        assert "mlflow.search_traces" in content
        assert 'extract_fields=["spans"]' in content
        assert "node_latency_stats" in content
        assert "span_metrics_raw" in content
        # no fabricated latency numbers
        assert "np.random" not in content
        assert "random.random()" not in content
        assert "time.time() - t0) * 10" not in content
```

- [ ] **Step 2: Run test and verify it fails**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure::test_has_trace_analysis_section -v
```

Expected: AssertionError.

- [ ] **Step 3: Append the §8 cell**

Append:

```python

    # COMMAND ----------

    # MAGIC %md
    # MAGIC ## §8 Trace-derived analysis — real latencies, no fabrication
    # MAGIC All per-node metrics derive from real spans recorded by MLflow during
    # MAGIC the `evaluate()` call above.

    # COMMAND ----------

    from notebooks.qa.utils import NODE_NAMES, build_box

    run_id = mlflow.active_run().info.run_id
    traces_df_raw = mlflow.search_traces(run_id=run_id, extract_fields=["spans"])

    flat_rows: list[dict] = []
    for _, trace_row in traces_df_raw.iterrows():
        spans = trace_row.get("spans") or []
        trace_id = trace_row.get("trace_id") or trace_row.get("request_id")
        for span in spans:
            name = getattr(span, "name", None) or (span.get("name") if isinstance(span, dict) else None)
            if name not in NODE_NAMES:
                continue
            start = getattr(span, "start_time_ns", None) or (span.get("start_time_ns") if isinstance(span, dict) else None)
            end = getattr(span, "end_time_ns", None) or (span.get("end_time_ns") if isinstance(span, dict) else None)
            if start is None or end is None:
                continue
            status = getattr(span, "status_code", None) or (span.get("status_code") if isinstance(span, dict) else "UNKNOWN")
            flat_rows.append({
                "trace_id": str(trace_id),
                "node": name,
                "latency_ms": (int(end) - int(start)) / 1_000_000.0,
                "status": str(status),
            })

    flat_span_df = pl.DataFrame(flat_rows) if flat_rows else pl.DataFrame(
        schema={"trace_id": pl.Utf8, "node": pl.Utf8, "latency_ms": pl.Float64, "status": pl.Utf8}
    )
    log_diag("span_metrics_raw", flat_span_df)

    if flat_span_df.height > 0:
        node_latency_stats = (
            flat_span_df
            .group_by(["node", "status"])
            .agg([
                pl.len().alias("n_invocations"),
                pl.col("latency_ms").median().alias("p50_latency_ms"),
                pl.col("latency_ms").quantile(0.95).alias("p95_latency_ms"),
                pl.col("latency_ms").quantile(0.99).alias("p99_latency_ms"),
            ])
            .sort(["node", "status"])
        )
        log_diag("node_latency_stats", node_latency_stats)
        log_plotly("node_latency_boxplot_real", build_box(flat_span_df, x="node", y="latency_ms"))
        error_count = int(flat_span_df.filter(pl.col("status") != "OK").height)
        mlflow.log_metric("errors.total", error_count)
    else:
        _display("No node spans extracted — check autolog + build_instrumented_mas_graph")
        mlflow.log_metric("errors.total", 0)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add notebooks/qa/mas/e2e_mas_deep_dive_research.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN feat: add §8 trace-derived real-latency analysis cell"
```

---

## Task 15: Add §9 execution-overlay topology diagram

**Files:**
- Modify: `notebooks/qa/mas/e2e_mas_deep_dive_research.py`
- Modify: `tests/notebooks/test_e2e_mas_deep_dive_research.py`

- [ ] **Step 1: Add the structural test**

```python
    def test_has_execution_overlay_section(self):
        content = _notebook_text()
        assert "## §9 Execution-overlay" in content
        assert "topology_overlay" in content
        assert "log_plotly" in content
```

- [ ] **Step 2: Run test and verify it fails**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure::test_has_execution_overlay_section -v
```

Expected: AssertionError.

- [ ] **Step 3: Append the §9 cell**

Append:

```python

    # COMMAND ----------

    # MAGIC %md
    # MAGIC ## §9 Execution-overlay topology diagram
    # MAGIC Plotly directed graph: node size ∝ throughput, color = median latency,
    # MAGIC annotation shows n + p50 + error count.

    # COMMAND ----------

    import plotly.graph_objects as go

    if flat_span_df.height > 0:
        overlay_stats = (
            flat_span_df
            .group_by("node")
            .agg([
                pl.len().alias("n"),
                pl.col("latency_ms").median().alias("p50_ms"),
                (pl.col("status") != "OK").sum().alias("errors"),
            ])
        ).to_dicts()

        overlay_stats_by_node = {row["node"]: row for row in overlay_stats}
        x_positions = list(range(len(NODE_NAMES)))

        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        for i, node in enumerate(NODE_NAMES):
            s = overlay_stats_by_node.get(node, {"n": 0, "p50_ms": 0.0, "errors": 0})
            node_x.append(float(i))
            node_y.append(0.0)
            node_text.append(
                f"<b>{node}</b><br>n={s['n']}<br>p50={s['p50_ms']:.1f}ms<br>errors={s['errors']}"
            )
            node_size.append(max(20, int(s["n"])))
            node_color.append(float(s["p50_ms"] or 0.0))

        edge_x = []
        edge_y = []
        for i in range(len(NODE_NAMES) - 1):
            edge_x += [float(i), float(i + 1), None]
            edge_y += [0.0, 0.0, None]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=2, color="#a2aaad"),
            hoverinfo="none", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(
                size=node_size, color=node_color, colorscale="Viridis",
                colorbar=dict(title="p50 latency (ms)"), showscale=True,
            ),
            text=list(NODE_NAMES),
            textposition="top center",
            hovertext=node_text,
            hoverinfo="text",
            showlegend=False,
        ))
        fig.update_layout(
            title="MAS graph execution overlay",
            xaxis=dict(showticklabels=False, zeroline=False),
            yaxis=dict(showticklabels=False, zeroline=False, range=[-1, 1]),
        )
        log_plotly("topology_overlay", fig)
    else:
        _display("§9 skipped: no span data available for overlay")
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add notebooks/qa/mas/e2e_mas_deep_dive_research.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN feat: add §9 execution-overlay topology diagram"
```

---

## Task 16: Add §10 scorer aggregation visualizations

**Files:**
- Modify: `notebooks/qa/mas/e2e_mas_deep_dive_research.py`
- Modify: `tests/notebooks/test_e2e_mas_deep_dive_research.py`

- [ ] **Step 1: Add the structural test**

```python
    def test_has_scorer_aggregation_section(self):
        content = _notebook_text()
        assert "## §10 Scorer result aggregation" in content
        assert "eval_metrics_table" in content
        assert "build_path_distribution" in content
        assert "build_label_distribution" in content
```

- [ ] **Step 2: Run test and verify it fails**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure::test_has_scorer_aggregation_section -v
```

Expected: AssertionError.

- [ ] **Step 3: Append the §10 cell**

Append:

```python

    # COMMAND ----------

    # MAGIC %md
    # MAGIC ## §10 Scorer result aggregation & visualization
    # MAGIC Pulls scorer outputs into Polars, logs per-scorer pass rates and plots.

    # COMMAND ----------

    from notebooks.qa.utils import build_label_distribution, build_path_distribution

    try:
        eval_metrics_df = pl.from_pandas(eval_result.tables["eval_results_table"])
    except Exception as e:
        _display(f"eval_result.tables lookup failed ({e!r}); falling back to eval_result.metrics")
        eval_metrics_df = pl.DataFrame([eval_result.metrics or {}])

    log_diag("eval_metrics_table", eval_metrics_df)

    # extraction_path distribution
    if "extraction_path" in eval_metrics_df.columns:
        log_plotly(
            "extraction_path_distribution",
            build_path_distribution(eval_metrics_df["extraction_path"]),
        )

    # validation_status distribution
    if "validation_status" in eval_metrics_df.columns:
        log_plotly(
            "validation_status_distribution",
            build_label_distribution(eval_metrics_df["validation_status"]),
        )

    # groundedness judge categorical distribution
    judge_col_candidates = [c for c in eval_metrics_df.columns if "groundedness" in c]
    if judge_col_candidates:
        log_plotly(
            "groundedness_judge_distribution",
            build_label_distribution(eval_metrics_df[judge_col_candidates[0]]),
        )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add notebooks/qa/mas/e2e_mas_deep_dive_research.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN feat: add §10 scorer result aggregation and visualization"
```

---

## Task 17: Add §11 failure-autopsy sandbox with widgets

**Files:**
- Modify: `notebooks/qa/mas/e2e_mas_deep_dive_research.py`
- Modify: `tests/notebooks/test_e2e_mas_deep_dive_research.py`

- [ ] **Step 1: Add the structural test**

```python
    def test_has_autopsy_sandbox_section(self):
        content = _notebook_text()
        assert "## §11 Failure autopsy sandbox" in content
        assert "def render_autopsy(" in content
        assert "def replay_single(" in content
        assert "_get_dbutils()" in content
        assert "dbu.widgets.text" in content
        assert "nested=True" in content
```

- [ ] **Step 2: Run test and verify it fails**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure::test_has_autopsy_sandbox_section -v
```

Expected: AssertionError.

- [ ] **Step 3: Append the §11 cell**

Append:

```python

    # COMMAND ----------

    # MAGIC %md
    # MAGIC ## §11 Failure autopsy sandbox
    # MAGIC Gracefully degrades when `dbutils` is unavailable (i.e. `uv run python`).

    # COMMAND ----------

    dbu = _get_dbutils()
    if dbu is not None:
        dbu.widgets.text("story_id", "", "story_id to replay")
        dbu.widgets.text("override_headline", "", "override headline (optional)")
        dbu.widgets.text("override_body", "", "override body (optional)")

    _sample_by_story = {r["story_id"]: r for r in sample.to_dicts()}

    def render_autopsy(story_id: str) -> None:
        """Render the trace for a given story_id: prompts, LLM outputs, validator verdict."""
        match = flat_span_df.filter(pl.col("trace_id").str.contains(story_id)) if flat_span_df.height > 0 else pl.DataFrame()
        if match.height == 0:
            _display(f"no span rows for story_id={story_id}")
            return
        _display(match)
        _display(
            f"Open the MLflow run UI and search traces for story_id={story_id} "
            f"to inspect full prompt + response payloads."
        )

    def replay_single(story_id: str, *, state_overrides: dict | None = None) -> None:
        row = _sample_by_story.get(story_id)
        if row is None:
            _display(f"story_id={story_id} not in sample")
            return
        state = {
            "headline": row["headline"],
            "body": row["body"],
            "published_at": str(row["published_at"]),
        }
        if state_overrides:
            state.update(state_overrides)
        with mlflow.start_run(run_name=f"replay-{story_id}", nested=True) as nested:
            mlflow.set_tag("replay", "true")
            mlflow.set_tag("replay_story_id", story_id)
            try:
                out = graph.invoke(state)
                _display(out)
                mlflow.log_dict(out, "replay/output.json")
            except Exception as e:
                _display(f"replay failed: {e!r}")
                mlflow.set_tag("replay_status", "fail")
                mlflow.log_text(repr(e), "replay/error.txt")

    # Top-K worst cases by groundedness (if the judge column exists)
    if judge_col_candidates:
        worst_col = judge_col_candidates[0]
        try:
            worst = (
                eval_metrics_df
                .sort(worst_col)
                .head(5)
                .select(["story_id"] if "story_id" in eval_metrics_df.columns else eval_metrics_df.columns[:3])
            )
            _display("Top-K worst by groundedness:")
            _display(worst)
        except Exception as e:
            _display(f"worst-case ranking skipped: {e!r}")

    # Demonstrate the sandbox on the first story_id
    if eval_dataset:
        first_story_id = eval_dataset[0]["inputs"]["story_id"]
        _display(f"Example autopsy for {first_story_id}:")
        render_autopsy(first_story_id)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add notebooks/qa/mas/e2e_mas_deep_dive_research.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN feat: add §11 failure-autopsy sandbox with widgets and replay"
```

---

## Task 18: Add §12 human-feedback hooks (Workflow-7 seed)

**Files:**
- Modify: `notebooks/qa/mas/e2e_mas_deep_dive_research.py`
- Modify: `tests/notebooks/test_e2e_mas_deep_dive_research.py`

- [ ] **Step 1: Add the structural test**

```python
    def test_has_feedback_stubs_section(self):
        content = _notebook_text()
        assert "## §12 Human feedback hooks" in content
        assert "mlflow.log_feedback" in content
        assert "AssessmentSource" in content
        assert "sme_verdict" in content
```

- [ ] **Step 2: Run test and verify it fails**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure::test_has_feedback_stubs_section -v
```

Expected: AssertionError.

- [ ] **Step 3: Append the §12 cell**

Append:

```python

    # COMMAND ----------

    # MAGIC %md
    # MAGIC ## §12 Human feedback hooks (Workflow-7 seed)
    # MAGIC Stubs an `sme_verdict` feedback slot against every trace so a
    # MAGIC downstream UC-backed labeling session can attach and MemAlign can
    # MAGIC align `groundedness_judge` in a follow-up.

    # COMMAND ----------

    try:
        from mlflow.entities import AssessmentSource, AssessmentSourceType

        unique_trace_ids = (
            flat_span_df["trace_id"].unique().to_list() if flat_span_df.height > 0 else []
        )
        for tid in unique_trace_ids:
            try:
                mlflow.log_feedback(
                    trace_id=tid,
                    name="sme_verdict",
                    value=None,
                    source=AssessmentSource(
                        source_type=AssessmentSourceType.HUMAN,
                        source_id="placeholder_sme",
                    ),
                )
            except Exception as e:
                _display(f"log_feedback failed for {tid}: {e!r}")
        mlflow.log_metric("feedback.sme_verdict_stubs", len(unique_trace_ids))
    except Exception as e:
        _display(f"feedback stub cell skipped: {e!r}")
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add notebooks/qa/mas/e2e_mas_deep_dive_research.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN feat: add §12 human-feedback stubs for judge alignment"
```

---

## Task 19: Add §13 sanity assertions + §14 summary + environment teardown

**Files:**
- Modify: `notebooks/qa/mas/e2e_mas_deep_dive_research.py`
- Modify: `tests/notebooks/test_e2e_mas_deep_dive_research.py`

- [ ] **Step 1: Add the structural test**

```python
    def test_has_assertions_and_summary_section(self):
        content = _notebook_text()
        assert "## §13 Sanity assertions" in content
        assert "assert eval_result" in content
        assert "## §14 Summary" in content
        assert "graph_env_ctx.__exit__(None, None, None)" in content

    def test_no_fabricated_latency_code_anywhere(self):
        """Regression guard for the 99_e2e_graph.py bug we are fixing."""
        content = _notebook_text()
        assert "10.0 + (time.time() - t0) * 10" not in content
        assert "np.random.uniform" not in content
        assert "random.random()" not in content

    def test_no_mock_mechanism_validator(self):
        """Regression guard: _mechanism_validator_node must never be MagicMock-patched in this notebook."""
        content = _notebook_text()
        assert "patch(\"trading_databricks.agents.graph._mechanism_validator_node\")" not in content
        assert "MagicMock" not in content
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py::TestNotebookStructure -v
```

Expected: the three new tests fail.

- [ ] **Step 3: Append §13 and §14 cells + environment teardown**

Append:

```python

    # COMMAND ----------

    # MAGIC %md
    # MAGIC ## §13 Sanity assertions
    # MAGIC Loose invariants only. Research notebooks observe — they do not gate.

    # COMMAND ----------

    assert eval_result is not None, "evaluate() returned None"
    assert eval_result.metrics is not None, "eval_result.metrics missing"
    assert len(eval_dataset) == SAMPLE_SIZE or len(eval_dataset) == news.height, \
        "sample size drift"
    assert any(r["name"].startswith("preflight.") for r in preflight_rows), \
        "no preflight metrics logged"
    if flat_span_df.height > 0:
        unique_traces = flat_span_df["trace_id"].n_unique()
        assert unique_traces >= 1, "no traces recorded"

    # COMMAND ----------

    # MAGIC %md
    # MAGIC ## §14 Summary & next steps
    # MAGIC - MLflow run: see link in the run card above
    # MAGIC - Next: launch a UC-backed labeling session against the `sme_verdict`
    # MAGIC   stubs (Workflow-7), run MemAlign on `groundedness_judge`, feed the
    # MAGIC   eval dataset into GEPA (Workflow-8) against `trading_databricks.agents.prompts`.

    # COMMAND ----------

    summary_payload = {
        "mode": params["mode"],
        "sample_size": len(eval_dataset),
        "cold_start_mode": cold_start_mode,
        "n_unique_traces": int(flat_span_df["trace_id"].n_unique()) if flat_span_df.height > 0 else 0,
        "errors_total": int(flat_span_df.filter(pl.col("status") != "OK").height) if flat_span_df.height > 0 else 0,
    }
    mlflow.log_dict(summary_payload, "summary.json")
    _display(summary_payload)

    # COMMAND ----------

    # Release the wire_graph_environment context opened in §5.
    graph_env_ctx.__exit__(None, None, None)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py -v
```

Expected: all green — structural + helpers + regression guards.

- [ ] **Step 5: Commit**

```bash
git add notebooks/qa/mas/e2e_mas_deep_dive_research.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN feat: add §13 assertions, §14 summary, and env teardown"
```

---

## Task 20: Full offline end-to-end run

**Files:** none (execution only)

- [ ] **Step 1: Run the notebook offline end-to-end**

```bash
MAS_RESEARCH_OFFLINE=1 MAS_RESEARCH_SAMPLE_SIZE=5 MAS_RESEARCH_NEG_CONTROL_SIZE=2 \
  uv run python notebooks/qa/mas/e2e_mas_deep_dive_research.py
```

Expected: script exits 0. If it fails, read the traceback, fix the offending cell, rerun until clean. Common failures and remedies:

- `ImportError: mlflow.genai.evaluate` — add `mlflow>=3.0` to `pyproject.toml`, run `uv sync`, retry. Commit the pyproject change separately with message `TRD-NN chore: bump mlflow for genai.evaluate API`.
- `ModuleNotFoundError: databricks.vector_search` — add `databricks-vectorsearch` to pyproject, `uv sync`, retry. Same separate commit pattern.
- `NotImplementedError: query_vs_index` — `wire_graph_environment` was not entered or patch target drifted; re-check `few_shot_retriever.query_vs_index` module path.
- `load_whitelist` crashes outside `_whitelist` try/except — extend the fallback in the §6 cell.

- [ ] **Step 2: Run the full test suite**

```bash
uv run pytest tests/notebooks/test_e2e_mas_deep_dive_research.py tests/agents -v
```

Expected: 0 failures. The `tests/agents` suite must still pass — we changed no production code, so any regression there means a monkey-patch leaked.

- [ ] **Step 3: Commit any incidental fixes**

```bash
git add notebooks/qa/mas/e2e_mas_deep_dive_research.py notebooks/qa/utils.py notebooks/qa/mas/_scorer_helpers.py tests/notebooks/test_e2e_mas_deep_dive_research.py
git commit -m "TRD-NN fix: offline end-to-end fixes uncovered by first run" || echo "nothing to commit"
```

Use `|| echo "nothing to commit"` so this step succeeds when the offline run was already clean.

---

## Task 21: Gated live end-to-end run

**Files:** none (execution + user confirmation)

The live run spends real LLM tokens against `databricks-claude-sonnet` and depends on Unity Catalog state that may not exist in dev. Do NOT run it without explicit user approval.

- [ ] **Step 1: Ask the user to approve the live run**

Post the following message verbatim to the user and wait for a reply:

> "Offline end-to-end is green. The live run will execute the notebook against real Databricks endpoints with `MAS_RESEARCH_SAMPLE_SIZE=50` (~535k tokens estimated). It requires `treatment_taxonomy` and `mechanism_whitelist` to be seeded in `smart-trading-dev.gold`. Do you want me to proceed?"

- [ ] **Step 2: If approved, run the notebook live**

```bash
uv run python notebooks/qa/mas/e2e_mas_deep_dive_research.py
```

Expected: script exits 0. If a critical preflight fails, the run raises `RuntimeError: preflight critical check failed` — report the specific failed check name to the user and stop. Do NOT set `MAS_RESEARCH_OFFLINE=1` as a workaround; that defeats the point of the live run.

- [ ] **Step 3: Verify MLflow artifacts**

Use Databricks MCP (or workspace UI) to confirm the latest run under `/Shared/trading-databricks-qa/mas/e2e_deep_dive_research` contains:

- `topology/graph.mmd` artifact
- `diagnostics/preflight.parquet`
- `diagnostics/span_metrics_raw.parquet`
- `diagnostics/node_latency_stats.parquet`
- `diagnostics/eval_metrics_table.parquet`
- `plots/node_latency_boxplot_real.html`
- `plots/topology_overlay.html`
- `plots/extraction_path_distribution.html` (if column present)
- `summary.json`
- At least `SAMPLE_SIZE` traces visible under the run's Traces tab
- At least one feedback record per trace with `name=sme_verdict`

- [ ] **Step 4: Commit any final polish**

Only if Step 3 uncovered issues fixable without a rerun, edit the notebook and commit:

```bash
git add notebooks/qa/mas/e2e_mas_deep_dive_research.py
git commit -m "TRD-NN fix: live-mode polish uncovered by end-to-end verification" || echo "nothing to commit"
```

- [ ] **Step 5: Final repository status**

```bash
git status
git log --oneline feat/TRD-NN-mas-e2e-deep-dive-research-notebook ^main
```

Expected: clean working tree; the commit list shows the ordered feature commits from Tasks 0 through 20 (and optionally 21).

---

## Out-of-scope follow-ups (tracked, not implemented here)

These are documented in the spec's "Follow-ups" section. Do not implement them in this branch; open separate tickets when the time comes:

- Workflow-7 judge-alignment notebook consuming the `sme_verdict` feedback slots
- Workflow-8 GEPA prompt-optimization notebook consuming this notebook's eval dataset
- Retrofitting `build_instrumented_mas_graph` + real-span instrumentation onto `01_gatekeeper.py`…`06_mechanism_validator.py`
- Nightly CI job running offline mode with a pytest marker
