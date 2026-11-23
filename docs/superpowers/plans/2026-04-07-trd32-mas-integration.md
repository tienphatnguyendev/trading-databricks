# TRD-32 — MAS Integration: LangGraph Pipeline + UC Tools + SDP Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire all 5 individually-built MAS nodes into a compiled LangGraph `StateGraph`, add the missing `fetch_mechanism_whitelist()` UC tool, and integrate the full pipeline into the SDP gold materialized view via `ai_query()`.

**Architecture:** Closure-bound graph — `build_mas_graph(*, config: MASConfig)` uses lambda closures to bind `MASConfig` to each node function. Zero changes to any "done" node files. A module-private `_mechanism_validator_node` in `graph.py` bridges `validate_extraction()`'s 5-arg signature to LangGraph's `(state) -> dict` contract. SDP follows the `ticks/pipeline.py` pattern: logic in `gold.py`, thin `@dp.materialized_view` wrapper in `pipeline.py`.

**Tech Stack:** LangGraph (`langgraph>=0.3`), MLflow (`mlflow.deployments`), PySpark (`pyspark.sql.functions.pandas_udf`, `ai_query`), Databricks SDP (`pyspark.pipelines as dp`), pytest + `unittest.mock.patch`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/trading_databricks/agents/state.py` | MODIFY | Add `body`, `extraction`, `validation_status`, `rejected_mechanism_raw`, `evidence_relevance_score` |
| `src/trading_databricks/agents/graph.py` | CREATE | `route_article`, `_mechanism_validator_node`, `build_mas_graph` |
| `src/trading_databricks/agents/tools.py` | MODIFY | Add `fetch_mechanism_whitelist()` |
| `src/trading_databricks/agents/gold.py` | CREATE | `news_causal_features_transform()` — gatekeeper UDF + ai_query + field expansion |
| `src/trading_databricks/agents/pipeline.py` | MODIFY | Replace placeholder with `@dp.materialized_view def news_causal_features()` |
| `tests/agents/test_graph.py` | CREATE | Unit + integration tests for graph.py |
| `docs/superpowers/specs/2026-04-07-trd32-mas-integration-design.md` | CREATE | Design spec |
| `docs/coordination/TRD-6-gen-ai-agent-layer.md` | MODIFY | Section 3 → done, Section 5 → in_progress |

---

## Task 1: Write design spec + set up worktree

**Files:**
- Create: `docs/superpowers/specs/2026-04-07-trd32-mas-integration-design.md`

- [ ] **Step 1: Create git worktree and branch**

```bash
cd /Users/aaronng/repos/trading-databricks
git worktree add ../trading-databricks-trd32 -b feat/TRD-32-w4-wire-langgraph-pipeline
cd ../trading-databricks-trd32
```

Expected: worktree created at `../trading-databricks-trd32`, new branch `feat/TRD-32-w4-wire-langgraph-pipeline` checked out.

- [ ] **Step 2: Write design spec**

Create `docs/superpowers/specs/2026-04-07-trd32-mas-integration-design.md`:

```markdown
# TRD-32 — MAS Integration Design Spec

**Date:** 2026-04-07  
**Status:** Approved  
**Linear:** TRD-32 — [W4] Wire complete 6-node LangGraph pipeline + UC function tools + SDP integration

## Context

TRD-32 is the integration step for TRD-6 (Build Gen-AI Agent Layer). All 5 LangGraph nodes
(context_retriever, few_shot_retriever, uncertainty_router, causality_extractor,
mechanism_validator) are individually implemented and merged (TRD-27–31). This ticket
assembles them into a compiled StateGraph, adds the missing UC tool, and wires the pipeline
into the SDP gold materialized view.

## Architecture

### Approach: Closure-bound graph (Approach A)

`build_mas_graph(*, config: MASConfig)` uses lambda closures to bind `MASConfig` to each
existing node function. Zero changes to "done" node files. `_mechanism_validator_node`
(module-private in graph.py) bridges `validate_extraction()`'s 5-arg signature to LangGraph's
`(state) -> dict` contract.

### Graph topology

```
ENTRY → context_retriever → few_shot_retriever → uncertainty_router
       ↓ conditional: route_article (V1: always → causality_extractor)
       causality_extractor → mechanism_validator → END
```

`route_article` returns `"causality_extractor"` for all paths (fast/standard/careful).
The `extraction_path` field in state drives path-based behaviour inside `causality_extractor`.
The conditional edge is kept for future extensibility.

### SDP pattern

Follows `ticks/pipeline.py`: logic in `agents/gold.py`, thin wrapper in `agents/pipeline.py`.

## State extension

Add to `MASState` TypedDict:
- `body: str` — article body text (input alongside headline)
- `extraction: dict | None` — LLM output from causality_extractor
- `validation_status: str | None` — "accepted" | "rejected_*"
- `rejected_mechanism_raw: str | None` — raw mechanism string when rejected
- `evidence_relevance_score: float | None` — Module B cosine score

## UC Tools

`tools.py` already has `retrieve_few_shot_chains()` and `fetch_market_context()`.
Add `fetch_mechanism_whitelist()` — reads `gold.mechanism_whitelist` as JSON array string.

## Gold Materialized View

`agents/gold.py :: news_causal_features_transform(spark, *, catalog, gatekeeper_config)`:
1. Gatekeeper pandas UDF calling `is_crypto_relevant()` per row
2. Filter `{catalog}.silver.news_cleaned`
3. `ai_query('mas-news-causal', to_json(struct(headline, body, published_at)))` column
4. Expand JSON causal_result into CAUSAL_FEATURES_SCHEMA fields

## Acceptance Criteria

- `build_mas_graph()` compiles without error
- End-to-end: 10 sample articles through full graph → outputs valid MASState
- All 3 UC function tools callable independently
- `uv run pytest tests/agents/test_graph.py` passes (mock LLM + mock VS)
```

- [ ] **Step 3: Update coordination doc**

In `docs/coordination/TRD-6-gen-ai-agent-layer.md`, update two sections:

Section 3 (RAG + Extractor): change `in_progress` → `done`:
```
- **Status:** done (TRD-28 done, TRD-29 done)
- **Branch:** main (both merged)
```

Section 5 (Integration + Deploy + Eval): change `not_started` → `in_progress`:
```
- **Status:** in_progress
- **Branch:** feat/TRD-32-w4-wire-langgraph-pipeline
- **Modified files:** (updating as work progresses)
- **Exports (public APIs):** agents/graph.py: `build_mas_graph`, `route_article`
- **Needs from others:** All exports from Sections 1–4 ✅
- **Notes:** TRD-32 in progress. Implements graph.py, tools.py addition, gold.py, pipeline.py.
```

- [ ] **Step 4: Commit spec and coordination update**

```bash
git add docs/superpowers/specs/2026-04-07-trd32-mas-integration-design.md
git add docs/coordination/TRD-6-gen-ai-agent-layer.md
git commit -m "TRD-32 docs: add integration design spec and update coordination doc"
```

---

## Task 2: Extend MASState

**Files:**
- Modify: `src/trading_databricks/agents/state.py`
- Test: `tests/agents/test_state.py`

- [ ] **Step 1: Write the failing test**

In `tests/agents/test_state.py`, add (after existing tests):

```python
def test_mas_state_has_body_and_extraction_fields():
    """state.py must have body, extraction, and validator output fields."""
    state: MASState = {
        "headline": "Test",
        "body": "Full article text here.",
        "extraction": {"treatment_class": "Regulation.etf_ruling"},
        "validation_status": "accepted",
        "rejected_mechanism_raw": None,
        "evidence_relevance_score": 0.87,
    }
    assert state["body"] == "Full article text here."
    assert state["extraction"]["treatment_class"] == "Regulation.etf_ruling"
    assert state["validation_status"] == "accepted"
    assert state["evidence_relevance_score"] == pytest.approx(0.87)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/agents/test_state.py::test_mas_state_has_body_and_extraction_fields -v
```

Expected: `FAILED` — `body` key triggers a `TypedDict` type error or the test runs but highlights missing annotations. (TypedDict with `total=False` allows any key at runtime, so the test may pass immediately — if it does, proceed directly.)

- [ ] **Step 3: Add fields to MASState**

Replace content of `src/trading_databricks/agents/state.py`:

```python
"""Centralized LangGraph state definitions for MAS agents."""
from __future__ import annotations

from typing import TypedDict


class MASState(TypedDict, total=False):
    """Type hint for MAS LangGraph state.

    total=False makes all fields optional, as nodes only interact with
    portions of the state.

    Input fields (caller-supplied):
        headline: Article headline text
        body: Full article body text
        published_at: ISO timestamp string

    Context fields (set by context_retriever):
        market_vol_at_publish, market_trend_at_publish, dollar_bar_close

    Retrieval fields (set by few_shot_retriever):
        few_shot_examples, retrieval_story_ids

    Routing fields (set by uncertainty_router):
        extraction_path

    Extraction fields (set by causality_extractor):
        extraction

    Validation fields (set by mechanism_validator):
        validation_status, rejected_mechanism_raw, evidence_relevance_score
    """

    # --- input ---
    headline: str
    body: str
    published_at: str

    # --- context (context_retriever) ---
    market_vol_at_publish: float | None
    market_trend_at_publish: str | None
    dollar_bar_close: float | None

    # --- retrieval (few_shot_retriever) ---
    few_shot_examples: list[dict]
    retrieval_story_ids: list[str]

    # --- routing (uncertainty_router) ---
    extraction_path: str

    # --- extraction (causality_extractor) ---
    extraction: dict | None

    # --- validation (mechanism_validator) ---
    validation_status: str | None
    rejected_mechanism_raw: str | None
    evidence_relevance_score: float | None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/agents/test_state.py -v
```

Expected: All `test_state.py` tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/agents/state.py tests/agents/test_state.py
git commit -m "TRD-32 feat: extend MASState with body, extraction, and validation fields"
```

---

## Task 3: `route_article` function

**Files:**
- Create: `src/trading_databricks/agents/graph.py` (partial — just route_article)
- Create: `tests/agents/test_graph.py` (partial)

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_graph.py`:

```python
"""Tests for agents/graph.py — LangGraph StateGraph assembly."""
from __future__ import annotations

import pytest

from trading_databricks.agents.state import MASState


def test_route_article_fast_path():
    """route_article returns 'causality_extractor' regardless of extraction_path."""
    from trading_databricks.agents.graph import route_article

    state: MASState = {"headline": "test", "extraction_path": "fast"}
    assert route_article(state) == "causality_extractor"


def test_route_article_standard_path():
    from trading_databricks.agents.graph import route_article

    state: MASState = {"headline": "test", "extraction_path": "standard"}
    assert route_article(state) == "causality_extractor"


def test_route_article_careful_path():
    from trading_databricks.agents.graph import route_article

    state: MASState = {"headline": "test", "extraction_path": "careful"}
    assert route_article(state) == "causality_extractor"


def test_route_article_no_path():
    """Missing extraction_path also routes to causality_extractor."""
    from trading_databricks.agents.graph import route_article

    state: MASState = {"headline": "test"}
    assert route_article(state) == "causality_extractor"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/agents/test_graph.py -v
```

Expected: `ImportError` — `cannot import name 'route_article' from 'trading_databricks.agents.graph'`

- [ ] **Step 3: Create graph.py with route_article**

Create `src/trading_databricks/agents/graph.py`:

```python
"""LangGraph StateGraph assembly for the MAS pipeline.

Wires all 5 nodes into a CompiledStateGraph using closure-bound config.
All transform logic lives in nodes/. This file owns topology only.
"""
from __future__ import annotations

from typing import Literal

from trading_databricks.agents.state import MASState


def route_article(state: MASState) -> Literal["causality_extractor"]:
    """Route from uncertainty_router to the next node.

    V1: all extraction paths (fast/standard/careful) proceed to causality_extractor.
    The extraction_path field is already set in state; causality_extractor reads it
    internally to select temperature/n_choices. The conditional edge is kept explicit
    here for future extensibility (e.g., a trivial path short-circuiting to END).
    """
    return "causality_extractor"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/agents/test_graph.py::test_route_article_fast_path \
              tests/agents/test_graph.py::test_route_article_standard_path \
              tests/agents/test_graph.py::test_route_article_careful_path \
              tests/agents/test_graph.py::test_route_article_no_path -v
```

Expected: All 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/agents/graph.py tests/agents/test_graph.py
git commit -m "TRD-32 feat: add route_article to graph.py with routing tests"
```

---

## Task 4: `_mechanism_validator_node`

**Files:**
- Modify: `src/trading_databricks/agents/graph.py`
- Modify: `tests/agents/test_graph.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/agents/test_graph.py`:

```python
from unittest.mock import MagicMock, patch

from trading_databricks.agents.config import MASConfig


def test_mechanism_validator_node_no_extraction():
    """Returns rejected_evidence when state has no extraction."""
    from trading_databricks.agents.graph import _mechanism_validator_node

    state: MASState = {"headline": "test", "body": "some body text"}
    result = _mechanism_validator_node(state, MASConfig())

    assert result["validation_status"] == "rejected_evidence"
    assert result["rejected_mechanism_raw"] is None
    assert result["evidence_relevance_score"] is None


@patch("trading_databricks.agents.graph.mlflow.deployments.get_deploy_client")
@patch("trading_databricks.agents.graph.load_whitelist")
@patch("trading_databricks.agents.graph.get_spark_session")
def test_mechanism_validator_node_accepted(mock_spark, mock_load_whitelist, mock_get_client):
    """Returns accepted when extraction matches whitelist and evidence found in body."""
    from trading_databricks.agents.graph import _mechanism_validator_node

    mock_load_whitelist.return_value = {
        ("Regulation", "institutional_access"): ["bullish"],
    }

    mock_client = MagicMock()
    mock_client.predict.return_value = {"data": [{"embedding": [0.1] * 384}]}
    mock_get_client.return_value = mock_client

    extraction = {
        "treatment_class": "Regulation.etf_ruling",
        "trigger_event": "SEC approves spot Bitcoin ETF",
        "mechanism": "institutional_access",
        "expected_direction": "bullish",
        "evidence_spans": ["sec approves spot bitcoin etf"],
        "confidence": 0.9,
    }
    body = "SEC approves spot Bitcoin ETF, enabling institutional investment."
    state: MASState = {"headline": "SEC approves ETF", "body": body, "extraction": extraction}

    result = _mechanism_validator_node(state, MASConfig())

    assert result["validation_status"] == "accepted"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/agents/test_graph.py::test_mechanism_validator_node_no_extraction \
              tests/agents/test_graph.py::test_mechanism_validator_node_accepted -v
```

Expected: `ImportError` — `cannot import name '_mechanism_validator_node'`

- [ ] **Step 3: Add _mechanism_validator_node to graph.py**

Replace the entire content of `src/trading_databricks/agents/graph.py`:

```python
"""LangGraph StateGraph assembly for the MAS pipeline.

Wires all 5 nodes into a CompiledStateGraph using closure-bound config.
All transform logic lives in nodes/. This file owns topology only.
"""
from __future__ import annotations

from typing import Literal

import mlflow.deployments

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.nodes.few_shot_retriever import get_spark_session
from trading_databricks.agents.nodes.mechanism_validator import validate_extraction
from trading_databricks.agents.state import MASState
from trading_databricks.agents.taxonomy import load_whitelist


def route_article(state: MASState) -> Literal["causality_extractor"]:
    """Route from uncertainty_router to the next node.

    V1: all extraction paths (fast/standard/careful) proceed to causality_extractor.
    The extraction_path field is already set in state; causality_extractor reads it
    internally to select temperature/n_choices. The conditional edge is kept explicit
    here for future extensibility (e.g., a trivial path short-circuiting to END).
    """
    return "causality_extractor"


def _mechanism_validator_node(state: MASState, config: MASConfig) -> dict:
    """Bridge validate_extraction() to the LangGraph (state) -> dict interface.

    validate_extraction() has a 5-arg signature incompatible with LangGraph nodes.
    This private helper closes over config and injects whitelist + embed_fn at call time.

    Returns dict with keys: validation_status, rejected_mechanism_raw, evidence_relevance_score.
    """
    extraction = state.get("extraction")
    if not extraction:
        return {
            "validation_status": "rejected_evidence",
            "rejected_mechanism_raw": None,
            "evidence_relevance_score": None,
        }

    spark = get_spark_session()
    whitelist = load_whitelist(spark, catalog=config.catalog, schema=config.schema)

    client = mlflow.deployments.get_deploy_client("databricks")

    def embed_fn(text: str) -> list[float]:
        resp = client.predict(
            endpoint=config.validator.embedding_model,
            inputs={"input": [text]},
        )
        return resp["data"][0]["embedding"]

    result = validate_extraction(
        extraction,
        state.get("body", ""),
        whitelist,
        config=config.validator,
        embed_fn=embed_fn,
    )
    return dict(result)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/agents/test_graph.py::test_mechanism_validator_node_no_extraction \
              tests/agents/test_graph.py::test_mechanism_validator_node_accepted -v
```

Expected: Both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/agents/graph.py tests/agents/test_graph.py
git commit -m "TRD-32 feat: add _mechanism_validator_node bridge to graph.py"
```

---

## Task 5: `build_mas_graph` — complete graph.py

**Files:**
- Modify: `src/trading_databricks/agents/graph.py`
- Modify: `tests/agents/test_graph.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/agents/test_graph.py`:

```python
@patch("trading_databricks.agents.nodes.few_shot_retriever.get_spark_session")
@patch("trading_databricks.agents.nodes.few_shot_retriever.get_accepted_chain_count")
def test_build_mas_graph_compiles(mock_count, mock_spark):
    """build_mas_graph() returns a CompiledStateGraph without raising."""
    from langgraph.graph.state import CompiledStateGraph

    from trading_databricks.agents.graph import build_mas_graph

    graph = build_mas_graph(config=MASConfig())
    assert isinstance(graph, CompiledStateGraph)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/agents/test_graph.py::test_build_mas_graph_compiles -v
```

Expected: `ImportError` — `cannot import name 'build_mas_graph'`

- [ ] **Step 3: Add build_mas_graph to graph.py**

Append to `src/trading_databricks/agents/graph.py` (after `_mechanism_validator_node`):

```python

def build_mas_graph(*, config: MASConfig):
    """Compile the 5-node MAS StateGraph.

    Node execution order:
        context_retriever → few_shot_retriever → uncertainty_router
        → [conditional: route_article] → causality_extractor
        → mechanism_validator → END

    All nodes receive MASConfig via lambda closure — zero changes to node files.

    Args:
        config: MASConfig containing nested configs for all nodes.

    Returns:
        CompiledStateGraph ready for .invoke() calls and MLflow logging.
    """
    from langgraph.graph import StateGraph

    from trading_databricks.agents.nodes import (
        causality_extractor,
        context_retriever,
        few_shot_retriever,
        uncertainty_router,
    )

    graph = StateGraph(MASState)

    graph.add_node("context_retriever",   lambda s: context_retriever(s, config=config))
    graph.add_node("few_shot_retriever",  lambda s: few_shot_retriever(s, config))
    graph.add_node("uncertainty_router",  lambda s: uncertainty_router(s, config))
    graph.add_node("causality_extractor", lambda s: causality_extractor(s, config))
    graph.add_node("mechanism_validator", lambda s: _mechanism_validator_node(s, config))

    graph.set_entry_point("context_retriever")
    graph.add_edge("context_retriever",   "few_shot_retriever")
    graph.add_edge("few_shot_retriever",  "uncertainty_router")
    graph.add_conditional_edges(
        "uncertainty_router",
        route_article,
        {"causality_extractor": "causality_extractor"},
    )
    graph.add_edge("causality_extractor", "mechanism_validator")
    graph.set_finish_point("mechanism_validator")

    return graph.compile()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/agents/test_graph.py::test_build_mas_graph_compiles -v
```

Expected: PASS. If `ImportError: cannot import langgraph`, run `uv add langgraph` first.

- [ ] **Step 5: Run all graph tests so far**

```bash
uv run pytest tests/agents/test_graph.py -v
```

Expected: 7 tests PASS (4 route_article + 2 validator_node + 1 compile).

- [ ] **Step 6: Commit**

```bash
git add src/trading_databricks/agents/graph.py tests/agents/test_graph.py
git commit -m "TRD-32 feat: implement build_mas_graph — 5-node CompiledStateGraph"
```

---

## Task 6: `fetch_mechanism_whitelist` in tools.py

**Files:**
- Modify: `src/trading_databricks/agents/tools.py`
- Create: `tests/agents/test_tools.py` (if not exists) or append to existing

- [ ] **Step 1: Check if test_tools.py exists**

```bash
ls tests/agents/test_tools.py 2>/dev/null && echo "exists" || echo "missing"
```

- [ ] **Step 2: Write the failing test**

If `tests/agents/test_tools.py` exists, append. Otherwise create it with:

```python
"""Tests for agents/tools.py — UC function tools."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from trading_databricks.agents.config import MASConfig


@patch("trading_databricks.agents.tools.get_spark_session")
def test_fetch_mechanism_whitelist_returns_json_string(mock_spark):
    """fetch_mechanism_whitelist returns a valid JSON array string."""
    import json

    from trading_databricks.agents.tools import fetch_mechanism_whitelist

    mock_row_1 = MagicMock()
    mock_row_1.asDict.return_value = {
        "treatment_l1": "Regulation",
        "mechanism": "regulatory_clarity",
        "valid_directions": ["bullish"],
        "description": "Clear regulation enables access",
    }
    mock_row_2 = MagicMock()
    mock_row_2.asDict.return_value = {
        "treatment_l1": "Macro",
        "mechanism": "liquidity_contraction",
        "valid_directions": ["bearish"],
        "description": "Tight monetary policy drains risk assets",
    }

    mock_spark_instance = MagicMock()
    mock_spark_instance.table.return_value.collect.return_value = [mock_row_1, mock_row_2]
    mock_spark.return_value = mock_spark_instance

    result = fetch_mechanism_whitelist()

    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["treatment_l1"] == "Regulation"
    assert parsed[1]["mechanism"] == "liquidity_contraction"

    mock_spark_instance.table.assert_called_once_with(
        "trading-signal-gen-dev.gold.mechanism_whitelist"
    )


@patch("trading_databricks.agents.tools.get_spark_session")
def test_fetch_mechanism_whitelist_uses_config_catalog(mock_spark):
    """fetch_mechanism_whitelist respects the catalog from MASConfig."""
    import json

    from trading_databricks.agents.tools import fetch_mechanism_whitelist

    mock_spark_instance = MagicMock()
    mock_spark_instance.table.return_value.collect.return_value = []
    mock_spark.return_value = mock_spark_instance

    custom_config = MASConfig(catalog="my-catalog", schema="gold")
    result = fetch_mechanism_whitelist(config=custom_config)

    assert json.loads(result) == []
    mock_spark_instance.table.assert_called_once_with("my-catalog.gold.mechanism_whitelist")
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/agents/test_tools.py -v
```

Expected: `ImportError` — `cannot import name 'fetch_mechanism_whitelist'`

- [ ] **Step 4: Add fetch_mechanism_whitelist to tools.py**

Open `src/trading_databricks/agents/tools.py` and add at top:

```python
import json
```

(Only add this line if `json` is not already imported.)

Then add this import at the bottom of the existing imports block:

```python
from trading_databricks.agents.nodes.few_shot_retriever import get_spark_session
```

(Only add if `get_spark_session` is not already imported in tools.py — check first.)

Then append the new function after `fetch_market_context`:

```python


def fetch_mechanism_whitelist(*, config: MASConfig | None = None) -> str:
    """Read the mechanism whitelist from gold.mechanism_whitelist as a JSON array.

    This is a UC function tool registered in Mosaic AI Agent Framework.
    Returns the full whitelist so the LLM can reference valid (treatment, mechanism)
    pairs and their allowed directions.

    Args:
        config: MASConfig — catalog/schema determine the table path.
                Uses default MASConfig() if not provided.

    Returns:
        JSON array string: [{treatment_l1, mechanism, valid_directions, description}, ...]

    Example:
        >>> whitelist_json = fetch_mechanism_whitelist()
        >>> import json; rows = json.loads(whitelist_json)
        >>> rows[0]["treatment_l1"]
        'Regulation'
    """
    if config is None:
        config = MASConfig()
    spark = get_spark_session()
    rows = spark.table(f"{config.catalog}.{config.schema}.mechanism_whitelist").collect()
    return json.dumps([row.asDict() for row in rows])
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/agents/test_tools.py -v
```

Expected: Both PASS.

- [ ] **Step 6: Commit**

```bash
git add src/trading_databricks/agents/tools.py tests/agents/test_tools.py
git commit -m "TRD-32 feat: add fetch_mechanism_whitelist UC tool to tools.py"
```

---

## Task 7: `agents/gold.py` — gatekeeper UDF + ai_query

**Files:**
- Create: `src/trading_databricks/agents/gold.py`
- Create: `tests/agents/test_gold.py`

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_gold.py`:

```python
"""Tests for agents/gold.py — news_causal_features_transform."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trading_databricks.agents.config import GatekeeperConfig


def test_gatekeeper_udf_passes_crypto_article():
    """gatekeeper_udf returns True for articles mentioning Bitcoin."""
    from trading_databricks.agents.gold import _make_gatekeeper_udf

    udf_fn = _make_gatekeeper_udf(GatekeeperConfig())
    result = udf_fn(
        pd.Series(["Bitcoin ETF approved by SEC"]),
        pd.Series(["The SEC has approved the first spot Bitcoin ETF."])
    )
    assert result.iloc[0] is True


def test_gatekeeper_udf_rejects_non_crypto_article():
    """gatekeeper_udf returns False for articles with no crypto entities."""
    from trading_databricks.agents.gold import _make_gatekeeper_udf

    udf_fn = _make_gatekeeper_udf(GatekeeperConfig())
    result = udf_fn(
        pd.Series(["Fed raises interest rates"]),
        pd.Series(["The Federal Reserve raised rates by 25bp in a unanimous vote."])
    )
    assert result.iloc[0] is False


def test_news_causal_features_transform_filters_and_adds_ai_query():
    """news_causal_features_transform filters via gatekeeper and adds causal_result column."""
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StringType, StructField, StructType, TimestampType

    from trading_databricks.agents.gold import news_causal_features_transform

    # This test verifies the structure — not the actual Spark execution.
    # Check that the function can be called and returns a DataFrame plan.
    # For structural validation only (no Spark execution).
    spark = MagicMock()

    mock_df = MagicMock()
    mock_df.filter.return_value = mock_df
    mock_df.withColumn.return_value = mock_df
    mock_df.withColumnRenamed.return_value = mock_df
    mock_df.select.return_value = mock_df
    spark.table.return_value = mock_df

    result = news_causal_features_transform(
        spark,
        catalog="trading-signal-gen-dev",
        gatekeeper_config=GatekeeperConfig(),
    )

    spark.table.assert_called_once_with("trading-signal-gen-dev.silver.news_cleaned")
    mock_df.filter.assert_called_once()
    mock_df.withColumn.assert_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/agents/test_gold.py -v
```

Expected: `ImportError` — `cannot import name '_make_gatekeeper_udf' from 'trading_databricks.agents.gold'`

- [ ] **Step 3: Create agents/gold.py**

Create `src/trading_databricks/agents/gold.py`:

```python
"""Gold layer transform for MAS news causal features.

Implements the news_causal_features materialized view logic:
  1. Gatekeeper pandas UDF filters ~95% non-crypto articles
  2. ai_query() calls the mas-news-causal Model Serving endpoint
  3. JSON response expanded into CAUSAL_FEATURES_SCHEMA fields

This module is imported by agents/pipeline.py — no logic lives there.
"""
from __future__ import annotations

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType

from trading_databricks.agents.config import GatekeeperConfig
from trading_databricks.agents.gatekeeper import is_crypto_relevant


def _make_gatekeeper_udf(config: GatekeeperConfig):
    """Return a pandas UDF that filters articles by crypto-relevance.

    The UDF calls is_crypto_relevant() (which caches build_nlp() per process)
    for each row. build_nlp() is @functools.cache — safe for repeated calls.

    Args:
        config: GatekeeperConfig — text_window used for body truncation.

    Returns:
        A pandas_udf(BooleanType()) callable taking (headlines, bodies) Series.
    """

    @F.pandas_udf(BooleanType())
    def gatekeeper_udf(headlines: pd.Series, bodies: pd.Series) -> pd.Series:
        results = []
        for headline, body in zip(headlines, bodies):
            is_impactful, _ = is_crypto_relevant(
                str(headline) if headline is not None else "",
                str(body) if body is not None else "",
                config=config,
            )
            results.append(is_impactful)
        return pd.Series(results)

    return gatekeeper_udf


def news_causal_features_transform(
    spark: SparkSession,
    *,
    catalog: str,
    gatekeeper_config: GatekeeperConfig,
) -> DataFrame:
    """Build the news_causal_features DataFrame plan.

    Reads silver.news_cleaned, filters via gatekeeper, enriches via
    ai_query('mas-news-causal', ...), and expands the JSON response into
    the 23-field CAUSAL_FEATURES_SCHEMA output.

    Args:
        spark: Active SparkSession (Databricks Connect or SDP runtime).
        catalog: Unity Catalog catalog name (e.g., "trading-signal-gen-dev").
        gatekeeper_config: GatekeeperConfig for NER-based filtering.

    Returns:
        DataFrame with schema matching CAUSAL_FEATURES_SCHEMA (23 fields).
        Not yet materialized — caller triggers execution.
    """
    gatekeeper_udf = _make_gatekeeper_udf(gatekeeper_config)

    return (
        spark.table(f"{catalog}.silver.news_cleaned")
        # Stage 1+2 gatekeeper: keep only crypto-impactful articles (~5%)
        .filter(gatekeeper_udf(F.col("headline"), F.col("body")))
        # Call MAS endpoint — returns JSON string with all causal fields
        .withColumn(
            "causal_result",
            F.expr(
                "ai_query('mas-news-causal', to_json(struct(headline, body, published_at)))"
            ),
        )
        # Expand causal_result JSON into individual output columns
        .select(
            F.col("story_id"),
            F.col("published_at"),
            F.lit(True).alias("is_impactful"),
            F.lit(1.0).cast("double").alias("gatekeeper_confidence"),
            F.lit("v1.0").alias("agent_model_version"),
            F.current_timestamp().alias("processed_at"),
            F.get_json_object(F.col("causal_result"), "$.treatment_class").alias("treatment_class"),
            F.get_json_object(F.col("causal_result"), "$.trigger_event").alias("trigger_event"),
            F.get_json_object(F.col("causal_result"), "$.mechanism").alias("mechanism"),
            F.get_json_object(F.col("causal_result"), "$.asset").alias("asset"),
            F.get_json_object(F.col("causal_result"), "$.expected_direction").alias("expected_direction"),
            F.get_json_object(F.col("causal_result"), "$.confidence").cast("double").alias("confidence"),
            F.get_json_object(F.col("causal_result"), "$.causal_chain_json").alias("causal_chain_json"),
            F.from_json(
                F.get_json_object(F.col("causal_result"), "$.evidence_spans"),
                "array<string>",
            ).alias("evidence_spans"),
            F.get_json_object(F.col("causal_result"), "$.validation_status").alias("validation_status"),
            F.get_json_object(F.col("causal_result"), "$.rejected_mechanism_raw").alias("rejected_mechanism_raw"),
            F.from_json(
                F.get_json_object(F.col("causal_result"), "$.retrieval_story_ids"),
                "array<string>",
            ).alias("retrieval_story_ids"),
            F.lit(None).cast("double").alias("market_vol_at_publish"),
            F.lit(None).cast("string").alias("market_trend_at_publish"),
            F.lit(None).cast("decimal(18,8)").alias("dollar_bar_close"),
            F.get_json_object(F.col("causal_result"), "$.extraction_path").alias("extraction_path"),
            F.get_json_object(F.col("causal_result"), "$.evidence_relevance_score").cast("double").alias("evidence_relevance_score"),
            F.lit(None).cast("string").alias("confounding_cluster_id"),
        )
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/agents/test_gold.py -v
```

Expected: All 3 PASS. If any spaCy-related error, ensure `uv run pytest` (not bare `pytest`) so the venv is active.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/agents/gold.py tests/agents/test_gold.py
git commit -m "TRD-32 feat: implement news_causal_features_transform in agents/gold.py"
```

---

## Task 8: Update `agents/pipeline.py`

**Files:**
- Modify: `src/trading_databricks/agents/pipeline.py`

No unit test needed — this is a thin infrastructure wrapper. Validated by `databricks bundle validate` in Task 10.

- [ ] **Step 1: Replace pipeline.py placeholder**

Overwrite `src/trading_databricks/agents/pipeline.py`:

```python
"""SDP pipeline definition for the MAS (agents) domain.

This file contains ONLY @dp.materialized_view decorators.
All transformation logic lives in agents/gold.py and is locally testable.

Pipeline:
  news_causal_features  — materialized view (full recompute, batch)
                          Reads silver.news_cleaned, applies gatekeeper,
                          calls ai_query('mas-news-causal'), expands 23 fields.

DABs pipeline resource: resources/agents_pipeline.pipeline.yml (to be created in TRD-33)

Note: ai_query() requires the mas-news-causal Model Serving endpoint to be live.
      Deploy the endpoint in TRD-33 before running this pipeline in production.
"""
from __future__ import annotations

from pyspark import pipelines as dp

from trading_databricks.agents.config import GatekeeperConfig
from trading_databricks.agents.gold import news_causal_features_transform

_CATALOG = "trading-signal-gen-dev"


@dp.materialized_view
def news_causal_features():
    """Gold materialized view: structured causal features from news articles.

    Applies gatekeeper (NER filter) then calls the MAS endpoint via ai_query.
    Full recompute on each pipeline run — no streaming state.
    """
    return news_causal_features_transform(
        spark,
        catalog=_CATALOG,
        gatekeeper_config=GatekeeperConfig(),
    )
```

- [ ] **Step 2: Verify import is clean**

```bash
uv run python -c "import trading_databricks.agents.pipeline; print('ok')"
```

Expected: `ok` (no errors).

- [ ] **Step 3: Commit**

```bash
git add src/trading_databricks/agents/pipeline.py
git commit -m "TRD-32 feat: implement agents/pipeline.py SDP materialized view"
```

---

## Task 9: End-to-end integration tests

**Files:**
- Modify: `tests/agents/test_graph.py`

- [ ] **Step 1: Write the failing end-to-end tests**

Append to `tests/agents/test_graph.py`:

```python
import json


SAMPLE_ARTICLES = [
    {
        "headline": "SEC approves first spot Bitcoin ETF applications",
        "body": "The SEC approves spot Bitcoin ETF applications from major asset managers, enabling regulated institutional access.",
        "published_at": "2026-01-15T10:00:00Z",
        "expected_treatment_l1": "Regulation",
        "expected_mechanism": "institutional_access",
        "expected_direction": "bullish",
    },
    {
        "headline": "Federal Reserve raises interest rates by 75 basis points",
        "body": "The Federal Reserve raised interest rates by 75 basis points to combat inflation, tightening monetary policy.",
        "published_at": "2026-01-16T14:00:00Z",
        "expected_treatment_l1": "Macro",
        "expected_mechanism": "liquidity_contraction",
        "expected_direction": "bearish",
    },
    {
        "headline": "Major cryptocurrency exchange confirms $480M hack",
        "body": "A major cryptocurrency exchange confirmed a $480M hack and suspended withdrawals, destroying user confidence.",
        "published_at": "2026-01-17T08:00:00Z",
        "expected_treatment_l1": "Security",
        "expected_mechanism": "trust_erosion",
        "expected_direction": "bearish",
    },
    {
        "headline": "Mt. Gox trustee confirms July BTC distribution",
        "body": "Mt. Gox trustee confirms July 2026 BTC distribution to creditors, adding sell pressure to Bitcoin markets.",
        "published_at": "2026-01-18T09:00:00Z",
        "expected_treatment_l1": "Supply",
        "expected_mechanism": "sell_pressure",
        "expected_direction": "bearish",
    },
    {
        "headline": "BlackRock announces Bitcoin treasury allocation",
        "body": "BlackRock announced a Bitcoin treasury allocation strategy, increasing institutional demand for BTC.",
        "published_at": "2026-01-19T11:00:00Z",
        "expected_treatment_l1": "Adoption",
        "expected_mechanism": "demand_increase",
        "expected_direction": "bullish",
    },
    {
        "headline": "Ethereum network completes major protocol upgrade",
        "body": "The Ethereum network completed a major protocol upgrade improving scalability, increasing adoption potential.",
        "published_at": "2026-01-20T15:00:00Z",
        "expected_treatment_l1": "Technology",
        "expected_mechanism": "scalability_improvement",
        "expected_direction": "bullish",
    },
    {
        "headline": "China bans crypto mining operations",
        "body": "China announced a ban on all cryptocurrency mining operations, restricting crypto activity in the region.",
        "published_at": "2026-01-21T06:00:00Z",
        "expected_treatment_l1": "Regulation",
        "expected_mechanism": "regulatory_restriction",
        "expected_direction": "bearish",
    },
    {
        "headline": "Bitcoin ETF sees record $2B inflow",
        "body": "Bitcoin ETF saw a record $2B single-day inflow, signaling strong institutional demand.",
        "published_at": "2026-01-22T12:00:00Z",
        "expected_treatment_l1": "Adoption",
        "expected_mechanism": "demand_increase",
        "expected_direction": "bullish",
    },
    {
        "headline": "Stablecoin issuer Tether freezes $100M in wallets",
        "body": "Tether froze $100M in wallets linked to sanctioned entities, raising concerns about crypto censorship.",
        "published_at": "2026-01-23T07:00:00Z",
        "expected_treatment_l1": "Regulation",
        "expected_mechanism": "regulatory_restriction",
        "expected_direction": "bearish",
    },
    {
        "headline": "Coinbase obtains MiCA license for European operations",
        "body": "Coinbase obtained full MiCA regulatory license enabling compliant crypto trading across the EU.",
        "published_at": "2026-01-24T10:00:00Z",
        "expected_treatment_l1": "Regulation",
        "expected_mechanism": "regulatory_clarity",
        "expected_direction": "bullish",
    },
]

_VALID_WHITELIST = {
    ("Regulation", "institutional_access"): ["bullish"],
    ("Regulation", "regulatory_clarity"): ["bullish"],
    ("Regulation", "regulatory_restriction"): ["bearish"],
    ("Macro", "liquidity_contraction"): ["bearish"],
    ("Security", "trust_erosion"): ["bearish"],
    ("Supply", "sell_pressure"): ["bearish"],
    ("Adoption", "demand_increase"): ["bullish"],
    ("Technology", "scalability_improvement"): ["bullish"],
}


def _make_llm_mock(article: dict) -> dict:
    """Return a valid LLM extraction dict for a sample article."""
    body_snippet = article["body"][:80]
    return {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "treatment_class": f"{article['expected_treatment_l1']}.subcategory",
                    "trigger_event": article["headline"][:100],
                    "mechanism": article["expected_mechanism"],
                    "expected_direction": article["expected_direction"],
                    "confidence": 0.9,
                    "asset": "BTC",
                    "evidence_spans": [body_snippet.lower()[:50]],
                    "reasoning": "Extracted from article.",
                })
            }
        }]
    }


@patch("trading_databricks.agents.graph.mlflow.deployments.get_deploy_client")
@patch("trading_databricks.agents.graph.load_whitelist")
@patch("trading_databricks.agents.graph.get_spark_session")
@patch("trading_databricks.agents.nodes.causality_extractor.mlflow.deployments.get_deploy_client")
@patch("trading_databricks.agents.nodes.causality_extractor.get_spark_session")
@patch("trading_databricks.agents.nodes.causality_extractor.load_taxonomy")
@patch("trading_databricks.agents.nodes.causality_extractor.load_whitelist")
@patch("trading_databricks.agents.nodes.few_shot_retriever.get_accepted_chain_count")
@patch("trading_databricks.agents.nodes.few_shot_retriever.get_spark_session")
def test_end_to_end_fast_path(
    mock_fsr_spark,
    mock_fsr_count,
    mock_ext_whitelist,
    mock_ext_taxonomy,
    mock_ext_spark,
    mock_ext_client,
    mock_val_spark,
    mock_val_whitelist,
    mock_val_client,
):
    """Full graph: single article, fast path (high VS similarity from static examples)."""
    from trading_databricks.agents.graph import build_mas_graph
    from trading_databricks.agents.nodes.few_shot_retriever import STATIC_FEW_SHOT_EXAMPLES

    article = SAMPLE_ARTICLES[0]

    # few_shot_retriever: cold start (count=0) → returns STATIC_FEW_SHOT_EXAMPLES
    mock_fsr_count.return_value = 0

    # causality_extractor mocks
    mock_ext_taxonomy.return_value = {"Regulation": ["etf_ruling"]}
    mock_ext_whitelist.return_value = _VALID_WHITELIST
    mock_llm_client = MagicMock()
    mock_llm_client.predict.return_value = _make_llm_mock(article)
    mock_ext_client.return_value = mock_llm_client

    # mechanism_validator mocks (embed_fn via graph.py's mlflow client)
    mock_val_whitelist.return_value = _VALID_WHITELIST
    mock_embed_client = MagicMock()
    mock_embed_client.predict.return_value = {"data": [{"embedding": [0.1] * 384}]}
    mock_val_client.return_value = mock_embed_client

    graph = build_mas_graph(config=MASConfig())
    state = {
        "headline": article["headline"],
        "body": article["body"],
        "published_at": article["published_at"],
    }
    result = graph.invoke(state)

    assert "validation_status" in result
    assert result["validation_status"] in (
        "accepted", "rejected_mechanism", "rejected_direction", "rejected_evidence"
    )
    assert "extraction_path" in result
    assert result["extraction_path"] in ("fast", "standard", "careful")


@patch("trading_databricks.agents.graph.mlflow.deployments.get_deploy_client")
@patch("trading_databricks.agents.graph.load_whitelist")
@patch("trading_databricks.agents.graph.get_spark_session")
@patch("trading_databricks.agents.nodes.causality_extractor.mlflow.deployments.get_deploy_client")
@patch("trading_databricks.agents.nodes.causality_extractor.get_spark_session")
@patch("trading_databricks.agents.nodes.causality_extractor.load_taxonomy")
@patch("trading_databricks.agents.nodes.causality_extractor.load_whitelist")
@patch("trading_databricks.agents.nodes.few_shot_retriever.get_accepted_chain_count")
@patch("trading_databricks.agents.nodes.few_shot_retriever.get_spark_session")
def test_end_to_end_10_articles(
    mock_fsr_spark,
    mock_fsr_count,
    mock_ext_whitelist,
    mock_ext_taxonomy,
    mock_ext_spark,
    mock_ext_client,
    mock_val_spark,
    mock_val_whitelist,
    mock_val_client,
):
    """Full graph: all 10 sample articles produce a valid MASState with validation_status."""
    from trading_databricks.agents.graph import build_mas_graph

    mock_fsr_count.return_value = 0
    mock_ext_taxonomy.return_value = {"Regulation": ["etf_ruling", "subcategory"]}
    mock_ext_whitelist.return_value = _VALID_WHITELIST
    mock_val_whitelist.return_value = _VALID_WHITELIST

    mock_embed_client = MagicMock()
    mock_embed_client.predict.return_value = {"data": [{"embedding": [0.1] * 384}]}
    mock_val_client.return_value = mock_embed_client

    graph = build_mas_graph(config=MASConfig())

    for article in SAMPLE_ARTICLES:
        mock_llm_client = MagicMock()
        mock_llm_client.predict.return_value = _make_llm_mock(article)
        mock_ext_client.return_value = mock_llm_client

        state = {
            "headline": article["headline"],
            "body": article["body"],
            "published_at": article["published_at"],
        }
        result = graph.invoke(state)

        assert "validation_status" in result, f"Missing validation_status for: {article['headline']}"
        assert result["validation_status"] in (
            "accepted", "rejected_mechanism", "rejected_direction", "rejected_evidence"
        ), f"Invalid validation_status '{result['validation_status']}' for: {article['headline']}"
        assert "extraction" in result, f"Missing extraction for: {article['headline']}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/agents/test_graph.py::test_end_to_end_fast_path \
              tests/agents/test_graph.py::test_end_to_end_10_articles -v
```

Expected: `FAILED` — graph not yet complete (or passes if previous tasks done — that's fine too, verify the test runs).

- [ ] **Step 3: Run full test_graph.py**

```bash
uv run pytest tests/agents/test_graph.py -v
```

Expected: All tests PASS (route_article × 4, validator_node × 2, compile × 1, e2e × 2 = 9 tests).

- [ ] **Step 4: Commit**

```bash
git add tests/agents/test_graph.py
git commit -m "TRD-32 test: add end-to-end integration tests for full MAS graph"
```

---

## Task 10: Final validation, coordination update, and PR

**Files:**
- Modify: `docs/coordination/TRD-6-gen-ai-agent-layer.md`

- [ ] **Step 1: Run full test suite (regression check)**

```bash
uv run pytest tests/agents/ -v
```

Expected: All tests PASS. No regressions from existing node tests.

- [ ] **Step 2: Run bundle validate**

```bash
databricks bundle validate
```

Expected: `Validation succeeded` (or similar). Fix any YAML/config issues before continuing.

- [ ] **Step 3: Update coordination doc — Section 5 to done**

In `docs/coordination/TRD-6-gen-ai-agent-layer.md`, update Section 5:

```markdown
## Section 5: Integration + Deploy + Eval — TRD-32, TRD-33, TRD-34 (Wave 4–6)
- **Status:** done (TRD-32 complete)
- **Branch:** feat/TRD-32-w4-wire-langgraph-pipeline (merged)
- **Modified files:**
  - `src/trading_databricks/agents/state.py` (MODIFIED — body, extraction, validation fields)
  - `src/trading_databricks/agents/graph.py` (NEW — build_mas_graph, route_article)
  - `src/trading_databricks/agents/tools.py` (MODIFIED — fetch_mechanism_whitelist)
  - `src/trading_databricks/agents/gold.py` (NEW — news_causal_features_transform)
  - `src/trading_databricks/agents/pipeline.py` (MODIFIED — @dp.materialized_view)
  - `tests/agents/test_graph.py` (NEW)
  - `tests/agents/test_tools.py` (NEW or MODIFIED)
  - `tests/agents/test_gold.py` (NEW)
- **Exports (public APIs):**
  - `agents/graph.py`: `build_mas_graph(*, config: MASConfig) -> CompiledStateGraph`
  - `agents/graph.py`: `route_article(state: MASState) -> str`
  - `agents/tools.py`: `fetch_mechanism_whitelist(*, config: MASConfig | None) -> str`
  - `agents/gold.py`: `news_causal_features_transform(spark, *, catalog, gatekeeper_config) -> DataFrame`
- **Needs from others:** All exports from Sections 1–4 ✅
- **Notes:** TRD-32 complete. TRD-33 (MLflow packaging + Model Serving) is next — needs
  `build_mas_graph()` to wrap in ResponsesAgent + log to MLflow.
```

- [ ] **Step 4: Final commit**

```bash
git add docs/coordination/TRD-6-gen-ai-agent-layer.md
git commit -m "TRD-32 docs: update coordination doc — Section 5 done"
```

- [ ] **Step 5: Push and open PR**

```bash
git push -u origin feat/TRD-32-w4-wire-langgraph-pipeline
```

Then open a PR via `gh pr create`:

```bash
gh pr create \
  --title "TRD-32: Wire complete LangGraph pipeline + UC tools + SDP integration" \
  --body "## Summary

- Wires all 5 MAS nodes into a compiled LangGraph StateGraph via closure-bound config
- Adds \`fetch_mechanism_whitelist()\` UC function tool to tools.py
- Implements \`news_causal_features_transform()\` in gold.py with gatekeeper UDF + ai_query
- Thin \`@dp.materialized_view\` wrapper in pipeline.py follows ticks/ SDP pattern
- Extends MASState with \`body\`, \`extraction\`, and validation fields
- 9 tests in test_graph.py (route_article, validator bridge, compile, e2e × 10 articles)

## Test plan
- [ ] \`uv run pytest tests/agents/test_graph.py -v\` — all 9 pass
- [ ] \`uv run pytest tests/agents/ -v\` — full suite, no regressions
- [ ] \`databricks bundle validate\` — pipeline config valid
- [ ] Manual smoke: \`build_mas_graph().invoke({...})\` returns state with \`validation_status\`

Closes TRD-32. Unblocks TRD-33.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| `build_mas_graph()` compiles | Task 5 |
| 5-node graph topology (spec §3.1) | Task 5 |
| Conditional edge from uncertainty_router (spec §2.2) | Task 5 |
| `fetch_market_context()` UC tool (spec §3.3) | Already in tools.py — no change needed |
| `fetch_mechanism_whitelist()` UC tool (spec §3.3) | Task 6 |
| `retrieve_few_shot_chains()` UC tool (spec §3.3) | Already in tools.py — no change needed |
| Gold MV `news_causal_features` with ai_query (spec §3.1) | Tasks 7–8 |
| Gatekeeper pandas UDF in gold MV (spec §5.4) | Task 7 |
| End-to-end: 10 sample articles (acceptance criteria) | Task 9 |
| `uv run pytest tests/agents/test_graph.py` passes (acceptance criteria) | Tasks 3–9 |
| `MASState` with `body` field (causality_extractor line 81) | Task 2 |
| `MASState` with `extraction` field | Task 2 |

All spec requirements covered. ✅
