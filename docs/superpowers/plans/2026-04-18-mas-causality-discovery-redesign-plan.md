# MAS Causality Discovery Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic `causality_extractor` node with a 5-node sub-pipeline (`evidence_extractor → rag_retriever → hypothesis_generator → judge_panel → reflector`) that decomposes extraction into observable steps and adds bounded reflection driven by an online LLM judge panel.

**Architecture:** Five new LangGraph nodes wired into `graph.py`, replacing the single `causality_extractor` between `uncertainty_router` and `mechanism_validator`. State carries new keys (`evidence`, `precedents`, `judge_results`, `judge_failures`, `reflection_count`, `extraction_status`). Reflector loop is bounded to `max_reflections=1`. Existing `mechanism_validator` is unchanged. Prompt YAMLs follow existing `prompts/` pattern; LLM calls reuse `mlflow.deployments.get_deploy_client`.

**Tech Stack:** LangGraph (StateGraph + Send API), MLflow Deployments client, Databricks Vector Search, Polars (for OHLC joins where needed), pytest with `unittest.mock.patch` for LLM/VS mocking.

**Spec:** `docs/superpowers/specs/2026-04-18-mas-causality-discovery-redesign.md`

---

## File Structure

**New files:**
- `src/trading_databricks/agents/nodes/evidence_extractor.py` — Node 1: structured evidence extraction (LLM)
- `src/trading_databricks/agents/nodes/rag_retriever.py` — Node 2: precedent news retrieval + OHLC join
- `src/trading_databricks/agents/nodes/hypothesis_generator.py` — Node 3: causal hypothesis generation (LLM, replaces old `causality_extractor` LLM call)
- `src/trading_databricks/agents/nodes/judge_panel.py` — Node 4: 3-judge parallel panel (LLM x 3)
- `src/trading_databricks/agents/nodes/reflector.py` — Node 5: pure routing logic
- `src/trading_databricks/agents/prompts/evidence_extractor.yaml` — Prompt for evidence extraction
- `src/trading_databricks/agents/prompts/hypothesis_generator.yaml` — Prompt for hypothesis generation (with reflection footer)
- `tests/agents/nodes/__init__.py` — Test package marker
- `tests/agents/nodes/test_evidence_extractor.py`
- `tests/agents/nodes/test_rag_retriever.py`
- `tests/agents/nodes/test_hypothesis_generator.py`
- `tests/agents/nodes/test_judge_panel.py`
- `tests/agents/nodes/test_reflector.py`
- `tests/agents/test_causality_subpipeline.py` — End-to-end sub-pipeline integration test

**Modified files:**
- `src/trading_databricks/agents/state.py` — Add 7 new state keys
- `src/trading_databricks/agents/config.py` — Add `ReflectionConfig`, `JudgePanelConfig`; nest in `MASConfig`
- `src/trading_databricks/common/uc_identifiers.py` — Add `news_firehose_idx` and `mechanism_candidates_review`
- `src/trading_databricks/agents/graph.py` — Rewire topology to use 5-node sub-pipeline
- `src/trading_databricks/agents/nodes/__init__.py` — Export new nodes
- `tests/agents/test_graph.py` — Update topology assertions
- `tests/agents/test_state.py` — Add coverage for new state keys

**Boundary principle:** Each node file owns one node function plus its private helpers. No node mutates upstream state keys — they only return dicts containing the keys they own (per LangGraph state-merge semantics). This keeps MLflow per-node spans replayable.

---

## Task 1: Extend MASState with new keys

**Files:**
- Modify: `src/trading_databricks/agents/state.py`
- Test: `tests/agents/test_state.py`

- [ ] **Step 1: Write failing test**

Append to `tests/agents/test_state.py`:

```python
def test_mas_state_accepts_new_subpipeline_keys():
    """Verify TypedDict accepts the 7 new keys added for causality discovery sub-pipeline."""
    from trading_databricks.agents.state import MASState

    state: MASState = {
        "evidence": [{"actor": "BlackRock", "action": "files"}],
        "evidence_status": "ok",
        "precedents": [{"headline": "x", "return_24h": 0.04}],
        "precedent_status": "ok",
        "extraction_status": "ok",
        "judge_results": {"direction": {"passed": True, "score": 0.9}},
        "judge_failures": None,
        "reflection_count": 0,
    }
    assert state["evidence_status"] == "ok"
    assert state["reflection_count"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_state.py::test_mas_state_accepts_new_subpipeline_keys -v`

Expected: This test passes trivially because `TypedDict(total=False)` accepts arbitrary keys at runtime. The "failing" check is on the type-level contract — verify by adding a `mypy` or `pyright` assertion equivalent. Since the project does not gate on type checks, treat this as a documentation/coverage test that simply asserts the keys can be set and read without error.

- [ ] **Step 3: Add the new state keys**

Replace the `MASState` class in `src/trading_databricks/agents/state.py`:

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
        body: Full article body text (always str, empty string if absent — never None)
        published_at: ISO timestamp string
        symbol: Asset symbol (e.g., "BTCUSDT", "ETHUSDT") — used by rag_retriever for OHLC join

    Context fields (set by context_retriever):
        market_vol_at_publish, market_trend_at_publish, dollar_bar_close

    Retrieval fields (set by few_shot_retriever):
        few_shot_examples, retrieval_story_ids

    Routing fields (set by uncertainty_router):
        extraction_path

    Sub-pipeline fields (added 2026-04-18):
        evidence_extractor → evidence, evidence_status
        rag_retriever      → precedents, precedent_status
        hypothesis_generator → extraction, extraction_status
        judge_panel        → judge_results, judge_failures
        reflector          → reflection_count, extraction_status (overwrites)

    Validation fields (set by mechanism_validator):
        validation_status, rejected_mechanism_raw, evidence_relevance_score
    """

    # --- input ---
    headline: str
    body: str                        # always str; use "" if absent, never None
    published_at: str
    symbol: str                      # e.g., "BTCUSDT" — used by rag_retriever

    # --- context (context_retriever) ---
    market_vol_at_publish: float | None
    market_trend_at_publish: str | None
    dollar_bar_close: float | None

    # --- retrieval (few_shot_retriever) ---
    few_shot_examples: list[dict]
    retrieval_story_ids: list[str]

    # --- routing (uncertainty_router) ---
    extraction_path: str

    # --- sub-pipeline: evidence_extractor ---
    evidence: list[dict]
    evidence_status: str             # "ok" | "empty" | "failed"

    # --- sub-pipeline: rag_retriever ---
    precedents: list[dict]
    precedent_status: str            # "ok" | "no_match" | "unavailable"

    # --- sub-pipeline: hypothesis_generator ---
    extraction: dict | None
    extraction_status: str           # "ok" | "ok_after_reflection" | "judge_failed_accepted" | "unjudged" | "parse_failed" | "failed"

    # --- sub-pipeline: judge_panel ---
    judge_results: dict | None
    judge_failures: dict | None      # populated when judges fail; fed back into hypothesis_generator on reflection

    # --- sub-pipeline: reflector ---
    reflection_count: int

    # --- validation (mechanism_validator) ---
    validation_status: str | None
    rejected_mechanism_raw: str | None
    evidence_relevance_score: float | None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agents/test_state.py -v`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/agents/state.py tests/agents/test_state.py
git commit -m "TRD feat: add sub-pipeline state keys to MASState"
```

---

## Task 2: Add UC identifiers for news firehose index and candidates review table

**Files:**
- Modify: `src/trading_databricks/common/uc_identifiers.py`
- Test: `tests/common/test_uc_identifiers.py` (create if missing)

- [ ] **Step 1: Inspect existing test file**

Run: `ls tests/common/`

If `test_uc_identifiers.py` does not exist, create it with the test below. Otherwise append.

- [ ] **Step 2: Write failing test**

Create or append to `tests/common/test_uc_identifiers.py`:

```python
"""Tests for UC identifier helpers."""
from trading_databricks.common.uc_identifiers import (
    NEWS_FIREHOSE_INDEX_TABLE,
    NEWS_FIREHOSE_INDEX_FQN,
    MECHANISM_CANDIDATES_REVIEW_TABLE,
    MECHANISM_CANDIDATES_REVIEW_FQN,
    EVAL_SCHEMA,
)


def test_news_firehose_index_identifiers():
    assert NEWS_FIREHOSE_INDEX_TABLE == "news_firehose_idx"
    assert NEWS_FIREHOSE_INDEX_FQN.endswith(".news_firehose_idx")
    assert "03_gold" in NEWS_FIREHOSE_INDEX_FQN


def test_mechanism_candidates_review_identifiers():
    assert MECHANISM_CANDIDATES_REVIEW_TABLE == "mechanism_candidates_review"
    assert MECHANISM_CANDIDATES_REVIEW_FQN.endswith(".mechanism_candidates_review")
    assert EVAL_SCHEMA == "04_eval"
    assert EVAL_SCHEMA in MECHANISM_CANDIDATES_REVIEW_FQN
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/common/test_uc_identifiers.py -v`

Expected: FAIL with `ImportError: cannot import name 'NEWS_FIREHOSE_INDEX_TABLE'`.

- [ ] **Step 4: Add the identifiers**

Append to `src/trading_databricks/common/uc_identifiers.py`:

```python
EVAL_SCHEMA = "04_eval"

NEWS_FIREHOSE_INDEX_TABLE = "news_firehose_idx"
MECHANISM_CANDIDATES_REVIEW_TABLE = "mechanism_candidates_review"

NEWS_FIREHOSE_INDEX_FQN = table_fqn(
    DEFAULT_CATALOG,
    GOLD_SCHEMA,
    NEWS_FIREHOSE_INDEX_TABLE,
)
MECHANISM_CANDIDATES_REVIEW_FQN = table_fqn(
    DEFAULT_CATALOG,
    EVAL_SCHEMA,
    MECHANISM_CANDIDATES_REVIEW_TABLE,
)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/common/test_uc_identifiers.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/trading_databricks/common/uc_identifiers.py tests/common/test_uc_identifiers.py
git commit -m "TRD feat: add news_firehose_idx and mechanism_candidates_review UC identifiers"
```

---

## Task 3: Add ReflectionConfig and JudgePanelConfig

**Files:**
- Modify: `src/trading_databricks/agents/config.py`
- Test: `tests/agents/test_config.py`

- [ ] **Step 1: Write failing test**

Append to `tests/agents/test_config.py`:

```python
def test_reflection_config_defaults():
    from trading_databricks.agents.config import ReflectionConfig

    cfg = ReflectionConfig()
    assert cfg.max_reflections == 1


def test_judge_panel_config_defaults():
    from trading_databricks.agents.config import JudgePanelConfig

    cfg = JudgePanelConfig()
    assert cfg.judge_endpoint == "databricks-meta-llama-3-1-8b-instruct"
    assert cfg.timeout_s == 10.0


def test_mas_config_includes_subpipeline_configs():
    from trading_databricks.agents.config import MASConfig

    cfg = MASConfig()
    assert cfg.reflection.max_reflections == 1
    assert cfg.judge_panel.judge_endpoint
    assert cfg.rag.firehose_index_name.endswith(".news_firehose_idx")
    assert cfg.rag.timeout_s == 5.0
    assert cfg.rag.num_precedents == 5
    assert cfg.rag.outcome_window_hours == 24
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_config.py -v`

Expected: FAIL — `ReflectionConfig` not defined.

- [ ] **Step 3: Add the configs**

Edit `src/trading_databricks/agents/config.py`. Add the imports and new dataclasses, and extend `MASConfig`:

```python
"""Frozen configuration dataclasses for all MAS components.

All values match the spec defaults exactly. Pass instances through
agent nodes rather than threading raw parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from trading_databricks.common.uc_identifiers import (
    EMBEDDING_ENDPOINT_NAME,
    GOLD_SCHEMA,
    DEFAULT_CATALOG,
    MAS_LLM_ENDPOINT_NAME,
    MAS_VS_ENDPOINT_NAME,
    MAS_VS_INDEX_NAME,
    NEWS_FIREHOSE_INDEX_FQN,
    OHLC_GOLD_TABLE,
)


@dataclass(frozen=True)
class GatekeeperConfig:
    model_uri: str = "models:/news-gatekeeper/Production"
    confidence_threshold: float = 0.5
    ner_only_mode: bool = True
    text_window: int = 500


@dataclass(frozen=True)
class UncertaintyRouterConfig:
    high_similarity_threshold: float = 0.85
    low_similarity_threshold: float = 0.50
    self_consistency_k: int = 3
    min_vs_results: int = 2
    min_index_size: int = 50


@dataclass(frozen=True)
class MechanismValidatorConfig:
    evidence_grounding_reject_threshold: float | None = None
    embedding_model: str = EMBEDDING_ENDPOINT_NAME


@dataclass(frozen=True)
class ReflectionConfig:
    """Bounded reflection loop config for the causality sub-pipeline.

    max_reflections=1 means at most one retry of hypothesis_generator after a
    judge_panel failure (worst case = 2 hypothesis_generator calls per article).
    """
    max_reflections: int = 1


@dataclass(frozen=True)
class JudgePanelConfig:
    """Online LLM judge panel config.

    Three judges (direction, mechanism, grounding) run in parallel via
    LangGraph Send API. Strict-AND aggregation: panel passes only if all three pass.
    Per-judge timeout prevents one slow judge from blocking the panel.
    """
    judge_endpoint: str = MAS_LLM_ENDPOINT_NAME
    timeout_s: float = 10.0
    grounding_threshold: float = 0.5  # grounding judge returns float; ≥ this → pass


@dataclass(frozen=True)
class RAGRetrieverConfig:
    """Deep RAG config: precedent news + market outcome join."""
    firehose_index_name: str = NEWS_FIREHOSE_INDEX_FQN
    timeout_s: float = 5.0
    num_precedents: int = 5
    outcome_window_hours: int = 24
    ohlc_table: str = OHLC_GOLD_TABLE


@dataclass(frozen=True)
class MASConfig:
    gatekeeper: GatekeeperConfig = field(default_factory=GatekeeperConfig)
    router: UncertaintyRouterConfig = field(default_factory=UncertaintyRouterConfig)
    validator: MechanismValidatorConfig = field(default_factory=MechanismValidatorConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    judge_panel: JudgePanelConfig = field(default_factory=JudgePanelConfig)
    rag: RAGRetrieverConfig = field(default_factory=RAGRetrieverConfig)
    catalog: str = DEFAULT_CATALOG
    schema: str = GOLD_SCHEMA
    vs_index_name: str = MAS_VS_INDEX_NAME
    vs_endpoint_name: str = MAS_VS_ENDPOINT_NAME
    llm_endpoint: str = MAS_LLM_ENDPOINT_NAME
    agent_model_version: str = "v1.0"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agents/test_config.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/agents/config.py tests/agents/test_config.py
git commit -m "TRD feat: add ReflectionConfig, JudgePanelConfig, RAGRetrieverConfig"
```

---

## Task 4: Implement evidence_extractor node

**Files:**
- Create: `src/trading_databricks/agents/prompts/evidence_extractor.yaml`
- Create: `src/trading_databricks/agents/nodes/evidence_extractor.py`
- Create: `tests/agents/nodes/__init__.py`
- Create: `tests/agents/nodes/test_evidence_extractor.py`

- [ ] **Step 1: Create the test package marker**

Create `tests/agents/nodes/__init__.py` (empty file):

```python
```

- [ ] **Step 2: Write failing tests**

Create `tests/agents/nodes/test_evidence_extractor.py`:

```python
"""Tests for evidence_extractor node."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trading_databricks.agents.config import MASConfig


@patch("trading_databricks.agents.nodes.evidence_extractor.mlflow.deployments.get_deploy_client")
def test_evidence_extractor_ok(mock_client_factory):
    from trading_databricks.agents.nodes.evidence_extractor import evidence_extractor

    mock_client = MagicMock()
    mock_client.predict.return_value = {
        "choices": [{"message": {"content": '[{"actor": "BlackRock", "action": "files", "object": "spot Ether ETF", "span": "BlackRock files for spot Ether ETF", "timestamp_hint": null}]'}}]
    }
    mock_client_factory.return_value = mock_client

    state = {"headline": "BlackRock files for spot Ether ETF", "body": "BlackRock files for spot Ether ETF with the SEC."}
    out = evidence_extractor(state, config=MASConfig())

    assert out["evidence_status"] == "ok"
    assert len(out["evidence"]) == 1
    assert out["evidence"][0]["actor"] == "BlackRock"


@patch("trading_databricks.agents.nodes.evidence_extractor.mlflow.deployments.get_deploy_client")
def test_evidence_extractor_empty_on_malformed_json(mock_client_factory):
    from trading_databricks.agents.nodes.evidence_extractor import evidence_extractor

    mock_client = MagicMock()
    mock_client.predict.return_value = {
        "choices": [{"message": {"content": "not json at all"}}]
    }
    mock_client_factory.return_value = mock_client

    out = evidence_extractor({"headline": "x", "body": "y"}, config=MASConfig())

    assert out["evidence_status"] == "empty"
    assert out["evidence"] == []


@patch("trading_databricks.agents.nodes.evidence_extractor.mlflow.deployments.get_deploy_client")
def test_evidence_extractor_failed_on_llm_exception(mock_client_factory):
    from trading_databricks.agents.nodes.evidence_extractor import evidence_extractor

    mock_client = MagicMock()
    mock_client.predict.side_effect = RuntimeError("LLM down")
    mock_client_factory.return_value = mock_client

    out = evidence_extractor({"headline": "x", "body": "y"}, config=MASConfig())

    assert out["evidence_status"] == "failed"
    assert out["evidence"] == []
    assert out["extraction_status"] == "failed"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/agents/nodes/test_evidence_extractor.py -v`

Expected: FAIL with `ImportError: cannot import name 'evidence_extractor'`.

- [ ] **Step 4: Create the prompt YAML**

Create `src/trading_databricks/agents/prompts/evidence_extractor.yaml`:

```yaml
name: evidence_extractor
description: Extracts structured evidence tuples from a news article for downstream causal reasoning.
variables:
  - headline
  - body
messages:
  - role: system
    content: |
      You extract structured evidence from financial news articles.

      Output a JSON array of evidence objects. Each object MUST have these fields:
      - actor: who performs the action (string, may be null if not present)
      - action: the verb describing what happened (string)
      - object: what the action targets (string, may be null if not present)
      - span: the verbatim text span from the article supporting this evidence (string)
      - timestamp_hint: any time reference inside the span (string or null)

      Only extract evidence that is verbatim grounded in the article. Do not paraphrase the span field.
      Return at most 5 evidence objects, prioritising the most causally relevant.
      Output JSON only — no prose, no markdown fences.

      HEADLINE: {headline}
      BODY: {body}
metadata:
  output_schema: list[evidence_object]
  used_by: [causality_subpipeline]
```

- [ ] **Step 5: Implement the node**

Create `src/trading_databricks/agents/nodes/evidence_extractor.py`:

```python
"""Evidence extractor node — first step of the causality sub-pipeline.

Extracts structured (actor, action, object, span, timestamp_hint) tuples from
the article using a single LLM call. Output flows into rag_retriever and
hypothesis_generator.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import mlflow.deployments

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.prompts import load_prompt, render_messages
from trading_databricks.agents.state import MASState

logger = logging.getLogger(__name__)


def _parse_evidence(text: str) -> list[dict] | None:
    """Parse a JSON array from LLM response, stripping markdown fences."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return [item for item in parsed if isinstance(item, dict)]


def _get_client():
    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE", os.environ.get("DATABRICKS_PROFILE", ""))
    target_uri = f"databricks://{profile}" if profile else "databricks"
    return mlflow.deployments.get_deploy_client(target_uri)


def evidence_extractor(state: MASState, config: MASConfig) -> dict[str, Any]:
    """Extract structured evidence tuples from the article.

    Returns a dict with keys:
        evidence: list[dict]   (empty list on failure or malformed JSON)
        evidence_status: str   "ok" | "empty" | "failed"
        extraction_status: str (only set to "failed" on LLM exception, propagates termination)
    """
    headline = state.get("headline", "")
    body = state.get("body", "")

    messages = load_prompt("evidence_extractor", version="production")
    rendered = render_messages(messages, {"headline": headline, "body": body})

    try:
        client = _get_client()
        response = client.predict(
            endpoint=config.llm_endpoint,
            inputs={"messages": rendered, "temperature": 0.0, "max_tokens": 800},
        )
    except Exception as exc:
        logger.warning("evidence_extractor LLM call failed: %s", exc)
        return {"evidence": [], "evidence_status": "failed", "extraction_status": "failed"}

    text = (response.get("choices") or [{}])[0].get("message", {}).get("content", "")
    parsed = _parse_evidence(text)
    if parsed is None or len(parsed) == 0:
        return {"evidence": [], "evidence_status": "empty"}
    return {"evidence": parsed, "evidence_status": "ok"}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/agents/nodes/test_evidence_extractor.py -v`

Expected: 3 PASS.

- [ ] **Step 7: Commit**

```bash
git add src/trading_databricks/agents/prompts/evidence_extractor.yaml src/trading_databricks/agents/nodes/evidence_extractor.py tests/agents/nodes/__init__.py tests/agents/nodes/test_evidence_extractor.py
git commit -m "TRD feat: evidence_extractor node + prompt"
```

---

## Task 5: Implement rag_retriever node

**Files:**
- Create: `src/trading_databricks/agents/nodes/rag_retriever.py`
- Create: `tests/agents/nodes/test_rag_retriever.py`

- [ ] **Step 1: Write failing tests**

Create `tests/agents/nodes/test_rag_retriever.py`:

```python
"""Tests for rag_retriever node."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trading_databricks.agents.config import MASConfig


def _state_with_evidence():
    return {
        "headline": "BlackRock files for spot Ether ETF",
        "body": "BlackRock files for spot Ether ETF.",
        "symbol": "ETHUSDT",
        "published_at": "2026-04-18T10:00:00Z",
        "evidence": [{"actor": "BlackRock", "action": "files", "span": "BlackRock files for spot Ether ETF"}],
        "evidence_status": "ok",
    }


@patch("trading_databricks.agents.nodes.rag_retriever._join_market_outcomes")
@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose")
def test_rag_retriever_ok(mock_query, mock_join):
    from trading_databricks.agents.nodes.rag_retriever import rag_retriever

    mock_query.return_value = [
        {"headline": "Fidelity files Bitcoin ETF", "body_excerpt": "...", "timestamp": "2024-06-15T12:00:00Z"},
    ]
    mock_join.return_value = [
        {"headline": "Fidelity files Bitcoin ETF", "body_excerpt": "...", "timestamp": "2024-06-15T12:00:00Z", "return_24h": 0.043, "direction": "up"},
    ]

    out = rag_retriever(_state_with_evidence(), config=MASConfig())
    assert out["precedent_status"] == "ok"
    assert len(out["precedents"]) == 1
    assert out["precedents"][0]["direction"] == "up"


@patch("trading_databricks.agents.nodes.rag_retriever._join_market_outcomes")
@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose")
def test_rag_retriever_no_match(mock_query, mock_join):
    from trading_databricks.agents.nodes.rag_retriever import rag_retriever

    mock_query.return_value = []
    out = rag_retriever(_state_with_evidence(), config=MASConfig())
    mock_join.assert_not_called()
    assert out["precedent_status"] == "no_match"
    assert out["precedents"] == []


@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose")
def test_rag_retriever_unavailable_on_exception(mock_query):
    from trading_databricks.agents.nodes.rag_retriever import rag_retriever

    mock_query.side_effect = RuntimeError("Vector Search down")
    out = rag_retriever(_state_with_evidence(), config=MASConfig())
    assert out["precedent_status"] == "unavailable"
    assert out["precedents"] == []


@patch("trading_databricks.agents.nodes.rag_retriever._join_market_outcomes")
@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose")
def test_rag_retriever_drops_precedents_with_failed_join(mock_query, mock_join):
    from trading_databricks.agents.nodes.rag_retriever import rag_retriever

    mock_query.return_value = [
        {"headline": "A", "timestamp": "t1"},
        {"headline": "B", "timestamp": "t2"},
    ]
    # Join only returns one — the other was dropped due to missing OHLC
    mock_join.return_value = [
        {"headline": "A", "timestamp": "t1", "return_24h": 0.01, "direction": "up"},
    ]

    out = rag_retriever(_state_with_evidence(), config=MASConfig())
    assert out["precedent_status"] == "ok"
    assert len(out["precedents"]) == 1


@patch("trading_databricks.agents.nodes.rag_retriever._join_market_outcomes")
@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose")
def test_rag_retriever_no_match_when_all_joins_fail(mock_query, mock_join):
    from trading_databricks.agents.nodes.rag_retriever import rag_retriever

    mock_query.return_value = [{"headline": "A", "timestamp": "t1"}]
    mock_join.return_value = []  # all dropped

    out = rag_retriever(_state_with_evidence(), config=MASConfig())
    assert out["precedent_status"] == "no_match"
    assert out["precedents"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/nodes/test_rag_retriever.py -v`

Expected: FAIL with import errors.

- [ ] **Step 3: Implement the node**

Create `src/trading_databricks/agents/nodes/rag_retriever.py`:

```python
"""RAG retriever node — second step of the causality sub-pipeline.

Queries the news firehose Vector Search index for precedent articles, then
joins each hit to OHLC outcomes (±N hours) so the downstream LLM sees what
actually moved the market after similar past events.

Externals are wrapped in module-level helpers (`_query_firehose`,
`_join_market_outcomes`) so tests can monkeypatch them without standing up
Vector Search or Spark.
"""
from __future__ import annotations

import logging
from typing import Any

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.state import MASState

logger = logging.getLogger(__name__)


def _build_query_text(headline: str, evidence: list[dict]) -> str:
    """Build the semantic query string from headline + top evidence span."""
    if not evidence:
        return headline
    span = (evidence[0].get("span") or "").strip()
    return f"{headline}\n{span}" if span else headline


def _query_firehose(query_text: str, *, index_name: str, num_results: int, timeout_s: float) -> list[dict]:
    """Query the news firehose VS index. Real implementation calls Databricks VS.

    For local testing, this function is monkeypatched. The real implementation
    is added once the firehose index is provisioned (separate ticket).

    Returns a list of dicts with at least: headline, body_excerpt, timestamp.
    """
    raise NotImplementedError(
        "Firehose VS query requires the news_firehose_idx to be provisioned. "
        "Tests must monkeypatch this function."
    )


def _join_market_outcomes(
    precedents: list[dict],
    *,
    symbol: str,
    ohlc_table: str,
    window_hours: int,
) -> list[dict]:
    """Join each precedent to OHLC return over ±window_hours.

    Returns the precedents list with `return_24h` and `direction` ("up" |
    "down" | "flat") fields added. Precedents whose OHLC join fails are
    dropped from the returned list.

    Real implementation runs a Spark SQL query; tests monkeypatch.
    """
    raise NotImplementedError(
        "OHLC join requires Spark and ohlc_gold table. Tests must monkeypatch."
    )


def rag_retriever(state: MASState, config: MASConfig) -> dict[str, Any]:
    """Retrieve precedent news + market outcomes.

    Returns a dict with keys:
        precedents: list[dict]
        precedent_status: str  "ok" | "no_match" | "unavailable"
    """
    query_text = _build_query_text(state.get("headline", ""), state.get("evidence", []))
    symbol = state.get("symbol", "")

    try:
        hits = _query_firehose(
            query_text,
            index_name=config.rag.firehose_index_name,
            num_results=config.rag.num_precedents,
            timeout_s=config.rag.timeout_s,
        )
    except Exception as exc:
        logger.warning("rag_retriever firehose query failed: %s", exc)
        return {"precedents": [], "precedent_status": "unavailable"}

    if not hits:
        return {"precedents": [], "precedent_status": "no_match"}

    try:
        joined = _join_market_outcomes(
            hits,
            symbol=symbol,
            ohlc_table=config.rag.ohlc_table,
            window_hours=config.rag.outcome_window_hours,
        )
    except Exception as exc:
        logger.warning("rag_retriever OHLC join failed: %s", exc)
        return {"precedents": [], "precedent_status": "unavailable"}

    if not joined:
        return {"precedents": [], "precedent_status": "no_match"}

    return {"precedents": joined, "precedent_status": "ok"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/nodes/test_rag_retriever.py -v`

Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/agents/nodes/rag_retriever.py tests/agents/nodes/test_rag_retriever.py
git commit -m "TRD feat: rag_retriever node with firehose VS + OHLC join hooks"
```

---

## Task 6: Implement hypothesis_generator node

**Files:**
- Create: `src/trading_databricks/agents/prompts/hypothesis_generator.yaml`
- Create: `src/trading_databricks/agents/nodes/hypothesis_generator.py`
- Create: `tests/agents/nodes/test_hypothesis_generator.py`

- [ ] **Step 1: Write failing tests**

Create `tests/agents/nodes/test_hypothesis_generator.py`:

```python
"""Tests for hypothesis_generator node."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from trading_databricks.agents.config import MASConfig

VALID_EXTRACTION = {
    "treatment_class": "Regulation.etf_ruling",
    "trigger_event": "BlackRock files for spot Ether ETF",
    "mechanism": "etf_inflow_demand",
    "expected_direction": "bullish",
    "evidence_spans": ["BlackRock files for spot Ether ETF"],
    "confidence": 0.78,
    "reasoning": "Filing signals institutional ETF demand.",
}


def _base_state():
    return {
        "headline": "BlackRock files for spot Ether ETF",
        "body": "BlackRock files for spot Ether ETF with the SEC.",
        "evidence": [{"actor": "BlackRock", "action": "files", "span": "BlackRock files for spot Ether ETF"}],
        "precedents": [{"headline": "Fidelity files Bitcoin ETF", "return_24h": 0.043, "direction": "up"}],
        "few_shot_examples": [],
        "extraction_path": "standard",
        "reflection_count": 0,
    }


@patch("trading_databricks.agents.nodes.hypothesis_generator.load_taxonomy")
@patch("trading_databricks.agents.nodes.hypothesis_generator.load_whitelist")
@patch("trading_databricks.agents.nodes.hypothesis_generator.get_spark_session")
@patch("trading_databricks.agents.nodes.hypothesis_generator.mlflow.deployments.get_deploy_client")
def test_hypothesis_generator_ok(mock_client_factory, mock_spark, mock_wl, mock_tax):
    from trading_databricks.agents.nodes.hypothesis_generator import hypothesis_generator

    mock_tax.return_value = {"Regulation": ["etf_ruling"]}
    mock_wl.return_value = {("Regulation", "etf_inflow_demand"): ["bullish"]}

    mock_client = MagicMock()
    mock_client.predict.return_value = {
        "choices": [{"message": {"content": json.dumps(VALID_EXTRACTION)}}]
    }
    mock_client_factory.return_value = mock_client

    out = hypothesis_generator(_base_state(), config=MASConfig())
    assert out["extraction_status"] == "ok"
    assert out["extraction"]["mechanism"] == "etf_inflow_demand"


@patch("trading_databricks.agents.nodes.hypothesis_generator.load_taxonomy")
@patch("trading_databricks.agents.nodes.hypothesis_generator.load_whitelist")
@patch("trading_databricks.agents.nodes.hypothesis_generator.get_spark_session")
@patch("trading_databricks.agents.nodes.hypothesis_generator.mlflow.deployments.get_deploy_client")
def test_hypothesis_generator_retries_on_parse_failure(mock_client_factory, mock_spark, mock_wl, mock_tax):
    from trading_databricks.agents.nodes.hypothesis_generator import hypothesis_generator

    mock_tax.return_value = {"Regulation": ["etf_ruling"]}
    mock_wl.return_value = {("Regulation", "etf_inflow_demand"): ["bullish"]}

    mock_client = MagicMock()
    mock_client.predict.side_effect = [
        {"choices": [{"message": {"content": "garbage not json"}}]},
        {"choices": [{"message": {"content": json.dumps(VALID_EXTRACTION)}}]},
    ]
    mock_client_factory.return_value = mock_client

    out = hypothesis_generator(_base_state(), config=MASConfig())
    assert mock_client.predict.call_count == 2
    assert out["extraction_status"] == "ok"


@patch("trading_databricks.agents.nodes.hypothesis_generator.load_taxonomy")
@patch("trading_databricks.agents.nodes.hypothesis_generator.load_whitelist")
@patch("trading_databricks.agents.nodes.hypothesis_generator.get_spark_session")
@patch("trading_databricks.agents.nodes.hypothesis_generator.mlflow.deployments.get_deploy_client")
def test_hypothesis_generator_parse_failed_after_two_attempts(mock_client_factory, mock_spark, mock_wl, mock_tax):
    from trading_databricks.agents.nodes.hypothesis_generator import hypothesis_generator

    mock_tax.return_value = {}
    mock_wl.return_value = {}

    mock_client = MagicMock()
    mock_client.predict.return_value = {"choices": [{"message": {"content": "still garbage"}}]}
    mock_client_factory.return_value = mock_client

    out = hypothesis_generator(_base_state(), config=MASConfig())
    assert mock_client.predict.call_count == 2
    assert out["extraction_status"] == "parse_failed"
    assert out["extraction"] is None


@patch("trading_databricks.agents.nodes.hypothesis_generator.load_taxonomy")
@patch("trading_databricks.agents.nodes.hypothesis_generator.load_whitelist")
@patch("trading_databricks.agents.nodes.hypothesis_generator.get_spark_session")
@patch("trading_databricks.agents.nodes.hypothesis_generator.mlflow.deployments.get_deploy_client")
def test_hypothesis_generator_includes_judge_failures_in_reflection(mock_client_factory, mock_spark, mock_wl, mock_tax):
    from trading_databricks.agents.nodes.hypothesis_generator import hypothesis_generator

    mock_tax.return_value = {"Regulation": ["etf_ruling"]}
    mock_wl.return_value = {("Regulation", "etf_inflow_demand"): ["bullish"]}

    mock_client = MagicMock()
    mock_client.predict.return_value = {"choices": [{"message": {"content": json.dumps(VALID_EXTRACTION)}}]}
    mock_client_factory.return_value = mock_client

    state = _base_state()
    state["judge_failures"] = {"grounding": "evidence not verbatim in body"}
    state["reflection_count"] = 1

    out = hypothesis_generator(state, config=MASConfig())
    # Find the user message and verify failure context was injected
    args, kwargs = mock_client.predict.call_args
    messages = kwargs["inputs"]["messages"]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "grounding" in user_content
    assert "evidence not verbatim" in user_content
    assert out["extraction_status"] == "ok_after_reflection"


@patch("trading_databricks.agents.nodes.hypothesis_generator.load_taxonomy")
@patch("trading_databricks.agents.nodes.hypothesis_generator.load_whitelist")
@patch("trading_databricks.agents.nodes.hypothesis_generator.get_spark_session")
@patch("trading_databricks.agents.nodes.hypothesis_generator.mlflow.deployments.get_deploy_client")
def test_hypothesis_generator_failed_on_llm_exception(mock_client_factory, mock_spark, mock_wl, mock_tax):
    from trading_databricks.agents.nodes.hypothesis_generator import hypothesis_generator

    mock_tax.return_value = {}
    mock_wl.return_value = {}

    mock_client = MagicMock()
    mock_client.predict.side_effect = RuntimeError("LLM down")
    mock_client_factory.return_value = mock_client

    out = hypothesis_generator(_base_state(), config=MASConfig())
    assert out["extraction_status"] == "failed"
    assert out["extraction"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/nodes/test_hypothesis_generator.py -v`

Expected: FAIL with import errors.

- [ ] **Step 3: Create the prompt YAML**

Create `src/trading_databricks/agents/prompts/hypothesis_generator.yaml`:

```yaml
name: hypothesis_generator
description: Generates a causal-chain hypothesis grounded in evidence and precedent market outcomes.
variables:
  - taxonomy_injection
  - whitelist_injection
  - few_shot_injection
  - reflection_footer
messages:
  - role: system
    content: |
      You are a causal reasoning agent for crypto news.

      Given a news article (headline + body), structured evidence tuples, and
      a table of precedent articles with their actual market outcomes, produce
      a causal-chain extraction.

      VOCABULARY (soft-closed with escape hatch):
      - treatment_class MUST be of the form "L1.l2_subcategory" matching the taxonomy below.
      - mechanism SHOULD be a snake_case identifier from the whitelist.
      - If the article truly does not match any whitelist mechanism, set mechanism="other"
        and add a "mechanism_candidate" field describing the novel mechanism in <= 30 words.

      TAXONOMY:
      {taxonomy_injection}

      WHITELIST (mechanism -> allowed directions):
      {whitelist_injection}

      FEW-SHOT EXEMPLARS:
      {few_shot_injection}

      OUTPUT — single JSON object with these fields:
        treatment_class: str   ("L1.l2_subcategory")
        trigger_event:   str   (<=200 chars, the causal trigger phrased as a sentence)
        mechanism:       str   (whitelist token, or "other" if escape-hatch)
        mechanism_candidate: str | null  (only when mechanism == "other")
        expected_direction: str   ("bullish" | "bearish" | "volatile")
        evidence_spans:  list[str]   (1-3 verbatim quotes from the body)
        confidence:      float in [0, 1]
        reasoning:       str   (one paragraph; reference precedents if relevant)

      Output JSON only — no prose, no markdown fences.
      {reflection_footer}
  - role: user
    content: |
      HEADLINE: {headline}
      BODY: {body}

      EVIDENCE TUPLES:
      {evidence_block}

      PRECEDENT NEWS + MARKET OUTCOMES:
      {precedent_block}
      {failure_context}
metadata:
  reflection_footer_template: |
    A previous attempt failed judge validation. Address each failure listed in the user message.
  parse_retry_suffix: |
    REMINDER: Output JSON only. No prose, no markdown fences. Begin with `{` and end with `}`.
  used_by: [causality_subpipeline]
```

- [ ] **Step 4: Implement the node**

Create `src/trading_databricks/agents/nodes/hypothesis_generator.py`:

```python
"""Hypothesis generator node — replaces the monolithic causality_extractor LLM call.

Consumes evidence (from evidence_extractor) and precedents (from rag_retriever),
produces a single causal-chain extraction. On reflection retries, injects the
previous judge_failures into the user message.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import mlflow.deployments
import yaml

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.nodes.few_shot_retriever import get_spark_session
from trading_databricks.agents.prompts import _PACKAGE_DIR, load_prompt, render_messages
from trading_databricks.agents.prompts._formatters import (
    format_few_shot,
    format_full_taxonomy,
    format_whitelist,
)
from trading_databricks.agents.state import MASState
from trading_databricks.agents.taxonomy import load_taxonomy, load_whitelist

logger = logging.getLogger(__name__)


def _parse_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _format_evidence_block(evidence: list[dict]) -> str:
    if not evidence:
        return "(no structured evidence extracted)"
    lines = []
    for i, ev in enumerate(evidence, 1):
        actor = ev.get("actor") or "?"
        action = ev.get("action") or "?"
        obj = ev.get("object") or ""
        span = ev.get("span") or ""
        lines.append(f"{i}. actor={actor} action={action} object={obj} span=\"{span}\"")
    return "\n".join(lines)


def _format_precedent_block(precedents: list[dict]) -> str:
    if not precedents:
        return "(no precedents found)"
    lines = ["headline | timestamp | return_24h | direction"]
    for p in precedents:
        lines.append(
            f"{p.get('headline', '?')} | {p.get('timestamp', '?')} | "
            f"{p.get('return_24h', '?')} | {p.get('direction', '?')}"
        )
    return "\n".join(lines)


def _format_failure_context(judge_failures: dict | None) -> str:
    if not judge_failures:
        return ""
    lines = ["", "PRIOR JUDGE FAILURES (address each in your new attempt):"]
    for judge_name, reason in judge_failures.items():
        lines.append(f"- {judge_name}: {reason}")
    return "\n".join(lines)


def _get_client():
    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE", os.environ.get("DATABRICKS_PROFILE", ""))
    target_uri = f"databricks://{profile}" if profile else "databricks"
    return mlflow.deployments.get_deploy_client(target_uri)


def _load_metadata() -> dict:
    """Load YAML metadata (reflection footer + parse retry suffix templates)."""
    doc = yaml.safe_load((_PACKAGE_DIR / "hypothesis_generator.yaml").read_text())
    return doc.get("metadata", {})


def hypothesis_generator(state: MASState, config: MASConfig) -> dict[str, Any]:
    """Generate a causal-chain extraction grounded in evidence + precedents.

    Returns a dict with keys:
        extraction: dict | None
        extraction_status: str  "ok" | "ok_after_reflection" | "parse_failed" | "failed"
    """
    spark = get_spark_session()
    taxonomy = load_taxonomy(spark, catalog=config.catalog, schema=config.schema)
    whitelist = load_whitelist(spark, catalog=config.catalog, schema=config.schema)

    metadata = _load_metadata()
    judge_failures = state.get("judge_failures")
    reflection_count = state.get("reflection_count", 0)
    is_reflection = reflection_count > 0 and judge_failures

    messages = load_prompt("hypothesis_generator", version="production")
    rendered = render_messages(
        messages,
        {
            "taxonomy_injection": format_full_taxonomy(taxonomy),
            "whitelist_injection": format_whitelist(whitelist),
            "few_shot_injection": format_few_shot(state.get("few_shot_examples", []), 5),
            "reflection_footer": metadata.get("reflection_footer_template", "") if is_reflection else "",
            "headline": state.get("headline", ""),
            "body": state.get("body", ""),
            "evidence_block": _format_evidence_block(state.get("evidence", [])),
            "precedent_block": _format_precedent_block(state.get("precedents", [])),
            "failure_context": _format_failure_context(judge_failures),
        },
    )

    try:
        client = _get_client()
        response = client.predict(
            endpoint=config.llm_endpoint,
            inputs={"messages": rendered, "temperature": 0.0, "max_tokens": 1000},
        )
    except Exception as exc:
        logger.warning("hypothesis_generator LLM call failed: %s", exc)
        return {"extraction": None, "extraction_status": "failed"}

    text = (response.get("choices") or [{}])[0].get("message", {}).get("content", "")
    parsed = _parse_json(text)

    if parsed is None:
        # Retry once with stricter suffix
        retry_suffix = metadata.get("parse_retry_suffix", "")
        retry_messages = list(rendered)
        retry_messages.append({"role": "user", "content": retry_suffix})
        try:
            response = client.predict(
                endpoint=config.llm_endpoint,
                inputs={"messages": retry_messages, "temperature": 0.0, "max_tokens": 1000},
            )
        except Exception as exc:
            logger.warning("hypothesis_generator retry LLM call failed: %s", exc)
            return {"extraction": None, "extraction_status": "failed"}
        text = (response.get("choices") or [{}])[0].get("message", {}).get("content", "")
        parsed = _parse_json(text)

    if parsed is None:
        return {"extraction": None, "extraction_status": "parse_failed"}

    status = "ok_after_reflection" if is_reflection else "ok"
    return {"extraction": parsed, "extraction_status": status}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/agents/nodes/test_hypothesis_generator.py -v`

Expected: 5 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/trading_databricks/agents/prompts/hypothesis_generator.yaml src/trading_databricks/agents/nodes/hypothesis_generator.py tests/agents/nodes/test_hypothesis_generator.py
git commit -m "TRD feat: hypothesis_generator node with reflection-aware prompt"
```

---

## Task 7: Implement judge_panel node

**Files:**
- Create: `src/trading_databricks/agents/nodes/judge_panel.py`
- Create: `tests/agents/nodes/test_judge_panel.py`

- [ ] **Step 1: Write failing tests**

Create `tests/agents/nodes/test_judge_panel.py`:

```python
"""Tests for judge_panel node."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from trading_databricks.agents.config import MASConfig


def _state():
    return {
        "headline": "BlackRock files for spot Ether ETF",
        "body": "BlackRock files for spot Ether ETF.",
        "extraction": {
            "treatment_class": "Regulation.etf_ruling",
            "mechanism": "etf_inflow_demand",
            "expected_direction": "bullish",
            "evidence_spans": ["BlackRock files for spot Ether ETF"],
            "confidence": 0.78,
        },
    }


@patch("trading_databricks.agents.nodes.judge_panel._invoke_judge")
def test_judge_panel_all_pass(mock_invoke):
    from trading_databricks.agents.nodes.judge_panel import judge_panel

    # NOTE: direction judge returns "up"/"down"/"neutral"; judge_panel maps
    # extraction.expected_direction ("bullish") → "up" before comparing.
    mock_invoke.side_effect = [
        {"passed": True, "score": "up", "reason": ""},
        {"passed": True, "score": "match", "reason": ""},
        {"passed": True, "score": 0.9, "reason": ""},
    ]

    out = judge_panel(_state(), config=MASConfig())
    assert out["judge_failures"] is None
    assert out["judge_results"]["direction"]["passed"] is True
    assert out["judge_results"]["mechanism"]["passed"] is True
    assert out["judge_results"]["grounding"]["passed"] is True


@patch("trading_databricks.agents.nodes.judge_panel._invoke_judge")
def test_judge_panel_strict_and_one_fails(mock_invoke):
    from trading_databricks.agents.nodes.judge_panel import judge_panel

    # Grounding score 0.4 < threshold 0.5 → judge_panel rewrites reason to
    # "score 0.40 < threshold 0.5". The mock's "reason" is overwritten.
    mock_invoke.side_effect = [
        {"passed": True, "score": "up", "reason": ""},
        {"passed": True, "score": "match", "reason": ""},
        {"passed": False, "score": 0.4, "reason": "raw reason discarded"},
    ]

    out = judge_panel(_state(), config=MASConfig())
    assert "grounding" in out["judge_failures"]
    assert "threshold" in out["judge_failures"]["grounding"]


@patch("trading_databricks.agents.nodes.judge_panel._invoke_judge")
def test_judge_panel_individual_judge_unavailable(mock_invoke):
    from trading_databricks.agents.nodes.judge_panel import judge_panel

    mock_invoke.side_effect = [
        {"passed": True, "score": "up", "reason": ""},
        RuntimeError("judge LLM down"),
        {"passed": True, "score": 0.9, "reason": ""},
    ]

    out = judge_panel(_state(), config=MASConfig())
    assert out["judge_results"]["mechanism"]["passed"] is False
    assert out["judge_results"]["mechanism"]["score"] is None
    assert out["judge_results"]["mechanism"]["reason"] == "judge_unavailable"
    assert "mechanism" in out["judge_failures"]


@patch("trading_databricks.agents.nodes.judge_panel._invoke_judge")
def test_judge_panel_all_unavailable_marks_unjudged(mock_invoke):
    from trading_databricks.agents.nodes.judge_panel import judge_panel

    mock_invoke.side_effect = [RuntimeError("x"), RuntimeError("y"), RuntimeError("z")]

    out = judge_panel(_state(), config=MASConfig())
    assert out["judge_results"] == {}
    assert out["extraction_status"] == "unjudged"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/nodes/test_judge_panel.py -v`

Expected: FAIL with import errors.

- [ ] **Step 3: Implement the node**

Create `src/trading_databricks/agents/nodes/judge_panel.py`:

```python
"""Judge panel node — fan-out to 3 LLM judges, strict-AND aggregation.

The same judge prompts (judge_direction.yaml, judge_mechanism.yaml,
judge_grounding.yaml) used in the offline eval scorers are reused here for the
online reflection signal. Direction and mechanism judges return categorical
verdicts (parsed via simple equality / "match" check). Grounding judge returns
a float in [0, 1] thresholded at config.judge_panel.grounding_threshold.

The three judges are invoked sequentially in the implementation below for
simplicity; LangGraph's Send API would parallelise them across separate node
instances. Sequential is acceptable for v1 because each call has its own
timeout and the panel's wall-clock cost is dominated by the slowest judge
either way (the LLM endpoint multiplexes requests). Parallelisation can be
added in a follow-up if profiling shows it matters.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import mlflow.deployments

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.prompts import load_prompt, render_messages
from trading_databricks.agents.state import MASState

logger = logging.getLogger(__name__)


JUDGE_NAMES = ("judge_direction", "judge_mechanism", "judge_grounding")
JUDGE_KEYS = ("direction", "mechanism", "grounding")


def _get_client():
    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE", os.environ.get("DATABRICKS_PROFILE", ""))
    target_uri = f"databricks://{profile}" if profile else "databricks"
    return mlflow.deployments.get_deploy_client(target_uri)


def _invoke_judge(
    judge_name: str,
    *,
    headline: str,
    body: str,
    extraction: dict,
    endpoint: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Invoke a single judge LLM. Returns {passed, score, reason}.

    Real implementation calls the LLM endpoint with the rendered judge prompt.
    Tests monkeypatch this function.
    """
    messages = load_prompt(judge_name, version="production")
    rendered = render_messages(
        messages,
        {
            "headline": headline,
            "body": body,
            "extraction": json.dumps(extraction),
        },
    )
    client = _get_client()
    response = client.predict(
        endpoint=endpoint,
        inputs={"messages": rendered, "temperature": 0.0, "max_tokens": 100},
    )
    text = (response.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    return _interpret_judge_response(judge_name, text)


def _interpret_judge_response(judge_name: str, text: str) -> dict[str, Any]:
    """Convert raw judge output to {passed, score, reason}.

    direction → expects "up" | "down" | "neutral"; passes if matches extraction direction.
    mechanism → expects "match" | "mismatch"; passes on "match".
    grounding → expects float in [0, 1]; passes if >= threshold (caller applies threshold).
    """
    text = text.strip().lower()
    if judge_name == "judge_direction":
        # Caller compares against extraction.expected_direction; we pass through the verdict.
        return {"passed": text in ("up", "down", "neutral"), "score": text, "reason": "" if text in ("up", "down", "neutral") else "unparseable verdict"}
    if judge_name == "judge_mechanism":
        passed = text.startswith("match")
        return {"passed": passed, "score": text, "reason": "" if passed else text}
    if judge_name == "judge_grounding":
        try:
            score = float(text.split()[0])
        except (ValueError, IndexError):
            return {"passed": False, "score": None, "reason": f"unparseable score: {text[:50]}"}
        return {"passed": True, "score": score, "reason": ""}
    return {"passed": False, "score": None, "reason": f"unknown judge: {judge_name}"}


def _direction_pass(extraction: dict, verdict: dict) -> tuple[bool, str]:
    """Direction judge passes only if its verdict matches the extraction's expected_direction.

    extraction.expected_direction in {"bullish", "bearish", "volatile"}
    verdict.score in {"up", "down", "neutral"}
    Mapping: bullish→up, bearish→down, volatile→neutral.
    """
    mapping = {"bullish": "up", "bearish": "down", "volatile": "neutral"}
    expected = mapping.get(extraction.get("expected_direction", ""), "")
    actual = verdict.get("score", "")
    if expected and actual == expected:
        return True, ""
    return False, f"direction mismatch: extraction={extraction.get('expected_direction')} judge={actual}"


def judge_panel(state: MASState, config: MASConfig) -> dict[str, Any]:
    """Run all three judges and aggregate via strict-AND.

    Returns:
        judge_results: dict[str, dict] keyed by ("direction", "mechanism", "grounding")
        judge_failures: dict[str, str] | None — failure reasons keyed by judge name
        extraction_status (only on all-failed): "unjudged"
    """
    extraction = state.get("extraction") or {}
    headline = state.get("headline", "")
    body = state.get("body", "")

    results: dict[str, dict] = {}
    failures: dict[str, str] = {}
    unavailable_count = 0

    for judge_name, key in zip(JUDGE_NAMES, JUDGE_KEYS):
        try:
            verdict = _invoke_judge(
                judge_name,
                headline=headline,
                body=body,
                extraction=extraction,
                endpoint=config.judge_panel.judge_endpoint,
                timeout_s=config.judge_panel.timeout_s,
            )
        except Exception as exc:
            logger.warning("%s unavailable: %s", judge_name, exc)
            verdict = {"passed": False, "score": None, "reason": "judge_unavailable"}
            unavailable_count += 1

        # Apply per-judge final pass logic
        if key == "direction" and verdict.get("score") is not None and verdict.get("reason") != "judge_unavailable":
            passed, reason = _direction_pass(extraction, verdict)
            verdict = {"passed": passed, "score": verdict["score"], "reason": reason}
        elif key == "grounding" and isinstance(verdict.get("score"), (int, float)):
            passed = verdict["score"] >= config.judge_panel.grounding_threshold
            verdict = {"passed": passed, "score": verdict["score"], "reason": "" if passed else f"score {verdict['score']:.2f} < threshold {config.judge_panel.grounding_threshold}"}

        results[key] = verdict
        if not verdict["passed"]:
            failures[key] = verdict.get("reason") or "failed"

    if unavailable_count == len(JUDGE_NAMES):
        return {"judge_results": {}, "judge_failures": None, "extraction_status": "unjudged"}

    return {
        "judge_results": results,
        "judge_failures": failures or None,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/nodes/test_judge_panel.py -v`

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/agents/nodes/judge_panel.py tests/agents/nodes/test_judge_panel.py
git commit -m "TRD feat: judge_panel node with strict-AND aggregation"
```

---

## Task 8: Implement reflector node (pure routing logic)

**Files:**
- Create: `src/trading_databricks/agents/nodes/reflector.py`
- Create: `tests/agents/nodes/test_reflector.py`

- [ ] **Step 1: Write failing tests**

Create `tests/agents/nodes/test_reflector.py`:

```python
"""Tests for reflector node — pure routing logic."""
from __future__ import annotations

import pytest

from trading_databricks.agents.config import MASConfig


def test_reflector_passed_routes_to_validator():
    from trading_databricks.agents.nodes.reflector import reflector, REFLECTOR_ROUTE_VALIDATOR

    state = {
        "judge_results": {"direction": {"passed": True}, "mechanism": {"passed": True}, "grounding": {"passed": True}},
        "judge_failures": None,
        "reflection_count": 0,
        "extraction_status": "ok",
    }
    out = reflector(state, config=MASConfig())
    assert out["_route"] == REFLECTOR_ROUTE_VALIDATOR
    assert out["reflection_count"] == 0
    assert "extraction_status" not in out  # don't overwrite ok


def test_reflector_failed_within_budget_routes_back():
    from trading_databricks.agents.nodes.reflector import reflector, REFLECTOR_ROUTE_RETRY

    state = {
        "judge_results": {"direction": {"passed": False}},
        "judge_failures": {"direction": "mismatch"},
        "reflection_count": 0,
        "extraction_status": "ok",
    }
    out = reflector(state, config=MASConfig())
    assert out["_route"] == REFLECTOR_ROUTE_RETRY
    assert out["reflection_count"] == 1


def test_reflector_failed_budget_exhausted_routes_to_validator():
    from trading_databricks.agents.nodes.reflector import reflector, REFLECTOR_ROUTE_VALIDATOR

    state = {
        "judge_results": {"direction": {"passed": False}},
        "judge_failures": {"direction": "mismatch"},
        "reflection_count": 1,  # already at max_reflections=1
        "extraction_status": "ok",
    }
    out = reflector(state, config=MASConfig())
    assert out["_route"] == REFLECTOR_ROUTE_VALIDATOR
    assert out["extraction_status"] == "judge_failed_accepted"


def test_reflector_unjudged_routes_to_validator():
    from trading_databricks.agents.nodes.reflector import reflector, REFLECTOR_ROUTE_VALIDATOR

    state = {
        "judge_results": {},
        "judge_failures": None,
        "reflection_count": 0,
        "extraction_status": "unjudged",
    }
    out = reflector(state, config=MASConfig())
    assert out["_route"] == REFLECTOR_ROUTE_VALIDATOR
    # Status preserved
    assert "extraction_status" not in out or out["extraction_status"] == "unjudged"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/nodes/test_reflector.py -v`

Expected: FAIL with import errors.

- [ ] **Step 3: Implement the node**

Create `src/trading_databricks/agents/nodes/reflector.py`:

```python
"""Reflector node — pure routing logic for the bounded reflection loop.

Reads judge_results + reflection_count, decides whether to retry
hypothesis_generator or proceed to mechanism_validator. Writes a
`_route` key in the returned dict that the graph's conditional edge
function reads.
"""
from __future__ import annotations

import logging
from typing import Any

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.state import MASState

logger = logging.getLogger(__name__)

REFLECTOR_ROUTE_VALIDATOR = "mechanism_validator"
REFLECTOR_ROUTE_RETRY = "hypothesis_generator"


def _judges_passed(judge_results: dict | None) -> bool:
    """All judges passed iff the dict is non-empty and every entry has passed=True."""
    if not judge_results:
        return False
    return all(r.get("passed") for r in judge_results.values())


def reflector(state: MASState, config: MASConfig) -> dict[str, Any]:
    """Decide whether to reflect (retry) or move on.

    Returns a dict with:
        _route: str — REFLECTOR_ROUTE_VALIDATOR or REFLECTOR_ROUTE_RETRY
        reflection_count: int (incremented when retrying)
        extraction_status: str (set to "judge_failed_accepted" when budget exhausted with failures)

    Note: graph.py wires `_route` through `add_conditional_edges`.
    """
    judge_results = state.get("judge_results")
    judge_failures = state.get("judge_failures")
    reflection_count = state.get("reflection_count", 0)
    current_status = state.get("extraction_status", "ok")

    # Unjudged path: judges all unavailable, can't reflect — proceed to validator
    if current_status == "unjudged":
        return {"_route": REFLECTOR_ROUTE_VALIDATOR}

    # Happy path: all judges passed
    if _judges_passed(judge_results):
        return {"_route": REFLECTOR_ROUTE_VALIDATOR, "reflection_count": reflection_count}

    # Failed but within budget: retry
    if reflection_count < config.reflection.max_reflections:
        return {
            "_route": REFLECTOR_ROUTE_RETRY,
            "reflection_count": reflection_count + 1,
        }

    # Failed and budget exhausted: accept anyway, flag the status
    logger.info(
        "reflection budget exhausted; accepting flagged extraction. failures=%s",
        judge_failures,
    )
    return {
        "_route": REFLECTOR_ROUTE_VALIDATOR,
        "extraction_status": "judge_failed_accepted",
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/nodes/test_reflector.py -v`

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/agents/nodes/reflector.py tests/agents/nodes/test_reflector.py
git commit -m "TRD feat: reflector node with bounded retry routing"
```

---

## Task 9: Wire the sub-pipeline into graph.py

**Files:**
- Modify: `src/trading_databricks/agents/graph.py`
- Modify: `src/trading_databricks/agents/nodes/__init__.py`
- Modify: `tests/agents/test_graph.py`

- [ ] **Step 1: Update node exports**

Edit `src/trading_databricks/agents/nodes/__init__.py`:

```python
"""LangGraph node implementations for the MAS agent pipeline."""
from __future__ import annotations

from trading_databricks.agents.nodes.causality_extractor import causality_extractor
from trading_databricks.agents.nodes.context_retriever import context_retriever
from trading_databricks.agents.nodes.evidence_extractor import evidence_extractor
from trading_databricks.agents.nodes.few_shot_retriever import (
    STATIC_FEW_SHOT_EXAMPLES,
    few_shot_retriever,
)
from trading_databricks.agents.nodes.hypothesis_generator import hypothesis_generator
from trading_databricks.agents.nodes.judge_panel import judge_panel
from trading_databricks.agents.nodes.mechanism_validator import validate_extraction
from trading_databricks.agents.nodes.rag_retriever import rag_retriever
from trading_databricks.agents.nodes.reflector import (
    REFLECTOR_ROUTE_RETRY,
    REFLECTOR_ROUTE_VALIDATOR,
    reflector,
)
from trading_databricks.agents.nodes.uncertainty_router import uncertainty_router

__all__ = [
    "REFLECTOR_ROUTE_RETRY",
    "REFLECTOR_ROUTE_VALIDATOR",
    "STATIC_FEW_SHOT_EXAMPLES",
    "causality_extractor",
    "context_retriever",
    "evidence_extractor",
    "few_shot_retriever",
    "hypothesis_generator",
    "judge_panel",
    "rag_retriever",
    "reflector",
    "uncertainty_router",
    "validate_extraction",
]
```

- [ ] **Step 2: Write failing test for new graph topology**

Edit `tests/agents/test_graph.py`. Find the existing topology test (look for `build_mas_graph`) and add a new test asserting the sub-pipeline nodes are present:

```python
def test_build_mas_graph_includes_subpipeline_nodes():
    from trading_databricks.agents.config import MASConfig
    from trading_databricks.agents.graph import build_mas_graph

    graph = build_mas_graph(config=MASConfig())
    node_names = set(graph.get_graph().nodes.keys())

    expected = {
        "context_retriever",
        "few_shot_retriever",
        "uncertainty_router",
        "evidence_extractor",
        "rag_retriever",
        "hypothesis_generator",
        "judge_panel",
        "reflector",
        "mechanism_validator",
    }
    assert expected.issubset(node_names), f"missing nodes: {expected - node_names}"
    # The monolithic causality_extractor should no longer be a graph node
    assert "causality_extractor" not in node_names
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_graph.py::test_build_mas_graph_includes_subpipeline_nodes -v`

Expected: FAIL — `causality_extractor` is still in the graph.

- [ ] **Step 4: Rewire `build_mas_graph`**

Replace the body of `build_mas_graph` and update `route_article` in `src/trading_databricks/agents/graph.py`:

```python
"""LangGraph StateGraph assembly for the MAS pipeline.

Outer topology:
    context_retriever → few_shot_retriever → uncertainty_router
    → [conditional: route_article] → evidence_extractor → ...

Inner causality sub-pipeline (replaces the old monolithic causality_extractor):
    evidence_extractor
    → rag_retriever
    → hypothesis_generator
    → judge_panel
    → reflector
        → (retry) hypothesis_generator
        → (proceed) mechanism_validator → END

All transform logic lives in nodes/. This file owns topology only.
"""
from __future__ import annotations

import os
from typing import Literal

import mlflow.deployments

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.nodes import (
    REFLECTOR_ROUTE_RETRY,
    REFLECTOR_ROUTE_VALIDATOR,
)
from trading_databricks.agents.nodes.few_shot_retriever import get_spark_session
from trading_databricks.agents.nodes.mechanism_validator import validate_extraction
from trading_databricks.agents.state import MASState
from trading_databricks.agents.taxonomy import load_whitelist


def route_article(state: MASState) -> Literal["evidence_extractor"]:
    """Route from uncertainty_router into the causality sub-pipeline.

    All extraction paths (fast/standard/careful) enter at evidence_extractor.
    Per-path behavior (e.g., temperature, prompt depth) is now owned by
    individual sub-pipeline nodes via state.extraction_path.
    """
    return "evidence_extractor"


def route_after_reflector(state: MASState) -> str:
    """Read the `_route` key set by the reflector node and dispatch."""
    return state.get("_route", REFLECTOR_ROUTE_VALIDATOR)


def mechanism_validator_node(state: MASState, config: MASConfig) -> dict:
    """Bridge validate_extraction() to the LangGraph (state) -> dict interface."""
    extraction = state.get("extraction")
    if not extraction:
        return {
            "validation_status": "rejected_evidence",
            "rejected_mechanism_raw": None,
            "evidence_relevance_score": None,
        }

    spark = get_spark_session()
    whitelist = load_whitelist(spark, catalog=config.catalog, schema=config.schema)

    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE", os.environ.get("DATABRICKS_PROFILE", ""))
    target_uri = f"databricks://{profile}" if profile else "databricks"
    client = mlflow.deployments.get_deploy_client(target_uri)

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


def build_mas_graph(*, config: MASConfig):
    """Compile the MAS StateGraph with the 5-node causality sub-pipeline."""
    from langgraph.graph import StateGraph

    from trading_databricks.agents.nodes import (
        context_retriever,
        evidence_extractor,
        few_shot_retriever,
        hypothesis_generator,
        judge_panel,
        rag_retriever,
        reflector,
        uncertainty_router,
    )

    graph = StateGraph(MASState)

    # Outer nodes (unchanged)
    graph.add_node("context_retriever",   lambda s: context_retriever(s, config=config))
    graph.add_node("few_shot_retriever",  lambda s: few_shot_retriever(s, config=config))
    graph.add_node("uncertainty_router",  lambda s: uncertainty_router(s, config=config))

    # Sub-pipeline nodes (new)
    graph.add_node("evidence_extractor",   lambda s: evidence_extractor(s, config=config))
    graph.add_node("rag_retriever",        lambda s: rag_retriever(s, config=config))
    graph.add_node("hypothesis_generator", lambda s: hypothesis_generator(s, config=config))
    graph.add_node("judge_panel",          lambda s: judge_panel(s, config=config))
    graph.add_node("reflector",            lambda s: reflector(s, config=config))

    # Validator (unchanged)
    graph.add_node("mechanism_validator",  lambda s: mechanism_validator_node(s, config))

    # Outer edges
    graph.set_entry_point("context_retriever")
    graph.add_edge("context_retriever",   "few_shot_retriever")
    graph.add_edge("few_shot_retriever",  "uncertainty_router")
    graph.add_conditional_edges(
        "uncertainty_router",
        route_article,
        {"evidence_extractor": "evidence_extractor"},
    )

    # Sub-pipeline edges
    graph.add_edge("evidence_extractor",   "rag_retriever")
    graph.add_edge("rag_retriever",        "hypothesis_generator")
    graph.add_edge("hypothesis_generator", "judge_panel")
    graph.add_edge("judge_panel",          "reflector")
    graph.add_conditional_edges(
        "reflector",
        route_after_reflector,
        {
            REFLECTOR_ROUTE_RETRY:     "hypothesis_generator",
            REFLECTOR_ROUTE_VALIDATOR: "mechanism_validator",
        },
    )

    graph.set_finish_point("mechanism_validator")

    return graph.compile()
```

- [ ] **Step 5: Run topology test to verify it passes**

Run: `uv run pytest tests/agents/test_graph.py -v`

Expected: PASS. If pre-existing tests assert `causality_extractor` is in the graph, update those assertions to assert the sub-pipeline node names instead. (Run the test, read failures, fix only the assertions about removed nodes.)

- [ ] **Step 6: Run the full agents test suite**

Run: `uv run pytest tests/agents/ -v`

Expected: All tests pass. If `tests/agents/test_causality_extractor.py` still imports the old node, the file should remain (the function is still exported via `nodes/__init__.py` for any non-graph callers and for backwards compatibility). No changes required there.

- [ ] **Step 7: Commit**

```bash
git add src/trading_databricks/agents/graph.py src/trading_databricks/agents/nodes/__init__.py tests/agents/test_graph.py
git commit -m "TRD feat: rewire MAS graph with 5-node causality sub-pipeline"
```

---

## Task 10: End-to-end sub-pipeline integration test

**Files:**
- Create: `tests/agents/test_causality_subpipeline.py`

- [ ] **Step 1: Write failing tests**

Create `tests/agents/test_causality_subpipeline.py`:

```python
"""End-to-end integration test for the causality sub-pipeline.

Runs build_mas_graph() with mocked LLM and Vector Search externals and
verifies the happy path and the reflection path produce the expected
terminal state.
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from trading_databricks.agents.config import MASConfig


VALID_EXTRACTION = {
    "treatment_class": "Regulation.etf_ruling",
    "trigger_event": "BlackRock files for spot Ether ETF",
    "mechanism": "etf_inflow_demand",
    "expected_direction": "bullish",
    "evidence_spans": ["BlackRock files for spot Ether ETF"],
    "confidence": 0.78,
    "reasoning": "Filing signals institutional ETF demand grounded in precedent.",
}


def _stub_outer_nodes():
    """Patch the outer nodes so the integration test focuses on the sub-pipeline."""
    return [
        patch("trading_databricks.agents.graph.get_spark_session"),
        patch("trading_databricks.agents.graph.load_whitelist"),
        patch("trading_databricks.agents.nodes.context_retriever.context_retriever", side_effect=lambda s, config: {"market_vol_at_publish": 0.02, "market_trend_at_publish": "up", "dollar_bar_close": 3500.0}),
        patch("trading_databricks.agents.nodes.few_shot_retriever.few_shot_retriever", side_effect=lambda s, config: {"few_shot_examples": [], "retrieval_story_ids": []}),
        patch("trading_databricks.agents.nodes.uncertainty_router.uncertainty_router", side_effect=lambda s, config: {"extraction_path": "standard"}),
    ]


def _entry_state():
    return {
        "headline": "BlackRock files for spot Ether ETF",
        "body": "BlackRock files for spot Ether ETF with the SEC.",
        "published_at": "2026-04-18T10:00:00Z",
        "symbol": "ETHUSDT",
    }


@patch("trading_databricks.agents.nodes.judge_panel._invoke_judge")
@patch("trading_databricks.agents.nodes.hypothesis_generator.load_taxonomy")
@patch("trading_databricks.agents.nodes.hypothesis_generator.load_whitelist")
@patch("trading_databricks.agents.nodes.hypothesis_generator.get_spark_session")
@patch("trading_databricks.agents.nodes.hypothesis_generator.mlflow.deployments.get_deploy_client")
@patch("trading_databricks.agents.nodes.evidence_extractor.mlflow.deployments.get_deploy_client")
@patch("trading_databricks.agents.nodes.rag_retriever._join_market_outcomes")
@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose")
def test_subpipeline_happy_path(
    mock_firehose, mock_join, mock_evidence_client, mock_hyp_client,
    mock_spark, mock_wl, mock_tax, mock_judge,
):
    from trading_databricks.agents.graph import build_mas_graph

    # Stub evidence LLM
    ev_client = MagicMock()
    ev_client.predict.return_value = {"choices": [{"message": {"content": '[{"actor": "BlackRock", "action": "files", "object": "spot Ether ETF", "span": "BlackRock files for spot Ether ETF", "timestamp_hint": null}]'}}]}
    mock_evidence_client.return_value = ev_client

    # Stub RAG
    mock_firehose.return_value = [{"headline": "Fidelity files Bitcoin ETF", "timestamp": "2024-06-15T12:00:00Z"}]
    mock_join.return_value = [{"headline": "Fidelity files Bitcoin ETF", "timestamp": "2024-06-15T12:00:00Z", "return_24h": 0.043, "direction": "up"}]

    # Stub hypothesis LLM
    hyp_client = MagicMock()
    hyp_client.predict.return_value = {"choices": [{"message": {"content": json.dumps(VALID_EXTRACTION)}}]}
    mock_hyp_client.return_value = hyp_client
    mock_tax.return_value = {"Regulation": ["etf_ruling"]}
    mock_wl.return_value = {("Regulation", "etf_inflow_demand"): ["bullish"]}

    # Stub judges — all pass
    mock_judge.side_effect = [
        {"passed": True, "score": "up", "reason": ""},
        {"passed": True, "score": "match", "reason": ""},
        {"passed": True, "score": 0.9, "reason": ""},
    ]

    # Stub the validator's external resources via graph helpers
    with patch("trading_databricks.agents.graph.get_spark_session"), \
         patch("trading_databricks.agents.graph.load_whitelist", return_value={("Regulation", "etf_inflow_demand"): ["bullish"]}), \
         patch("trading_databricks.agents.graph.mlflow.deployments.get_deploy_client") as mock_validator_client:
        mock_validator_client.return_value.predict.return_value = {"data": [{"embedding": [0.1] * 8}]}
        graph = build_mas_graph(config=MASConfig())
        result = graph.invoke(_entry_state())

    assert result["extraction"]["mechanism"] == "etf_inflow_demand"
    assert result["extraction_status"] == "ok"
    assert result["reflection_count"] == 0
    assert result["validation_status"] == "accepted"


@patch("trading_databricks.agents.nodes.judge_panel._invoke_judge")
@patch("trading_databricks.agents.nodes.hypothesis_generator.load_taxonomy")
@patch("trading_databricks.agents.nodes.hypothesis_generator.load_whitelist")
@patch("trading_databricks.agents.nodes.hypothesis_generator.get_spark_session")
@patch("trading_databricks.agents.nodes.hypothesis_generator.mlflow.deployments.get_deploy_client")
@patch("trading_databricks.agents.nodes.evidence_extractor.mlflow.deployments.get_deploy_client")
@patch("trading_databricks.agents.nodes.rag_retriever._join_market_outcomes")
@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose")
def test_subpipeline_reflects_then_passes(
    mock_firehose, mock_join, mock_evidence_client, mock_hyp_client,
    mock_spark, mock_wl, mock_tax, mock_judge,
):
    from trading_databricks.agents.graph import build_mas_graph

    ev_client = MagicMock()
    ev_client.predict.return_value = {"choices": [{"message": {"content": '[{"actor": "BlackRock", "action": "files", "span": "BlackRock files for spot Ether ETF"}]'}}]}
    mock_evidence_client.return_value = ev_client

    mock_firehose.return_value = [{"headline": "x", "timestamp": "t"}]
    mock_join.return_value = [{"headline": "x", "timestamp": "t", "return_24h": 0.04, "direction": "up"}]

    hyp_client = MagicMock()
    hyp_client.predict.return_value = {"choices": [{"message": {"content": json.dumps(VALID_EXTRACTION)}}]}
    mock_hyp_client.return_value = hyp_client
    mock_tax.return_value = {"Regulation": ["etf_ruling"]}
    mock_wl.return_value = {("Regulation", "etf_inflow_demand"): ["bullish"]}

    # Judges fail first call, pass second
    mock_judge.side_effect = [
        # First panel — grounding fails
        {"passed": True, "score": "up", "reason": ""},
        {"passed": True, "score": "match", "reason": ""},
        {"passed": False, "score": 0.3, "reason": "low score"},
        # Second panel — all pass
        {"passed": True, "score": "up", "reason": ""},
        {"passed": True, "score": "match", "reason": ""},
        {"passed": True, "score": 0.9, "reason": ""},
    ]

    with patch("trading_databricks.agents.graph.get_spark_session"), \
         patch("trading_databricks.agents.graph.load_whitelist", return_value={("Regulation", "etf_inflow_demand"): ["bullish"]}), \
         patch("trading_databricks.agents.graph.mlflow.deployments.get_deploy_client") as mock_validator_client:
        mock_validator_client.return_value.predict.return_value = {"data": [{"embedding": [0.1] * 8}]}
        graph = build_mas_graph(config=MASConfig())
        result = graph.invoke(_entry_state())

    assert result["extraction_status"] == "ok_after_reflection"
    assert result["reflection_count"] == 1


@patch("trading_databricks.agents.nodes.judge_panel._invoke_judge")
@patch("trading_databricks.agents.nodes.hypothesis_generator.load_taxonomy")
@patch("trading_databricks.agents.nodes.hypothesis_generator.load_whitelist")
@patch("trading_databricks.agents.nodes.hypothesis_generator.get_spark_session")
@patch("trading_databricks.agents.nodes.hypothesis_generator.mlflow.deployments.get_deploy_client")
@patch("trading_databricks.agents.nodes.evidence_extractor.mlflow.deployments.get_deploy_client")
@patch("trading_databricks.agents.nodes.rag_retriever._join_market_outcomes")
@patch("trading_databricks.agents.nodes.rag_retriever._query_firehose")
def test_subpipeline_budget_exhausted(
    mock_firehose, mock_join, mock_evidence_client, mock_hyp_client,
    mock_spark, mock_wl, mock_tax, mock_judge,
):
    from trading_databricks.agents.graph import build_mas_graph

    ev_client = MagicMock()
    ev_client.predict.return_value = {"choices": [{"message": {"content": "[]"}}]}
    mock_evidence_client.return_value = ev_client

    mock_firehose.return_value = []
    mock_join.return_value = []

    hyp_client = MagicMock()
    hyp_client.predict.return_value = {"choices": [{"message": {"content": json.dumps(VALID_EXTRACTION)}}]}
    mock_hyp_client.return_value = hyp_client
    mock_tax.return_value = {"Regulation": ["etf_ruling"]}
    mock_wl.return_value = {("Regulation", "etf_inflow_demand"): ["bullish"]}

    # Judges always fail
    mock_judge.side_effect = [
        {"passed": False, "score": 0.3, "reason": "fail"} for _ in range(6)
    ]

    with patch("trading_databricks.agents.graph.get_spark_session"), \
         patch("trading_databricks.agents.graph.load_whitelist", return_value={("Regulation", "etf_inflow_demand"): ["bullish"]}), \
         patch("trading_databricks.agents.graph.mlflow.deployments.get_deploy_client") as mock_validator_client:
        mock_validator_client.return_value.predict.return_value = {"data": [{"embedding": [0.1] * 8}]}
        graph = build_mas_graph(config=MASConfig())
        result = graph.invoke(_entry_state())

    assert result["extraction_status"] == "judge_failed_accepted"
    assert result["reflection_count"] == 1  # exactly max_reflections
    # Worst case = 2 hypothesis_generator calls
    assert hyp_client.predict.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail (or pass — they may already work)**

Run: `uv run pytest tests/agents/test_causality_subpipeline.py -v`

Expected: PASS — all individual nodes are implemented; this test just verifies they compose correctly through `build_mas_graph`.

- [ ] **Step 3: Investigate any failures**

Common failure modes:
- LangGraph state-merge dropped a key → check that the failing node returned a dict with the right key.
- `extraction_status` ended up unexpected → trace through the reflector's logic with the actual judge result dict.

If you hit issues, run with `pytest -s -vv` to see logger output.

- [ ] **Step 4: Run the full test suite to confirm no regressions**

Run: `uv run pytest -v`

Expected: All tests green.

- [ ] **Step 5: Commit**

```bash
git add tests/agents/test_causality_subpipeline.py
git commit -m "TRD test: end-to-end causality sub-pipeline integration"
```

---

## Task 11: Verify eval harness still works with the new pipeline

**Files:**
- (No code changes expected; only verification)

- [ ] **Step 1: Run the eval harness's existing tests**

Run: `uv run pytest tests/agents/eval/ -v`

Expected: All pass. The eval harness consumes the graph's output state via standard keys (`extraction`, `validation_status`); our new pipeline preserves all those keys, so no harness changes are required.

- [ ] **Step 2: Inspect runner.py for hardcoded references to removed nodes**

Run: `Grep -rn "causality_extractor\|_mechanism_validator_node" src/trading_databricks/agents/eval/`

Expected output: zero references that would break (the function is still exported from `nodes/__init__.py`). If anything inside the eval harness directly references graph node names, audit and update only what's needed.

- [ ] **Step 3: Verify final state is unchanged**

Run: `uv run pytest -v`

Expected: All green.

- [ ] **Step 4: Commit (no-op or any audit fixes)**

If no fixes were needed, skip this commit. Otherwise:

```bash
git add -p
git commit -m "TRD chore: align eval harness with new sub-pipeline node names"
```

---

## Verification

After all tasks complete, run:

```bash
uv run pytest -v
uv run pytest tests/agents/test_causality_subpipeline.py -v
uv run pytest tests/agents/nodes/ -v
```

Expected: every test green, including the 3 sub-pipeline integration tests covering happy path, reflection path, and budget-exhausted path.

Inspect the new graph topology in a notebook or REPL:

```python
from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.graph import build_mas_graph

g = build_mas_graph(config=MASConfig())
print(g.get_graph().draw_ascii())
```

Expected: shows `evidence_extractor → rag_retriever → hypothesis_generator → judge_panel → reflector → {hypothesis_generator | mechanism_validator}`.

---

## Out of Scope (deferred to follow-up tickets)

- Provisioning the `news_firehose_idx` Vector Search index and associated indexing pipeline (`_query_firehose` raises NotImplementedError until then).
- Real Spark implementation of `_join_market_outcomes` against `ohlc_gold` (raises NotImplementedError; mocked in tests).
- Latency profiling and parallelisation of `judge_panel` via `Send` API.
- Eval-harness wiring for `prompt_versions` of the new prompts (`evidence_extractor`, `hypothesis_generator`).
- Persistence of `mechanism_candidate` to `04_eval.mechanism_candidates_review` (validator change — separate ticket).
