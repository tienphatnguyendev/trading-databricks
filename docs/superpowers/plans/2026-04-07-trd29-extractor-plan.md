# TRD-29: Causality Extractor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `causality_extractor` LangGraph node for the trading-databricks MAS, orchestrating fast, standard, and careful extraction paths using the Databricks `mlflow.deployments` client.

**Architecture:** 
1. **Prompts:** Create modular formatters in `prompts.py` to inject taxonomy (compact/full), whitelist, and dynamic few-shot examples into path-specific prompt templates.
2. **Node Logic:** Create the `causality_extractor` node in `causality_extractor.py` that reads `MASState`, fetches the taxonomy/whitelist, formats the prompt, calls `databricks-claude-sonnet` (with `n=1` or `n=3`), performs a majority vote (for careful path), and updates `MASState` with the resulting JSON extraction.
3. **Testing:** Create comprehensive mocked tests in `test_causality_extractor.py` to verify routing parameters (`temperature`, `n`), system prompt construction, parsing, and majority voting.

**Tech Stack:** Python 3.11+, PySpark, MLflow Deployments, Databricks Model Serving, `pytest`.

---

### Task 1: Prompt Formatters & Templates (`prompts.py`)

**Files:**
- Create: `src/trading_databricks/agents/prompts.py`
- Modify: `tests/agents/test_prompts.py` (Create)

- [ ] **Step 1.1: Write failing tests for prompt formatters**
Create `tests/agents/test_prompts.py`:
```python
"""Tests for causality extractor prompts."""
import pytest
from trading_databricks.agents.prompts import (
    format_compact_taxonomy,
    format_full_taxonomy,
    format_whitelist,
    format_few_shot,
    build_system_prompt
)

def test_format_compact_taxonomy():
    taxonomy = {"Macro": ["rate_decision", "inflation"], "Security": ["exchange_hack"]}
    res = format_compact_taxonomy(taxonomy)
    assert res == "Macro.inflation, Macro.rate_decision, Security.exchange_hack"

def test_format_full_taxonomy():
    taxonomy = {"Macro": ["rate_decision"]}
    res = format_full_taxonomy(taxonomy)
    assert "- Macro:" in res
    assert "  - Macro.rate_decision" in res

def test_format_whitelist():
    whitelist = {("Macro", "risk_off"): ["bearish"]}
    res = format_whitelist(whitelist)
    assert "L1 Category: Macro" in res
    assert "  - risk_off" in res

def test_format_few_shot():
    examples = [{"trigger_event": "Hack", "causal_chain_json": '{"a": 1}'}]
    res = format_few_shot(examples, 1)
    assert "Headline: Hack" in res
    assert "Extraction:" in res
    
def test_build_system_prompt_fast():
    taxonomy = {"A": ["b"]}
    whitelist = {("A", "c"): ["bullish"]}
    prompt = build_system_prompt(taxonomy, whitelist, [], "fast")
    assert "A.b" in prompt
    assert "MECHANISM WHITELIST" not in prompt

def test_build_system_prompt_careful():
    taxonomy = {"A": ["b"]}
    whitelist = {("A", "c"): ["bullish"]}
    prompt = build_system_prompt(taxonomy, whitelist, [], "careful")
    assert "- A:" in prompt
    assert "CAREFUL PATH: Please provide explicit chain-of-thought" in prompt
```

- [ ] **Step 1.2: Run tests to verify they fail**
Run: `uv run pytest tests/agents/test_prompts.py -v`
Expected: FAIL (ModuleNotFoundError or ImportError)

- [ ] **Step 1.3: Implement `prompts.py`**
Create `src/trading_databricks/agents/prompts.py`:
```python
"""Prompt templates and formatting for the MAS causality extractor."""
from __future__ import annotations

def format_compact_taxonomy(taxonomy: dict[str, list[str]]) -> str:
    classes = []
    for l1, l2s in sorted(taxonomy.items()):
        for l2 in sorted(l2s):
            classes.append(f"{l1}.{l2}")
    return ", ".join(classes)

def format_full_taxonomy(taxonomy: dict[str, list[str]]) -> str:
    lines = []
    for l1, l2s in sorted(taxonomy.items()):
        lines.append(f"- {l1}:")
        for l2 in sorted(l2s):
            lines.append(f"  - {l1}.{l2}")
    return "\n".join(lines)

def format_whitelist(whitelist: dict[tuple[str, str], list[str]]) -> str:
    grouped: dict[str, list[str]] = {}
    for (l1, mechanism), _ in whitelist.items():
        grouped.setdefault(l1, []).append(mechanism)

    lines = []
    for l1, mechs in sorted(grouped.items()):
        lines.append(f"L1 Category: {l1}")
        for m in sorted(mechs):
            lines.append(f"  - {m}")
    return "\n".join(lines)

def format_few_shot(examples: list[dict], limit: int) -> str:
    selected = examples[:limit]
    if not selected:
        return "No examples available."
    formatted = []
    for i, ex in enumerate(selected, 1):
        chain_json = ex.get('causal_chain_json', '{}')
        if not isinstance(chain_json, str):
            import json
            chain_json = json.dumps(chain_json)
        formatted.append(f"Example {i}:\nHeadline: {ex.get('trigger_event', '')}\nExtraction:\n{chain_json}")
    return "\n\n".join(formatted)

_BASE_SYSTEM_PROMPT = """You are a financial news causality extractor for BTC/ETH markets.

TASK: Extract a single-hop causal chain from the article.
Respond only with the JSON extraction object. Do not include markdown formatting like ```json.

TREATMENT TAXONOMY (classify the event into one of these):
{taxonomy_injection}

MECHANISM WHITELIST (the mechanism MUST be one of these):
{whitelist_injection}

SIMILAR PAST EXTRACTIONS (reference, do not copy blindly):
{few_shot_injection}

RULES:
1. Copy evidence_spans VERBATIM from the article body — exact character sequences.
2. mechanism must exactly match a whitelist entry for the chosen L1 category.
3. If the article does not describe a clear causal event for BTC/ETH, respond with {{"is_impactful": false}}.
4. Use retrieved examples as reference patterns, but extract based on THIS article's actual content.
"""

_FAST_SYSTEM_PROMPT = """You are a financial news causality extractor for BTC/ETH markets.

TASK: Extract a single-hop causal chain from the article.
Respond only with the JSON extraction object. Do not include markdown formatting like ```json.

CRITICAL: `treatment_class` MUST be exactly one of the following 34 categories, or you will fail:
{taxonomy_injection}

SIMILAR PAST EXTRACTIONS (reference, do not copy blindly):
{few_shot_injection}

RULES:
1. Copy evidence_spans VERBATIM from the article body — exact character sequences.
2. If the article does not describe a clear causal event for BTC/ETH, respond with {{"is_impactful": false}}.
3. Use retrieved examples as reference patterns, but extract based on THIS article's actual content.
"""

_CO_T_INSTRUCTION = """
5. CAREFUL PATH: Please provide explicit chain-of-thought reasoning before your final extraction. Explain your classification and mechanism selection in the `reasoning` field.
"""

def build_system_prompt(
    taxonomy: dict[str, list[str]],
    whitelist: dict[tuple[str, str], list[str]],
    few_shot_examples: list[dict],
    path: str,
) -> str:
    if path == "fast":
        return _FAST_SYSTEM_PROMPT.format(
            taxonomy_injection=format_compact_taxonomy(taxonomy),
            few_shot_injection=format_few_shot(few_shot_examples, 3),
        )
    
    prompt = _BASE_SYSTEM_PROMPT.format(
        taxonomy_injection=format_full_taxonomy(taxonomy),
        whitelist_injection=format_whitelist(whitelist),
        few_shot_injection=format_few_shot(few_shot_examples, 5),
    )
    if path == "careful":
        prompt += _CO_T_INSTRUCTION
    return prompt
```

- [ ] **Step 1.4: Run tests to verify they pass**
Run: `uv run pytest tests/agents/test_prompts.py -v`
Expected: PASS

- [ ] **Step 1.5: Commit**
Run: 
```bash
git add src/trading_databricks/agents/prompts.py tests/agents/test_prompts.py
git commit -m "TRD-29 feat: prompt formatters and templates for causality extractor"
```

---

### Task 2: Extractor Node Logic & Majority Voting (`causality_extractor.py`)

**Files:**
- Create: `src/trading_databricks/agents/nodes/causality_extractor.py`
- Modify: `tests/agents/test_causality_extractor.py` (Create)

- [ ] **Step 2.1: Write failing tests for majority vote and JSON parsing**
Create `tests/agents/test_causality_extractor.py`:
```python
"""Tests for the causality_extractor node."""
import pytest
from trading_databricks.agents.nodes.causality_extractor import (
    parse_json_response,
    majority_vote
)

def test_parse_json_response():
    assert parse_json_response('{"a": 1}') == {"a": 1}
    assert parse_json_response('```json\n{"b": 2}\n```') == {"b": 2}
    assert parse_json_response('invalid') is None

def test_majority_vote_consensus():
    extractions = [
        {"treatment_class": "A", "mechanism": "m", "expected_direction": "bullish", "confidence": 0.8},
        {"treatment_class": "A", "mechanism": "m", "expected_direction": "bullish", "confidence": 0.7},
        {"treatment_class": "B", "mechanism": "n", "expected_direction": "bearish", "confidence": 0.9}
    ]
    winner = majority_vote(extractions)
    assert winner["confidence"] == 0.8  # First occurrence of consensus

def test_majority_vote_tie_breaker():
    extractions = [
        {"treatment_class": "A", "mechanism": "m", "expected_direction": "bullish", "confidence": 0.6},
        {"treatment_class": "B", "mechanism": "n", "expected_direction": "bearish", "confidence": 0.9},
        {"treatment_class": "C", "mechanism": "o", "expected_direction": "volatile", "confidence": 0.7}
    ]
    winner = majority_vote(extractions)
    assert winner["treatment_class"] == "B"
    assert winner["confidence"] == 0.6  # Penalized: max(0, 0.9 - 0.3)
```

- [ ] **Step 2.2: Run tests to verify they fail**
Run: `uv run pytest tests/agents/test_causality_extractor.py -v`
Expected: FAIL

- [ ] **Step 2.3: Implement parsing and voting logic**
Create `src/trading_databricks/agents/nodes/causality_extractor.py`:
```python
"""Causality extractor node for MAS."""
from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

def parse_json_response(text: str) -> dict[str, Any] | None:
    """Parse JSON from LLM response, stripping markdown wrappers if present."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON from LLM: {text[:100]}...")
        return None

def majority_vote(extractions: list[dict]) -> dict:
    """Select the majority consensus extraction, or highest confidence if tie.
    
    Consensus key: (treatment_class, mechanism, expected_direction).
    If no consensus, penalizes confidence by 0.3.
    """
    valid = [e for e in extractions if e]
    if not valid:
        return {}

    # Build consensus keys
    keys = []
    for ex in valid:
        k = (
            ex.get("treatment_class"),
            ex.get("mechanism"),
            ex.get("expected_direction")
        )
        keys.append(k)
        
    counts = Counter(keys)
    most_common_key, count = counts.most_common(1)[0]
    
    if count >= 2:
        # Return first matching extraction
        for ex, k in zip(valid, keys):
            if k == most_common_key:
                return ex
    
    # Tie-breaker: highest confidence
    best_ex = max(valid, key=lambda x: float(x.get("confidence", 0.0)))
    penalized_conf = max(0.0, float(best_ex.get("confidence", 0.0)) - 0.3)
    best_ex["confidence"] = penalized_conf
    return best_ex
```

- [ ] **Step 2.4: Run tests to verify they pass**
Run: `uv run pytest tests/agents/test_causality_extractor.py -v`
Expected: PASS

- [ ] **Step 2.5: Commit**
Run:
```bash
git add src/trading_databricks/agents/nodes/causality_extractor.py tests/agents/test_causality_extractor.py
git commit -m "TRD-29 feat: causality extractor JSON parser and majority vote logic"
```

---

### Task 3: Node Implementation (`causality_extractor.py`)

**Files:**
- Modify: `src/trading_databricks/agents/nodes/causality_extractor.py`
- Modify: `tests/agents/test_causality_extractor.py`

- [ ] **Step 3.1: Write failing node tests**
Append to `tests/agents/test_causality_extractor.py`:
```python
from unittest.mock import MagicMock, patch
from trading_databricks.agents.nodes.causality_extractor import causality_extractor
from trading_databricks.agents.config import MASConfig

@patch("trading_databricks.agents.nodes.causality_extractor.mlflow.deployments.get_deploy_client")
@patch("trading_databricks.agents.nodes.causality_extractor.get_spark_session")
@patch("trading_databricks.agents.nodes.causality_extractor.load_taxonomy")
@patch("trading_databricks.agents.nodes.causality_extractor.load_whitelist")
def test_causality_extractor_fast_path(mock_whitelist, mock_taxonomy, mock_spark, mock_get_client):
    mock_taxonomy.return_value = {"A": ["b"]}
    mock_whitelist.return_value = {("A", "m"): ["bullish"]}
    
    mock_client = MagicMock()
    mock_client.predict.return_value = {
        "choices": [{"message": {"content": '{"treatment_class": "A.b"}'}}]
    }
    mock_get_client.return_value = mock_client
    
    state = {
        "headline": "Test",
        "body": "Text",
        "extraction_path": "fast",
        "few_shot_examples": []
    }
    config = MASConfig()
    
    new_state = causality_extractor(state, config)
    
    assert new_state["extraction"] == {"treatment_class": "A.b"}
    
    call_kwargs = mock_client.predict.call_args[1]
    assert call_kwargs["inputs"]["n"] == 1
    assert call_kwargs["inputs"]["temperature"] == 0.0

@patch("trading_databricks.agents.nodes.causality_extractor.mlflow.deployments.get_deploy_client")
@patch("trading_databricks.agents.nodes.causality_extractor.get_spark_session")
@patch("trading_databricks.agents.nodes.causality_extractor.load_taxonomy")
@patch("trading_databricks.agents.nodes.causality_extractor.load_whitelist")
def test_causality_extractor_careful_path(mock_whitelist, mock_taxonomy, mock_spark, mock_get_client):
    mock_taxonomy.return_value = {"A": ["b"]}
    mock_whitelist.return_value = {("A", "m"): ["bullish"]}
    
    mock_client = MagicMock()
    mock_client.predict.return_value = {
        "choices": [
            {"message": {"content": '{"treatment_class": "A.b", "confidence": 0.8}'}},
            {"message": {"content": '{"treatment_class": "A.b", "confidence": 0.9}'}},
            {"message": {"content": '{"treatment_class": "C.d", "confidence": 0.7}'}}
        ]
    }
    mock_get_client.return_value = mock_client
    
    state = {
        "headline": "Test",
        "body": "Text",
        "extraction_path": "careful",
        "few_shot_examples": []
    }
    config = MASConfig()
    
    new_state = causality_extractor(state, config)
    
    assert new_state["extraction"]["treatment_class"] == "A.b"
    assert new_state["extraction"]["confidence"] == 0.8
    
    call_kwargs = mock_client.predict.call_args[1]
    assert call_kwargs["inputs"]["n"] == 3
    assert call_kwargs["inputs"]["temperature"] == 0.7
```

- [ ] **Step 3.2: Run tests to verify they fail**
Run: `uv run pytest tests/agents/test_causality_extractor.py -v`
Expected: FAIL (causality_extractor not defined)

- [ ] **Step 3.3: Implement node**
Update `src/trading_databricks/agents/nodes/causality_extractor.py` to include:
```python
import mlflow.deployments
from trading_databricks.agents.nodes.few_shot_retriever import get_spark_session, MASState
from trading_databricks.agents.taxonomy import load_taxonomy, load_whitelist
from trading_databricks.agents.prompts import build_system_prompt
from trading_databricks.agents.config import MASConfig

def causality_extractor(state: MASState, config: MASConfig) -> MASState:
    """Extract causal chain using LLM with path-based routing."""
    path = state.get("extraction_path", "standard")
    
    spark = get_spark_session()
    taxonomy = load_taxonomy(spark, catalog=config.catalog, schema=config.schema)
    whitelist = load_whitelist(spark, catalog=config.catalog, schema=config.schema)
    
    system_prompt = build_system_prompt(
        taxonomy,
        whitelist,
        state.get("few_shot_examples", []),
        path
    )
    
    user_msg = f"Headline: {state['headline']}\nBody excerpt: {state['body'][:config.gatekeeper.text_window]}"
    if path in ("standard", "careful"):
        user_msg = f"Headline: {state['headline']}\nBody: {state['body']}"
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg}
    ]
    
    temperature = 0.7 if path == "careful" else 0.0
    n_choices = 3 if path == "careful" else 1
    
    client = mlflow.deployments.get_deploy_client("databricks")
    
    logger.info(f"Calling LLM {config.llm_endpoint} for path={path}, n={n_choices}, temp={temperature}")
    response = client.predict(
        endpoint=config.llm_endpoint,
        inputs={
            "messages": messages,
            "temperature": temperature,
            "n": n_choices,
            "max_tokens": 1000
        }
    )
    
    extractions = []
    for choice in response.get("choices", []):
        text = choice.get("message", {}).get("content", "")
        parsed = parse_json_response(text)
        if parsed:
            extractions.append(parsed)
            
    if not extractions:
        logger.error("LLM failed to produce valid JSON across all choices.")
        state["extraction"] = None
        return state
        
    if path == "careful":
        final_ext = majority_vote(extractions)
    else:
        final_ext = extractions[0]
        
    state["extraction"] = final_ext
    return state
```

- [ ] **Step 3.4: Run tests to verify they pass**
Run: `uv run pytest tests/agents/test_causality_extractor.py -v`
Expected: PASS

- [ ] **Step 3.5: Commit**
Run:
```bash
git add src/trading_databricks/agents/nodes/causality_extractor.py tests/agents/test_causality_extractor.py
git commit -m "TRD-29 feat: causality_extractor node orchestrates LLM endpoints via databricks deployments"
```

---

### Task 4: Export Node (`__init__.py`)

**Files:**
- Modify: `src/trading_databricks/agents/nodes/__init__.py`

- [ ] **Step 4.1: Export the node**
Modify `src/trading_databricks/agents/nodes/__init__.py` to import and export `causality_extractor`.
```python
"""LangGraph node implementations for the MAS agent pipeline."""
from __future__ import annotations

from trading_databricks.agents.nodes.few_shot_retriever import (
    STATIC_FEW_SHOT_EXAMPLES,
    few_shot_retriever,
)
from trading_databricks.agents.nodes.mechanism_validator import validate_extraction
from trading_databricks.agents.nodes.causality_extractor import causality_extractor

__all__ = [
    "STATIC_FEW_SHOT_EXAMPLES",
    "few_shot_retriever",
    "validate_extraction",
    "causality_extractor",
]
```

- [ ] **Step 4.2: Commit**
Run:
```bash
git add src/trading_databricks/agents/nodes/__init__.py
git commit -m "TRD-29 feat: export causality_extractor from nodes"
```
