# Uncertainty Router Implementation Plan (TRD-31)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Module A of the V3 reliability layer—the ABCI-inspired `uncertainty_router` Node—to dynamically determine the causal extraction path (fast, standard, or careful) based on Vector Search similarity scores.

**Architecture:** We are creating a pure LangGraph Node (`uncertainty_router`) that reads `few_shot_examples` from `MASState` and compares the top similarity score against thresholds in `MASConfig`. The node returns a dict with `{"extraction_path": path}` to update the state reducer.

**Tech Stack:** Python 3.12+, Pytest, LangGraph (State TypedDicts), Databricks MAS Architecture.

---

### Task 1: Update MASState schema

**Files:**
- Modify: `src/trading_databricks/agents/state.py`

- [ ] **Step 1: Write the failing test**

```python
# Create file: tests/agents/test_state.py (or append to it)
from trading_databricks.agents.state import MASState
from typing import get_type_hints

def test_mas_state_has_extraction_path():
    hints = get_type_hints(MASState)
    assert "extraction_path" in hints
    assert hints["extraction_path"] is str
```

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/agents/test_state.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**
Edit `src/trading_databricks/agents/state.py` to add `extraction_path: str`:
```python
class MASState(TypedDict, total=False):
    """Type hint for MAS LangGraph state.
    
    total=False makes all fields optional, as nodes only interact with
    portions of the state.
    """
    headline: str
    few_shot_examples: list[dict]
    retrieval_story_ids: list[str]
    published_at: str
    market_vol_at_publish: float | None
    market_trend_at_publish: str | None
    dollar_bar_close: float | None
    extraction_path: str
```

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/agents/test_state.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/trading_databricks/agents/state.py tests/agents/test_state.py
git commit -m "TRD-31 feat: Add extraction_path to MASState schema"
```

---

### Task 2: Implement Uncertainty Router Logic & Tests

**Files:**
- Create: `src/trading_databricks/agents/nodes/uncertainty_router.py`
- Create: `tests/agents/test_uncertainty_router.py`

- [ ] **Step 1: Write the failing test**
Create `tests/agents/test_uncertainty_router.py`:
```python
import pytest
from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.nodes.uncertainty_router import uncertainty_router

@pytest.fixture
def config():
    return MASConfig()

def test_router_cold_start(config):
    # Empty examples
    state = {"few_shot_examples": []}
    result = uncertainty_router(state, config)
    assert result == {"extraction_path": "careful"}

def test_router_insufficient_results(config):
    # Less than min_vs_results (which is 2 by default)
    state = {"few_shot_examples": [{"similarity_score": 0.99}]}
    result = uncertainty_router(state, config)
    assert result == {"extraction_path": "careful"}

def test_router_fast_path(config):
    # top_similarity >= 0.85
    state = {
        "few_shot_examples": [
            {"similarity_score": 0.90},
            {"similarity_score": 0.88}
        ]
    }
    result = uncertainty_router(state, config)
    assert result == {"extraction_path": "fast"}

def test_router_standard_path(config):
    # 0.50 <= top_similarity < 0.85
    state = {
        "few_shot_examples": [
            {"similarity_score": 0.70},
            {"similarity_score": 0.65}
        ]
    }
    result = uncertainty_router(state, config)
    assert result == {"extraction_path": "standard"}

def test_router_careful_path_low_similarity(config):
    # top_similarity < 0.50
    state = {
        "few_shot_examples": [
            {"similarity_score": 0.30},
            {"similarity_score": 0.25}
        ]
    }
    result = uncertainty_router(state, config)
    assert result == {"extraction_path": "careful"}
```

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/agents/test_uncertainty_router.py -v`
Expected: FAIL (ModuleNotFoundError for `uncertainty_router`)

- [ ] **Step 3: Write minimal implementation**
Create `src/trading_databricks/agents/nodes/uncertainty_router.py`:
```python
from __future__ import annotations

from trading_databricks.agents.state import MASState
from trading_databricks.agents.config import MASConfig

def uncertainty_router(state: MASState, config: MASConfig) -> dict[str, str]:
    """Determine the extraction path based on retrieval uncertainty.
    
    ABCI insight: focus compute budget on edges with highest posterior entropy.
    VS cosine similarity is the proxy for extraction certainty.
    """
    examples = state.get("few_shot_examples", [])
    
    # Cold start or insufficient examples
    if not examples or len(examples) < config.router.min_vs_results:
        return {"extraction_path": "careful"}
        
    top_similarity = examples[0].get("similarity_score", 0.0)
    
    if top_similarity >= config.router.high_similarity_threshold:
        return {"extraction_path": "fast"}
    elif top_similarity < config.router.low_similarity_threshold:
        return {"extraction_path": "careful"}
        
    return {"extraction_path": "standard"}
```

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/agents/test_uncertainty_router.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/trading_databricks/agents/nodes/uncertainty_router.py tests/agents/test_uncertainty_router.py
git commit -m "TRD-31 feat: Implement uncertainty_router node and unit tests"
```

---

### Task 3: Export Uncertainty Router in __init__.py

**Files:**
- Modify: `src/trading_databricks/agents/nodes/__init__.py`

- [ ] **Step 1: Write the failing test**
Append to `tests/agents/test_uncertainty_router.py`:
```python
def test_uncertainty_router_is_exported():
    import trading_databricks.agents.nodes as nodes
    assert hasattr(nodes, "uncertainty_router")
    assert "uncertainty_router" in nodes.__all__
```

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/agents/test_uncertainty_router.py::test_uncertainty_router_is_exported -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**
Edit `src/trading_databricks/agents/nodes/__init__.py`:
```python
"""LangGraph node implementations for the MAS agent pipeline."""
from __future__ import annotations

from trading_databricks.agents.nodes.context_retriever import context_retriever
from trading_databricks.agents.nodes.few_shot_retriever import (
    STATIC_FEW_SHOT_EXAMPLES,
    few_shot_retriever,
)
from trading_databricks.agents.nodes.mechanism_validator import validate_extraction
from trading_databricks.agents.nodes.causality_extractor import causality_extractor
from trading_databricks.agents.nodes.uncertainty_router import uncertainty_router

__all__ = [
    "STATIC_FEW_SHOT_EXAMPLES",
    "causality_extractor",
    "context_retriever",
    "few_shot_retriever",
    "uncertainty_router",
    "validate_extraction",
]
```

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/agents/test_uncertainty_router.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/trading_databricks/agents/nodes/__init__.py tests/agents/test_uncertainty_router.py
git commit -m "TRD-31 feat: Export uncertainty_router from nodes package"
```
