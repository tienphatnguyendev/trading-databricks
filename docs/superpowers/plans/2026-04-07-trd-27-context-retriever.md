# TRD-27 Context Retriever Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize the `MASState` definition and implement the `context_retriever` node and `fetch_market_context` UC tool as stubs returning `None` for market context fields until TRD-5 is complete.

**Architecture:** 
- A new file `agents/state.py` will hold the centralized `MASState` TypedDict.
- `few_shot_retriever.py` will be refactored to use the centralized state.
- A new file `agents/nodes/context_retriever.py` will define the node returning `None` for the context fields.
- `agents/tools.py` will get a new `fetch_market_context` function stub.

**Tech Stack:** Python 3.12, Pytest, TypedDict

---

### Task 1: Centralize MASState and Refactor Existing Code

**Files:**
- Create: `src/trading_databricks/agents/state.py`
- Modify: `src/trading_databricks/agents/nodes/few_shot_retriever.py`
- Create: `tests/agents/test_state.py`

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_state.py`:
```python
from trading_databricks.agents.state import MASState

def test_mas_state_instantiation():
    """Verify MASState can be instantiated with all required fields."""
    state: MASState = {
        "headline": "Test",
        "few_shot_examples": [],
        "retrieval_story_ids": [],
        "published_at": "2026-04-07T12:00:00Z",
        "market_vol_at_publish": None,
        "market_trend_at_publish": None,
        "dollar_bar_close": None,
    }
    assert state["headline"] == "Test"
    assert state["market_vol_at_publish"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_state.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'trading_databricks.agents.state'"

- [ ] **Step 3: Write minimal implementation**

Create `src/trading_databricks/agents/state.py`:
```python
"""Centralized LangGraph state definitions for MAS agents."""
from __future__ import annotations

from typing import TypedDict

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
```

Modify `src/trading_databricks/agents/nodes/few_shot_retriever.py`:
Remove the local `MASState` class definition (lines 17-26 roughly).
Add import at the top (after `from typing import TypedDict` or similar):
```python
from trading_databricks.agents.state import MASState
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agents/test_state.py -v`
Expected: PASS

Run: `uv run pytest tests/agents/test_few_shot_retriever.py -v`
Expected: PASS (to ensure refactoring didn't break existing tests)

- [ ] **Step 5: Commit**

```bash
git add tests/agents/test_state.py src/trading_databricks/agents/state.py src/trading_databricks/agents/nodes/few_shot_retriever.py
git commit -m "TRD-27 feat: centralize MASState definition in agents.state"
```

---

### Task 2: Implement Context Retriever Node and UC Tool

**Files:**
- Create: `src/trading_databricks/agents/nodes/context_retriever.py`
- Modify: `src/trading_databricks/agents/tools.py`
- Create: `tests/agents/test_context_retriever.py`

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_context_retriever.py`:
```python
from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.nodes.context_retriever import context_retriever
from trading_databricks.agents.state import MASState
from trading_databricks.agents.tools import fetch_market_context

def test_context_retriever_stub():
    """Verify context_retriever returns None for market fields."""
    initial_state: MASState = {
        "headline": "Bitcoin surges",
        "published_at": "2026-04-07T12:00:00Z"
    }
    config = MASConfig()
    
    new_state = context_retriever(initial_state, config=config)
    
    assert new_state["headline"] == "Bitcoin surges"
    assert new_state["published_at"] == "2026-04-07T12:00:00Z"
    assert new_state["market_vol_at_publish"] is None
    assert new_state["market_trend_at_publish"] is None
    assert new_state["dollar_bar_close"] is None

def test_fetch_market_context_tool():
    """Verify fetch_market_context tool returns None for market fields."""
    result = fetch_market_context("2026-04-07T12:00:00Z")
    
    assert result == {
        "market_vol_at_publish": None,
        "market_trend_at_publish": None,
        "dollar_bar_close": None,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_context_retriever.py -v`
Expected: FAIL with "ImportError: cannot import name 'context_retriever'"

- [ ] **Step 3: Write minimal implementation**

Create `src/trading_databricks/agents/nodes/context_retriever.py`:
```python
"""Context retriever node for MAS agent pipeline.

V1 stub: returns NULL for market context fields until TRD-5 is complete.
"""
from __future__ import annotations

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.state import MASState

def context_retriever(state: MASState, *, config: MASConfig) -> MASState:
    """Fetch pre-treatment market context X.
    
    V1 stub: gold.dollar_bars not yet available (TRD-5 dependency).
    Returns NULL for all market context fields.
    Schema fields 18-20 in CAUSAL_FEATURES_SCHEMA remain nullable for this reason.
    
    V2 (post-TRD-5): query gold.dollar_bars for bar nearest to
    state["published_at"], extract vol, trend, close.
    """
    return {
        **state,
        "market_vol_at_publish": None,
        "market_trend_at_publish": None,
        "dollar_bar_close": None,
    }
```

Modify `src/trading_databricks/agents/tools.py` by adding the following at the bottom of the file:
```python
def fetch_market_context(published_at: str) -> dict:
    """Fetch market context features for a given publication time.
    
    # TODO: implement post TRD-5
    
    Args:
        published_at: Timestamp string of the article publication
        
    Returns:
        Dict containing market_vol_at_publish, market_trend_at_publish,
        and dollar_bar_close. V1 returns None for all fields.
    """
    return {
        "market_vol_at_publish": None,
        "market_trend_at_publish": None,
        "dollar_bar_close": None,
    }
```

Modify `src/trading_databricks/agents/nodes/__init__.py` to export `context_retriever`:
```python
from .context_retriever import context_retriever
from .few_shot_retriever import few_shot_retriever
from .mechanism_validator import validate_extraction

__all__ = [
    "context_retriever",
    "few_shot_retriever",
    "validate_extraction",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agents/test_context_retriever.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/agents/test_context_retriever.py src/trading_databricks/agents/nodes/context_retriever.py src/trading_databricks/agents/tools.py src/trading_databricks/agents/nodes/__init__.py
git commit -m "TRD-27 feat: implement context_retriever node and fetch_market_context tool as stubs"
```