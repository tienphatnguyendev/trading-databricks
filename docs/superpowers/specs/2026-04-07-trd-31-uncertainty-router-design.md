# Uncertainty Router Design (TRD-31)

**Date:** 2026-04-07
**Related Tickets:** TRD-31, TRD-6 (Parent)
**Status:** Approved
**Author:** AI Agent

## 1. Overview

This document specifies the design for Module A of the V3 reliability layer: the Uncertainty Router (`uncertainty_router` node) inspired by Active Bayesian Causal Inference (ABCI). 

The goal is to focus compute budget on edges with the highest posterior entropy (uncertainty). In the MAS analogy, Vector Search (VS) cosine similarity serves as a proxy for extraction certainty. Routine articles with strong VS priors take a "fast" path ($0.01), while novel articles with no VS match take a "careful" path ($0.09 with k=3 self-consistency). The expected distribution is 60% fast, 30% standard, and 10% careful.

## 2. Architecture & LangGraph Topology

Based on discussions and architectural refinement, the `uncertainty_router` will be implemented as a **pure Node** in the LangGraph, rather than a LangGraph conditional edge function.

**Why a pure Node?**
1. **State Mutation:** It allows us to legally update the state by returning `{"extraction_path": path}`, adhering to LangGraph's state reducer mechanics.
2. **Simplified Graph:** We avoid cluttering the graph with redundant extractor nodes (`fast_extractor`, `standard_extractor`, etc.). The router simply calculates the path and flows unconditionally into a single `causality_extractor`.
3. **Testability:** The node can be unit-tested purely with mock VS scores, without needing to mock LLM chains.

**Topology:**
```text
few_shot_retriever -> uncertainty_router -> causality_extractor
```

## 3. Component Design

### 3.1 State Modification (`src/trading_databricks/agents/state.py`)

The `MASState` `TypedDict` will be extended to include the `extraction_path` field, which will ultimately map to the output schema.

```python
class MASState(TypedDict, total=False):
    # ... existing fields ...
    extraction_path: str  # Literal["fast", "standard", "careful"]
```

### 3.2 The Router Node (`src/trading_databricks/agents/nodes/uncertainty_router.py`)

The node will be a straightforward function that takes the state and config, evaluates the retrieval results, and returns the path update.

```python
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

### 3.3 Public API (`src/trading_databricks/agents/nodes/__init__.py`)

The `uncertainty_router` will be exported from the `nodes` package alongside `few_shot_retriever` and others to facilitate clean imports when constructing the graph.

## 4. Testing Strategy (`tests/agents/test_uncertainty_router.py`)

The unit tests will verify the routing logic against the acceptance criteria defined in TRD-31:

1. **Fast Path:** `top_similarity >= 0.85` AND `len(examples) >= 2` -> returns `{"extraction_path": "fast"}`
2. **Standard Path:** `0.50 <= top_similarity < 0.85` AND `len(examples) >= 2` -> returns `{"extraction_path": "standard"}`
3. **Careful Path (Low Similarity):** `top_similarity < 0.50` -> returns `{"extraction_path": "careful"}`
4. **Careful Path (Cold Start/Insufficient Results):** Empty examples list OR `len(examples) < 2` -> returns `{"extraction_path": "careful"}`
5. **Config Validation:** Ensure the node respects overrides passed via `MASConfig` (e.g., custom thresholds).

## 5. Deployment & Dependencies

* **Dependencies:** None beyond the existing `MASState` and `MASConfig`.
* **Deployment:** This node will be integrated into the main MAS LangGraph in TRD-32.
