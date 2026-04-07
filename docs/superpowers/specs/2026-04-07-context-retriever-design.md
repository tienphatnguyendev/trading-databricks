# Context Retriever Design (TRD-27)

## Business Objective
Implement the `context_retriever` node for the MAS Agent Layer, acting as a stub until TRD-5 (dollar bars) is complete. The node needs to return `NULL` (`None`) for all market context features required by the causal identification model, allowing the pipeline to work end-to-end today while preserving the schema contract.

## State Centralization (Agent State)
Currently, LangGraph state `MASState` is defined partially and locally in nodes like `few_shot_retriever`. To support the new node and cleaner typings, `MASState` will be centralized:
- **New File:** `src/trading_databricks/agents/state.py`
- **Definition:** `MASState(TypedDict, total=False)` containing fields from both current and new nodes:
    - `headline: str`
    - `few_shot_examples: list[dict]`
    - `retrieval_story_ids: list[str]`
    - `published_at: str` (or datetime)
    - `market_vol_at_publish: float | None`
    - `market_trend_at_publish: str | None`
    - `dollar_bar_close: float | None`
- **Refactoring:** Update `few_shot_retriever.py` to import `MASState` from `agents.state` instead of defining it locally.

## Context Retriever Node
- **File:** `src/trading_databricks/agents/nodes/context_retriever.py`
- **Signature:** `def context_retriever(state: MASState, *, config: MASConfig) -> MASState:`
- **Logic:** Stub that returns a new state dictionary copying all existing state and explicitly setting:
    - `market_vol_at_publish = None`
    - `market_trend_at_publish = None`
    - `dollar_bar_close = None`

## UC Function Stub
- **File:** `src/trading_databricks/agents/tools.py`
- **Function:** `fetch_market_context(published_at: str)`
- **Logic:** Return a dictionary with the three market context fields set to `None`.
- **Annotation:** Add a `# TODO: implement post TRD-5` comment.

## Testing
- **File:** `tests/agents/test_context_retriever.py`
- **Tests:**
    - Verify `context_retriever` node correctly preserves existing state and appends the 3 `None` fields.
    - Verify `fetch_market_context` tool correctly returns the `None` dictionary.