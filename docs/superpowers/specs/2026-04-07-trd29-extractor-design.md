# TRD-29: Causality Extractor Node Design

**Date:** 2026-04-07
**Version:** 1.0
**Status:** Approved
**Tickets:** TRD-29
**Parent:** TRD-6

## 1. Context

The `causality_extractor` is the only stochastic component in the pipeline. It calls a `sonnet-class` LLM via Databricks Model Serving (`databricks-claude-sonnet`) to extract a structured causal chain JSON from each impactful article.

It receives its extraction path (`fast`, `standard`, `careful`) from the uncertainty router and adjusts prompt verbosity and the number of LLM calls accordingly to optimize latency, cost, and accuracy.

## 2. Architecture & Approach

We will use the **Option C2** approach: relying on the direct `mlflow.deployments.get_deploy_client` and utilizing the `n=3` parameter on the OpenAI-compatible Databricks endpoint to batch `careful` path predictions concurrently in a single network round-trip.

### 2.1 Extraction Paths

* **Fast path:**
  * Uses `n=1` with `temperature=0.0`.
  * Lean prompt: Includes the headline, article excerpt, top-3 dynamic few-shot examples.
  * Taxonomy is flattened into a 34-item string. Whitelist and CoT instructions are omitted to save tokens/latency.
* **Standard path:**
  * Uses `n=1` with `temperature=0.0`.
  * Full prompt: Headline, full article body, full hierarchical taxonomy, full 40-item whitelist, and top-5 few-shot examples.
* **Careful path:**
  * Uses `n=3` with `temperature=0.7`.
  * Full prompt + Explicit Chain-of-Thought (CoT) instructions.
  * Three generations are parsed. We perform a majority vote on `(treatment_class, mechanism, expected_direction)`.
  * Tie-breaker: If no majority exists, the extraction with the highest `confidence` score wins.

### 2.2 Prompt Engineering (`prompts.py`)

We will construct modular formatters:
* `format_compact_taxonomy()`: returns a comma-separated string `L1.l2` for the fast path.
* `format_full_taxonomy()`: returns a hierarchical markdown list.
* `format_whitelist()`: returns grouped whitelist items.
* `format_few_shot()`: handles dynamic insertion of `limit` examples.
* `build_system_prompt()`: orchestrates the path-based templates.

## 3. Testing Strategy (`tests/agents/test_causality_extractor.py`)

* **Mocking:** We will mock `get_spark_session`, `load_taxonomy`, `load_whitelist`, and `mlflow.deployments.get_deploy_client`.
* **Fast/Standard Tests:** Verify that only 1 choice is parsed and the proper limit of examples/prompts is applied.
* **Careful Tests:** Verify that `n=3` is passed to the client. Verify majority voting correctly resolves a 2-1 vote. Verify tie-breaking logic correctly falls back to the highest confidence score.

## 4. Output Contract

The node modifies `MASState` by setting the `extraction` key to the parsed JSON dict of the chosen generation. It also catches invalid JSON and returns `None` if the LLM completely fails to generate valid JSON across all `n` attempts.
