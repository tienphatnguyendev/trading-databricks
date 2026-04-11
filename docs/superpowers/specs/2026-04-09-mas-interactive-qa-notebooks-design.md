# MAS Interactive QA & Debugging Notebooks Design

## Objective
Build a comprehensive, interactive Databricks notebook suite for data scientists to debug, evaluate, and tune the Multi-Agent System (MAS) causality pipeline. This tool will allow users to transition seamlessly from macro-level pipeline evaluation (using MLflow Tracing and aggregate metrics) to micro-level component inspection (step-by-step state mutations).

## Core Approach: "Time-Travel" Trace & Sandbox
The QA notebooks operate as a hybrid tool. They leverage MLflow Tracing to record state mutations, LLM prompts, and tool calls during End-to-End (E2E) execution, while providing isolated "sandbox" sections to manually feed specific data (`MASState`) into individual pipeline nodes.

## Data Sources & Dependencies
- **Dataset:** `smart-trading-dev.others.mas_training_dataset`
- **Models:** `databricks-claude-sonnet` (Extractor), `databricks-gte-large-en` (Validator Embedding)
- **Vector Search:** `smart-trading-dev.03_gold.causal_chains_idx`
- **Libraries:** `pyspark.sql`, `mlflow`, `langgraph`, `spacy`

## Notebook Architecture & Modules

### Section 1: Setup & Data Ingestion
- **Goal:** Initialize connections to Databricks Model Serving, Vector Search, MLflow, and Unity Catalog.
- **Implementation:** Query `smart-trading-dev.others.mas_training_dataset`. Use `dbutils.widgets` to filter the dataset interactively (e.g., by L1 category or date). Convert the Spark DataFrame to a Pandas DataFrame or a list of `MASState` dictionaries for local memory iteration.

### Section 2: Gatekeeper (Stage 1) Deep Dive
- **Goal:** Tune and evaluate the rule-based spaCy EntityRuler (`ner_only_mode=True`).
- **Implementation:** Run the ingested subset through `is_crypto_relevant()`.
- **UX/Output:** Display a DataFrame focusing on False Negatives (crypto-relevant articles dropped by the Gatekeeper) and highlight exactly which NER tokens were extracted versus missed.

### Section 3: RAG & Routing Inspection
- **Goal:** Tune the Vector Search cosine similarity thresholds for the Uncertainty Router (`fast` >= 0.85, `careful` < 0.50).
- **Implementation:** Execute `context_retriever` and `few_shot_retriever`.
- **UX/Output:** For a selected headline, display the top 5 retrieved examples. Generate a histogram of similarity scores across the dataset to visually assist in tuning the routing thresholds.

### Section 4: LLM Causality Extractor (Autopsy)
- **Goal:** Inspect LLM prompts, raw outputs, and the `majority_vote` resolution logic on the `careful` path.
- **Implementation:** Run the `causality_extractor` node on specific problematic states.
- **UX/Output:** Render the exact system/user prompts sent to `databricks-claude-sonnet`. Display raw text outputs of all 3 LLM choices, parsed JSONs side-by-side, and the final resolved consensus.

### Section 5: Mechanism Validator (Constraints Debugging)
- **Goal:** Debug strict deterministic checks, specifically evidence span substring matching and semantic relevance scoring (Module B).
- **Implementation:** Run the `mechanism_validator` node.
- **UX/Output:** Create a visual diff output for `rejected_evidence`. Highlight the exact characters in the article body that closely match the LLM's generated span to help data scientists adjust prompts (e.g., "DO NOT PARAPHRASE") or validation logic. Plot the `evidence_relevance_score` distribution.

### Section 6: End-to-End Evaluation via MLflow
- **Goal:** Execute the full compiled LangGraph on a test batch and log traces.
- **Implementation:** Wrap `build_mas_graph(config).invoke(state)` in an MLflow run.
- **UX/Output:** Aggregate summary metrics:
  - % Accepted vs. Rejected (broken down by rejection reason: mechanism, direction, evidence)
  - % Fast vs. Standard vs. Careful paths taken
  - Average latency per path
  - Links directly to the MLflow Experiment UI for trace inspection.

## Spec Self-Review
- **Placeholder Check:** No TBDs remaining. Specific models, tables, and libraries are explicitly named.
- **Internal Consistency:** The 6 sections cover all identified nodes from the codebase (`gatekeeper`, `context_retriever`, `few_shot_retriever`, `uncertainty_router`, `causality_extractor`, `mechanism_validator`, `graph`).
- **Scope Check:** Appropriately scoped as an interactive QA tool for data scientists, avoiding production deployment complexities.
- **Ambiguity Check:** The difference between macro (Section 6) and micro (Sections 2-5) evaluation is clearly defined.

## Git Tracking
This spec will be committed to `docs/superpowers/specs/2026-04-09-mas-interactive-qa-notebooks-design.md`.