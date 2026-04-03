# Research Notebooks Orchestration Plan

**Date:** 2026-04-03
**Status:** In Progress
**Orchestrator:** Gemini CLI

## 1. Objective
Establish two foundational research and development notebooks:
1. **Agent Domain (`notebooks/research_mas_causal_news.py`):** Prototype the Multi-Agent System (MAS) for causal financial news analysis.
2. **ML Domain (`notebooks/research_ml_signal_gen.py`):** Prototype the ML signal generation pipeline using Structural Causal Models (SCM) and tick-based bars.

These notebooks will serve as the experimental ground for the currently stubbed `src/trading_databricks/agents/` and `src/trading_databricks/ml/` directories, as well as the unwritten `gold` layer transformations.

## 2. Architecture & Data Flow

As defined in `multi-agent-system-for-financial-news-analysis-specs.md`:
*   **Inputs:** `bronze.news_agent_eval_dataset` (proxied for silver news) and `bronze.raw_aggtrades_samples` (proxied for gold dollar bars during research).
*   **Agent Notebook (MAS):** 
    *   Implements the 4-stage workflow: Context Retriever -> Deep Impact Assessor -> Causality Extractor -> Ontology-Backed Validator.
    *   *Output:* A DataFrame matching the `gold.news_causal_features` schema contract.
*   **ML Notebook (Signal Gen):**
    *   Implements prototype bar construction (e.g., Dollar Bars) from the raw aggTrades.
    *   Joins the MAS output ($T$) with the market state ($X$) to predict forward returns ($Y$).
    *   *Output:* Prototype causal ML features and training loops.

## 3. Parallel Agent Dispatch Strategy

We will deploy two parallel subagents (or parallel execution streams) to build these notebooks independently without colliding.

### Stream A: Multi-Agent System (MAS) Research
**Focus:** NLP, LLM Routing, Databricks Model Serving / AI Functions.
**Tasks:**
1.  Initialize `notebooks/research_mas_causal_news.py`.
2.  Set up Databricks `ai_query` or LangChain/MLflow wrappers to call the LLM.
3.  Implement the **Gatekeeper Agent** (Deep Impact Assessor) prompt and routing logic.
4.  Implement the **Causality Extractor** and **Ontology Validator** prompts.
5.  Run the 500-row `news_agent_eval_dataset` through the pipeline to generate a mock `news_causal_features` dataset.

### Stream B: ML Signal Generation Research
**Focus:** Quantitative Finance, Time-Series, Causal Inference.
**Tasks:**
1.  Initialize `notebooks/research_ml_signal_gen.py`.
2.  Load the `raw_aggtrades_samples` (52M rows).
3.  Prototype the `build_dollar_bars` logic (from `ticks/gold/dollar_bars.py` stubs) to create the $X$ (confounders) dataset.
4.  Prototype the AS-OF join logic to merge tick bars with the mocked `news_causal_features` ($T$).
5.  Define the target label $Y$ (e.g., 5-minute forward return) and set up the baseline MLflow tracking experiment.

## 4. Orchestration Checklist
- [ ] Answer clarification questions.
- [ ] Create Linear tickets for both streams (TRD-XXX).
- [ ] Provision isolated Git worktrees for Stream A and Stream B.
- [ ] Execute Stream A (MAS Prototype).
- [ ] Execute Stream B (ML Prototype).
- [ ] Review and merge findings.