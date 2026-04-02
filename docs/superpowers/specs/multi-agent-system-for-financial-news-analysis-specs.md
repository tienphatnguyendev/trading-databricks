# Multi-Agent System for Financial News Causal Analysis

## 1. Business Use Case
The objective of this Multi-Agent System (MAS) is to systematically extract structured causal chains from raw financial news to predict price movements for BTC/USDT and ETH/USDT. Rather than relying on simple sentiment analysis, the system models the market using a Structural Causal Model (SCM). 

The MAS acts as the feature engineering engine for the treatment ($T$) and confounders ($X$), separating high-impact market events from linguistic noise. By leveraging deep analysis rather than shallow zero-shot prompts, the system filters out irrelevant news, identifies the discrete causal event, validates it against a crypto-economic ontology, and passes a mathematically rigorous signal to a downstream causal ML model.

## 2. Inputs & Outputs

### Inputs
*   **News Data:** `trading-signal-gen-[env].silver.news_cleaned` (Cleaned NDJSON stream of financial news).
*   **Market Confounders ($X$):** `trading-signal-gen-[env].gold.dollar_bars` (Recent market volatility and state at the time the news is published, injected upstream into the MAS context).

### Outputs
*   **Target Table:** `trading-signal-gen-[env].gold.news_causal_features` (Materialized View or Streaming Table via Spark Declarative Pipelines).
*   **Schema Contract:**
    *   `story_id` (String): Primary key linking to the silver news table.
    *   `published_at` (Timestamp): Time of the news.
    *   `treatment_class` ($T$) (String): Discrete categorical event class (e.g., "Hack", "Regulation", "Macro", "Adoption", "None").
    *   `market_context` ($X$) (Struct): Snapshot of the market state (volatility, trend) retrieved from `dollar_bars`.
    *   `causal_chain` (String/JSON): The validated "Cause -> Effect" reasoning.
    *   `is_impactful` (Boolean): Flag indicating if the news passed the deep analysis gatekeeper.

## 3. Agent Architecture
The MAS consists of a coordinated workflow of specialized agents, executed within Databricks (e.g., via Model Serving endpoints queried via Spark `mapInPandas` or `ai_query`).

1.  **Context Retriever (Tool/Function):**
    *   Before analysis begins, fetches the latest market state ($X$) from `gold.dollar_bars` corresponding to the news timestamp.
2.  **Deep Impact Assessor Agent (The Gatekeeper):**
    *   *Role:* Replaces naive zero-shot filtering with deep reasoning (e.g., Chain-of-Thought or a multi-step sub-graph).
    *   *Function:* Evaluates the news combined with current market volatility ($X$) to determine if the event has genuine, material impact on BTC or ETH. If no impact is found, it safely bypasses the rest of the pipeline to save compute and reduce noise.
3.  **Causality Extractor Agent:**
    *   *Role:* Information extraction.
    *   *Function:* Parses impactful news into a structured causal chain. Crucially, it maps the core event to a **discrete categorical class** ($T$).
4.  **Ontology-Backed Validator Agent:**
    *   *Role:* Quality assurance and logical constraint enforcement.
    *   *Function:* Cross-references the extracted causal chain against a predefined Knowledge Graph / whitelist of valid crypto-economic causal relationships (e.g., "Interest Rates -> Fiat Strength -> BTC Price"). 
    *   *Action:* Rejects or flags chains that violate established financial ontology (e.g., effects preceding causes, or impossible market mechanics).

## 4. Downstream ML Handshake
The output of the MAS feeds directly into an advanced causal inference machine learning pipeline.

*   **Structural Causal Model (SCM) Mapping:**
    *   **$T$ (Treatment):** The discrete categorical class extracted by the Causality Extractor.
    *   **$X$ (Confounders):** The market state/volatility injected into the MAS and passed downstream.
    *   **$Y$ (Outcome):** The subsequent BTC/USDT or ETH/USDT price return.
*   **Causally Sufficient Embeddings:**
    *   The downstream ML pipeline will utilize a fine-tuned embedding architecture (e.g., CausalBERT).
    *   The loss function will be designed to predict *both* the discrete event occurrence ($T$) and the market return ($Y$).
    *   *Rationale:* Forcing the embeddings to predict both $T$ and $Y$ given $X$ ensures the model isolates the true causal semantic features of the text, stripping away spurious linguistic correlations and noise.
