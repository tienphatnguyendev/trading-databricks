# Gatekeeper Stage 2 — sklearn LogReg Classifier (TRD-26)

**Date:** 2026-04-06
**Ticket:** TRD-26
**Parent:** TRD-6

## 1. Objective

Implement the second stage of the Gatekeeper agent. This stage uses a scikit-learn `LogisticRegression` classifier to determine if a news article is `is_impactful`. It leverages a `ColumnTransformer` that combines TF-IDF on the text (headline + body) with the dense 6-feature output from the Stage 1 NER filter.

The model artifact is managed by MLflow and executed within a safe `pandas_udf` to avoid Spark serialization errors.

## 2. Architecture & Components

### 2.1 Training Script (`scripts/train_gatekeeper.py`)

A standalone script to generate the training dataset, train the pipeline, evaluate against quality gates, and register the model to MLflow.

*   **Data Sourcing (Approach A - Hugging Face Blend):**
    *   **Positive Class:** Bullish/Bearish articles from `SahandNZ/cryptonews-articles-with-price-momentum-labels`.
    *   **Negative Class:** Neutral articles from `SahandNZ/cryptonews` + General finance tweets from `zeroshot/twitter-financial-news-topic`.
*   **Pipeline Definition (`sklearn.pipeline.Pipeline`):**
    *   **Text Feature Extraction:** `TfidfVectorizer(max_features=500, ngram_range=(1,2))` applied to `headline + " " + body[:500]`.
    *   **NER Feature Extraction:** A custom `FunctionTransformer` that applies `extract_ner_features(build_nlp()(text))` to generate 6 dense numerical features.
    *   **Combining:** `ColumnTransformer` to merge the sparse TF-IDF matrix and the dense NER features.
    *   **Classifier:** `LogisticRegression(class_weight="balanced")` to handle the heavy class imbalance (assuming ~95/5 negative/positive).
*   **Evaluation & Quality Gates:**
    *   Hold out 20% of the dataset for testing.
    *   Assert **Precision > 0.80** (to avoid wasting LLM compute on false positives).
    *   Assert **Recall > 0.95** (to ensure we don't miss truly impactful articles).
*   **MLflow Registration:**
    *   Log the trained model using `mlflow.sklearn.log_model(model, "news-gatekeeper")`.
    *   Register the model to the MLflow Model Registry as `models:/news-gatekeeper`.
    *   **Crucial:** Register to the `Staging` environment (manual promotion to `Production` required).

### 2.2 Gatekeeper UDF Integration (`src/trading_databricks/agents/gatekeeper.py`)

Integrate the trained model into the PySpark pipeline.

*   **Safe Serialization:** The spaCy pipeline (`build_nlp()`) and the MLflow model (`mlflow.sklearn.load_model()`) must be instantiated *inside* the body of the `pandas_udf` to ensure they are created on the Spark executors, avoiding driver-to-executor serialization errors.
*   **Configuration (`GatekeeperConfig`):**
    *   If `ner_only_mode=True` (V1 default): Bypass MLflow. Only execute the Stage 1 NER filter (`is_crypto_relevant`). Return `is_impactful` based on NER presence, with `gatekeeper_confidence` = 1.0 (if present) or 0.0 (if absent).
    *   If `ner_only_mode=False`: Load the MLflow model from `config.model_uri` (e.g., `models:/news-gatekeeper/Production`). Execute `model.predict_proba()` on the concatenated text to generate `gatekeeper_confidence`. Threshold this confidence (e.g., `> config.confidence_threshold`) to determine `is_impactful`.
*   **Return Schema:** A DataFrame matching `GATEKEEPER_RETURN_SCHEMA`: `is_impactful` (bool) and `gatekeeper_confidence` (float).

### 2.3 Testing (`tests/agents/test_gatekeeper.py`)

*   **`test_classifier`:** Train a small, dummy `Pipeline` on synthetic data within the test setup. Ensure the precision/recall evaluation logic functions correctly.
*   **`test_gatekeeper_udf_ner_mode`:** Verify the UDF correctly applies Stage 1 rules when `ner_only_mode=True` and returns the expected schema without attempting to load an MLflow model.
*   **`test_gatekeeper_udf_mlflow_mode`:** (Mocked MLflow load) Verify the UDF correctly calls the sklearn model's `predict_proba` when `ner_only_mode=False`.

## 3. Data Flow & Dependencies

1.  `scripts/train_gatekeeper.py` (Local execution) -> Downloads HF datasets -> Trains -> MLflow `Staging`.
2.  (Manual step) Promote model to `Production` in Databricks MLflow UI.
3.  `gatekeeper_udf` (Spark execution) -> Reads `GatekeeperConfig` -> Loads model from `models:/news-gatekeeper/Production` -> Predicts on `silver.news_cleaned`.

## 4. Acceptance Criteria

*   The training script runs end-to-end, evaluates against the precision/recall gates, and registers the model in MLflow Staging.
*   The `gatekeeper_udf` processes a Spark DataFrame without serialization errors.
*   The UDF correctly toggles behavior based on `GatekeeperConfig.ner_only_mode`.
*   All tests in `tests/agents/test_gatekeeper.py` pass.