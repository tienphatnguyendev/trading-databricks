# Gatekeeper Stage 2 Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Stage 2 scikit-learn Logistic Regression classifier for the news gatekeeper, train it on synthetic data, and integrate it into the PySpark pandas UDF.

**Architecture:** A standalone training script uses Hugging Face datasets to train a pipeline combining TF-IDF and spaCy NER features. The trained model is registered to MLflow. The existing `gatekeeper_udf` is updated to load this model internally (to prevent Spark serialization errors) and conditionally execute it based on the configuration.

**Tech Stack:** Python 3.10+, PySpark (pandas_udf), scikit-learn, MLflow, Hugging Face `datasets`, spaCy.

---

### Task 1: Create Dummy/Test Model Generation

Before touching the main application code, we need a reliable way to test the MLflow integration without waiting for a full training run or relying on a remote tracking server during unit tests.

**Files:**
- Create: `tests/agents/conftest.py` (if it doesn't exist, or modify)

- [ ] **Step 1: Write a fixture to generate a dummy MLflow model**

Add this to `tests/agents/conftest.py`:

```python
import pytest
import pandas as pd
import numpy as np
import mlflow
import tempfile
from sklearn.base import BaseEstimator, ClassifierMixin

class DummyGatekeeperModel(BaseEstimator, ClassifierMixin):
    """A dummy sklearn model that always predicts class 1 (impactful) for testing."""
    def fit(self, X, y):
        self.classes_ = np.array([False, True])
        return self
        
    def predict_proba(self, X):
        # Return probability 1.0 for the positive class (index 1)
        # Handle single strings or iterables
        n_samples = len(X) if isinstance(X, (list, pd.Series, np.ndarray)) else 1
        return np.array([[0.0, 1.0]] * n_samples)
        
    def predict(self, X):
        n_samples = len(X) if isinstance(X, (list, pd.Series, np.ndarray)) else 1
        return np.array([True] * n_samples)

@pytest.fixture(scope="session")
def dummy_model_uri():
    """Creates a dummy MLflow model and returns its URI for testing."""
    model = DummyGatekeeperModel()
    model.fit(["dummy text"], [True])
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # We use file:// URI for local MLflow tracking during tests
        mlflow.set_tracking_uri(f"file://{tmp_dir}/mlruns")
        with mlflow.start_run():
            model_info = mlflow.sklearn.log_model(model, "dummy-model")
            yield model_info.model_uri
```

- [ ] **Step 2: Commit**

```bash
git add tests/agents/conftest.py
git commit -m "test: add dummy MLflow model fixture for gatekeeper tests"
```

---

### Task 2: Implement Gatekeeper UDF Stage 2 Logic

Update the existing `gatekeeper_udf` to load the MLflow model (inside the UDF to avoid serialization issues) and use it when `ner_only_mode` is False.

**Files:**
- Modify: `src/trading_databricks/agents/gatekeeper.py`
- Modify: `tests/agents/test_gatekeeper.py`

- [ ] **Step 1: Update the UDF implementation**

Modify `src/trading_databricks/agents/gatekeeper.py`. Ensure `mlflow` and `pandas` imports are present.

```python
# Add imports at the top
import pandas as pd
import mlflow
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, BooleanType, DoubleType
from trading_databricks.agents.config import GatekeeperConfig

# Define schema if not already imported/defined
GATEKEEPER_RETURN_SCHEMA = StructType([
    StructField("is_impactful", BooleanType(), nullable=False),
    StructField("gatekeeper_confidence", DoubleType(), nullable=True),
])

def apply_gatekeeper(df, config: GatekeeperConfig):
    """
    Applies the gatekeeper UDF to the DataFrame.
    """
    
    @pandas_udf(GATEKEEPER_RETURN_SCHEMA)
    def gatekeeper_udf(headlines: pd.Series, bodies: pd.Series) -> pd.DataFrame:
        """
        Pandas UDF for Gatekeeper classification.
        Loads spaCy and MLflow models INSIDE the UDF to prevent Spark serialization errors.
        """
        # 1. Load resources inside the UDF body
        nlp = build_nlp()
        model = None
        if not config.ner_only_mode:
            # Safe to load here; executed on workers
            model = mlflow.sklearn.load_model(config.model_uri)
            
        # 2. Process the batch
        results = []
        for headline, body in zip(headlines, bodies):
            # Stage 1: NER check
            is_impactful_ner, confidence_ner = is_crypto_relevant(headline, body, config=config)
            
            if config.ner_only_mode:
                # V1 behavior: Return Stage 1 results directly
                results.append((is_impactful_ner, confidence_ner))
            else:
                # Stage 2 behavior: Sklearn classifier
                # We concatenate text identically to training
                text = headline + " " + body[:config.text_window]
                
                # Predict probabilities. predict_proba returns shape (n_samples, n_classes)
                # We assume class 1 is the positive class (is_impactful=True)
                probs = model.predict_proba([text])[0]
                confidence = float(probs[1]) 
                
                is_impactful = confidence >= config.confidence_threshold
                results.append((is_impactful, confidence))
                
        return pd.DataFrame(results, columns=["is_impactful", "gatekeeper_confidence"])

    return df.withColumn(
        "gatekeeper_result", 
        gatekeeper_udf(df["headline"], df["body"])
    ).select(
        "*", 
        "gatekeeper_result.is_impactful", 
        "gatekeeper_result.gatekeeper_confidence"
    ).drop("gatekeeper_result")

```

- [ ] **Step 2: Write tests for MLflow and NER modes**

Add to `tests/agents/test_gatekeeper.py`:

```python
import pandas as pd
from trading_databricks.agents.config import GatekeeperConfig
from trading_databricks.agents.gatekeeper import apply_gatekeeper

def test_apply_gatekeeper_ner_only_mode(spark):
    """Test V1 behavior where only NER is used."""
    # Given
    data = [
        ("Bitcoin surges today", "The price of BTC went up."), # Should pass NER
        ("Apple earnings report", "Tim Cook announced record profits.") # Should fail NER
    ]
    df = spark.createDataFrame(data, ["headline", "body"])
    config = GatekeeperConfig(ner_only_mode=True)
    
    # When
    result_df = apply_gatekeeper(df, config).toPandas()
    
    # Then
    assert len(result_df) == 2
    # First row has crypto entities
    assert result_df.iloc[0]["is_impactful"] is True
    assert result_df.iloc[0]["gatekeeper_confidence"] == 1.0
    # Second row has no crypto entities
    assert result_df.iloc[1]["is_impactful"] is False
    assert result_df.iloc[1]["gatekeeper_confidence"] == 0.0


def test_apply_gatekeeper_mlflow_mode(spark, dummy_model_uri):
    """Test Stage 2 behavior where MLflow model is loaded and used."""
    # Given
    data = [
        ("Random text", "More text."), 
    ]
    df = spark.createDataFrame(data, ["headline", "body"])
    
    # Configure to use MLflow and point to our dummy model fixture
    config = GatekeeperConfig(
        ner_only_mode=False, 
        model_uri=dummy_model_uri,
        confidence_threshold=0.5
    )
    
    # When
    result_df = apply_gatekeeper(df, config).toPandas()
    
    # Then
    assert len(result_df) == 1
    # The dummy model always predicts True with 1.0 confidence
    assert result_df.iloc[0]["is_impactful"] is True
    assert result_df.iloc[0]["gatekeeper_confidence"] == 1.0

```

- [ ] **Step 3: Run tests to verify**

Run: `uv run pytest tests/agents/test_gatekeeper.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/trading_databricks/agents/gatekeeper.py tests/agents/test_gatekeeper.py
git commit -m "feat: implement gatekeeper stage 2 pandas UDF with mlflow loading"
```

---

### Task 3: Create Training Script

Build the standalone training script that utilizes Hugging Face datasets and logs the `Pipeline` to MLflow.

**Files:**
- Create: `scripts/train_gatekeeper.py`
- Modify: `pyproject.toml` (to add `datasets` and `scikit-learn` if missing)

- [ ] **Step 1: Update dependencies**

Run: `uv add scikit-learn datasets`

- [ ] **Step 2: Create training script**

Create `scripts/train_gatekeeper.py`:

```python
#!/usr/bin/env python3
"""
Train Gatekeeper Stage 2 Model (TRD-26).
Uses Hugging Face datasets to train a scikit-learn pipeline (TF-IDF + NER features).
Registers the model to MLflow Staging.
"""

import argparse
import mlflow
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from trading_databricks.agents.gatekeeper import build_nlp, extract_ner_features

def _extract_ner_wrapper(texts: list[str]) -> np.ndarray:
    """Wrapper to apply NER extraction to a list of strings for scikit-learn."""
    nlp = build_nlp()
    features_list = []
    for doc in nlp.pipe(texts):
        feats = extract_ner_features(doc)
        # Convert boolean/int dict to flat array: [asset, exch, stable, prot, concept, count]
        row = [
            float(feats["has_crypto_asset"]),
            float(feats["has_exchange"]),
            float(feats["has_stablecoin"]),
            float(feats["has_protocol"]),
            float(feats["has_concept"]),
            float(feats["entity_count"]),
        ]
        features_list.append(row)
    return np.array(features_list)

def load_training_data() -> pd.DataFrame:
    """
    Loads and blends data from Hugging Face for Gatekeeper training.
    Positive: SahandNZ/cryptonews with bullish/bearish momentum.
    Negative: zeroshot/twitter-financial-news-topic + neutral cryptonews.
    """
    print("Loading datasets from Hugging Face...")
    
    # 1. Crypto News (Positive & Irrelevant Crypto Negatives)
    # Note: Using a small split for speed; adjust in production
    ds_crypto = load_dataset("SahandNZ/cryptonews-articles-with-price-momentum-labels", split="train[:5000]")
    df_crypto = ds_crypto.to_pandas()
    
    # Map momentum to impactful (bullish/bearish -> True, neutral -> False)
    # Assuming labels exist like 'bullish', 'bearish', 'neutral' or numerical equivalents.
    # We will simulate a simplistic mapping here based on common structures.
    # IMPORTANT: Adjust this mapping based on the actual dataset schema.
    df_crypto["is_impactful"] = df_crypto["label"].apply(lambda x: x != 0) # Assuming 0 is neutral
    df_crypto = df_crypto[["title", "text", "is_impactful"]].rename(columns={"title": "headline", "text": "body"})
    
    # 2. General Finance Noise (Hard Negatives)
    ds_fin = load_dataset("zeroshot/twitter-financial-news-topic", split="train[:2000]")
    df_fin = ds_fin.to_pandas()
    df_fin["headline"] = df_fin["text"]
    df_fin["body"] = ""
    df_fin["is_impactful"] = False
    df_fin = df_fin[["headline", "body", "is_impactful"]]
    
    # Blend and shuffle
    df_combined = pd.concat([df_crypto, df_fin], ignore_index=True)
    df_combined = df_combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Concatenate text as done in inference
    df_combined["text"] = df_combined["headline"] + " " + df_combined["body"].str.slice(0, 500)
    
    return df_combined

def train_and_register(register_model: bool = False):
    df = load_training_data()
    X = df["text"].tolist()
    y = df["is_impactful"].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training on {len(X_train)} samples. Testing on {len(X_test)} samples.")
    print(f"Class distribution (Train): {pd.Series(y_train).value_counts().to_dict()}")
    
    # Define Pipeline
    ner_transformer = FunctionTransformer(_extract_ner_wrapper, validate=False)
    tfidf_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1,2))
    
    # We apply both transformers to the SAME input (the concatenated text)
    # Unfortunately ColumnTransformer applies transformations to specific columns.
    # Since X is a 1D list of strings, we use FeatureUnion (via make_union) or a custom approach.
    # FeatureUnion is better suited for applying multiple transformers to the same input array.
    from sklearn.pipeline import FeatureUnion
    
    combined_features = FeatureUnion([
        ("tfidf", tfidf_vectorizer),
        ("ner", ner_transformer)
    ])
    
    pipeline = Pipeline([
        ("features", combined_features),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
    ])
    
    # Train
    print("Training pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(classification_report(y_test, y_pred))
    print(f"Precision: {precision:.4f} (Gate: >0.80)")
    print(f"Recall: {recall:.4f} (Gate: >0.95)")
    
    # Quality Gates (Warn if failed, but allow execution for testing)
    if precision <= 0.80:
        print("WARNING: Precision quality gate failed (< 0.80).")
    if recall <= 0.95:
        print("WARNING: Recall quality gate failed (< 0.95).")
        
    # MLflow Registration
    mlflow.set_experiment("/Shared/news-gatekeeper-training")
    with mlflow.start_run():
        mlflow.log_param("dataset_size", len(X))
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        print("Logging model to MLflow...")
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline, 
            artifact_path="model",
            registered_model_name="news-gatekeeper" if register_model else None
        )
        
        if register_model:
            print(f"Model registered as 'news-gatekeeper'. URI: {model_info.model_uri}")
            print("Action Required: Please manually promote the model to 'Staging' or 'Production' in the MLflow UI.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true", help="Register model to MLflow Model Registry")
    args = parser.parse_args()
    
    train_and_register(register_model=args.register)

```

- [ ] **Step 3: Run the script locally (Dry run without registering)**

Run: `uv run python scripts/train_gatekeeper.py`
Expected: Script downloads data (may take a minute), trains, prints classification report, and logs to local MLflow.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_gatekeeper.py pyproject.toml uv.lock
git commit -m "feat: add training script for gatekeeper stage 2 classifier"
```
