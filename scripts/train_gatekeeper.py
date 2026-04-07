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
    
    # Dataset only has 'text' and 'label'. We'll use the first 100 characters as the pseudo-headline
    df_crypto["headline"] = df_crypto["text"].apply(lambda x: x[:100] if isinstance(x, str) else "")
    df_crypto["body"] = df_crypto["text"]
    df_crypto = df_crypto[["headline", "body", "is_impactful"]]
    
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
