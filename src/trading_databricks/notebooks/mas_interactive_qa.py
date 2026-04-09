# Databricks notebook source
# MAGIC %md
# MAGIC # MAS Interactive QA & Debugging Notebook
# MAGIC This notebook provides tools to evaluate the MAS causality pipeline both
# MAGIC macroscopically and microscopically.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Section 1: Setup & Data Ingestion
# MAGIC Initialize connections, pull sample dataset, and map to MASState.

# COMMAND ----------
import mlflow  # noqa: F401
import pandas as pd  # noqa: F401
from pyspark.sql import SparkSession

# Fallback for local testing (Databricks injects spark/dbutils)
try:
    dbutils.widgets.text("limit", "10", "Number of articles to sample") # type: ignore
    dbutils.widgets.text("l1_category", "All", "Filter by L1 Category") # type: ignore
except NameError:
    class MockWidgets:
        def text(self, *args: object, **kwargs: object) -> None:
            pass
        def get(self, key: str) -> str:
            return "10" if key == "limit" else "All"
    class MockDbutils:
        widgets = MockWidgets()
    dbutils = MockDbutils()
    spark = SparkSession.builder.getOrCreate()

limit = int(dbutils.widgets.get("limit"))
l1_category = dbutils.widgets.get("l1_category")

print(f"Loading {limit} rows from smart-trading-dev.others.mas_training_dataset...")
query = f"SELECT * FROM `smart-trading-dev`.others.mas_training_dataset LIMIT {limit}"
df = spark.sql(query)
pdf = df.toPandas()

# Convert to list of MASState dicts for easier manual feeding to nodes
states = []
for _, row in pdf.iterrows():
    states.append({
        "headline": row.get("headline", ""),
        "body": row.get("body", ""),
        "published_at": str(row.get("published_at", "")),
        "story_id": row.get("story_id", "")
    })
print(f"Loaded {len(states)} articles.")
