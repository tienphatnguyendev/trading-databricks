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

# COMMAND ----------
# MAGIC %md
# MAGIC ## Section 2: Gatekeeper (Stage 1) Deep Dive
# MAGIC Inspect articles dropped by the spaCy EntityRuler (False Negatives).

# COMMAND ----------
from trading_databricks.agents.config import GatekeeperConfig  # noqa: E402
from trading_databricks.agents.gatekeeper import (  # noqa: E402
    build_nlp,
    extract_ner_features,
    is_crypto_relevant,
)

gk_config = GatekeeperConfig()
nlp = build_nlp()

gatekeeper_results = []
for state in states:
    is_impactful, conf = is_crypto_relevant(
        state["headline"], state["body"], config=gk_config
    )

    # Deep dive into exact entities
    text = state["headline"] + " " + state["body"][:gk_config.text_window]
    doc = nlp(text)
    features = extract_ner_features(doc)
    extracted_entities = [(ent.text, ent.label_) for ent in doc.ents]

    gatekeeper_results.append({
        "story_id": state.get("story_id"),
        "headline": state["headline"],
        "is_impactful": is_impactful,
        "extracted_entities": extracted_entities,
        **features
    })

gk_df = pd.DataFrame(gatekeeper_results)
print("Gatekeeper Breakdown:")
print(gk_df["is_impactful"].value_counts())

# Show false negatives (if you know the dataset contains crypto articles)
print("\nSample articles dropped by Gatekeeper (is_impactful=False):")
display(gk_df[gk_df["is_impactful"] == False].head(5))  # type: ignore  # noqa: F821, E712

# COMMAND ----------
# MAGIC %md
# MAGIC ## Section 3: RAG & Routing Inspection
# MAGIC Evaluate Vector Search similarity scores to tune routing thresholds.

# COMMAND ----------
import matplotlib.pyplot as plt  # noqa: E402

from trading_databricks.agents.config import MASConfig  # noqa: E402
from trading_databricks.agents.nodes.few_shot_retriever import (  # noqa: E402
    few_shot_retriever,
)
from trading_databricks.agents.nodes.uncertainty_router import (  # noqa: E402
    uncertainty_router,
)

# We wrap this in try-except because it requires active VS/Spark connection
try:
    routing_results = []
    for state in states[:5]:  # Limit to 5 for fast sandbox iteration
        # Run Few Shot Retriever
        retrieved_state = few_shot_retriever(state.copy(), config=MASConfig())

        # Run Uncertainty Router
        routed_state = uncertainty_router(retrieved_state, config=MASConfig())

        examples = retrieved_state.get("few_shot_examples", [])
        top_sim = examples[0].get("similarity_score", 0.0) if examples else 0.0

        routing_results.append({
            "headline": state["headline"],
            "top_similarity": top_sim,
            "extraction_path": routed_state.get("extraction_path")
        })

    routing_df = pd.DataFrame(routing_results)
    display(routing_df)  # type: ignore  # noqa: F821

    # Plot histogram if enough data
    if len(routing_df) > 1:
        routing_df["top_similarity"].plot(
            kind="hist", bins=20, title="Top Similarity Score Distribution"
        )
        plt.axvline(
            x=MASConfig().router.high_similarity_threshold,
            color="g",
            linestyle="--",
            label="Fast Threshold",
        )
        plt.axvline(
            x=MASConfig().router.low_similarity_threshold,
            color="r",
            linestyle="--",
            label="Careful Threshold",
        )
        plt.legend()
        plt.show()
except Exception as e:
    print(f"Skipping RAG execution (requires active VS/Spark): {e}")
