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
    dbutils.widgets.text("limit", "10", "Number of articles to sample")  # type: ignore
    dbutils.widgets.text("l1_category", "All", "Filter by L1 Category")  # type: ignore
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
query = "SELECT * FROM `smart-trading-dev`.others.mas_training_dataset LIMIT :limit"
df = spark.sql(query, args={"limit": limit})
pdf = df.toPandas()

# Convert to list of MASState dicts for easier manual feeding to nodes
states = []
for _, row in pdf.iterrows():
    states.append(
        {
            "headline": row.get("headline", ""),
            "body": row.get("body", ""),
            "published_at": str(row.get("published_at", "")),
            "story_id": row.get("story_id", ""),
        }
    )
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
    text = state["headline"] + " " + state["body"][: gk_config.text_window]
    doc = nlp(text)
    features = extract_ner_features(doc)
    extracted_entities = [(ent.text, ent.label_) for ent in doc.ents]

    gatekeeper_results.append(
        {
            "story_id": state.get("story_id"),
            "headline": state["headline"],
            "is_impactful": is_impactful,
            "extracted_entities": extracted_entities,
            **features,
        }
    )

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

        routing_results.append(
            {
                "headline": state["headline"],
                "top_similarity": top_sim,
                "extraction_path": routed_state.get("extraction_path"),
            }
        )

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

# COMMAND ----------
# MAGIC %md
# MAGIC ## Section 4: LLM Causality Extractor (Autopsy)
# MAGIC Inspect exact prompts and raw LLM responses for the careful path.

# COMMAND ----------
from trading_databricks.agents.nodes.causality_extractor import (  # noqa: E402
    causality_extractor,
)
from trading_databricks.agents.prompts import build_system_prompt  # noqa: E402
from trading_databricks.agents.taxonomy import load_taxonomy, load_whitelist  # noqa: E402

try:
    # Sandbox: Manually set a state to 'careful' path to observe majority voting
    sandbox_state = states[0].copy()
    sandbox_state["extraction_path"] = "careful"
    sandbox_state["few_shot_examples"] = []  # Empty for cold-start test

    print(f"Testing extraction for headline: {sandbox_state['headline']}")

    # Show what the LLM sees
    taxonomy = load_taxonomy(
        spark, catalog=MASConfig().catalog, schema=MASConfig().schema
    )
    whitelist = load_whitelist(
        spark, catalog=MASConfig().catalog, schema=MASConfig().schema
    )
    system_prompt = build_system_prompt(taxonomy, whitelist, [], "careful")
    print(
        f"\n--- SYSTEM PROMPT (Truncated) ---\n{system_prompt[:500]}...\n"
        "---------------------------------"
    )

    # Run the extractor (calls databricks-claude-sonnet)
    extracted_state = causality_extractor(sandbox_state, config=MASConfig())
    print("\nFinal Majority Vote Extraction:")
    print(pd.Series(extracted_state.get("extraction", {})))

except Exception as e:
    print(f"Skipping Extractor execution (requires active MLflow deployments): {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Section 5: Mechanism Validator (Constraints Debugging)
# MAGIC Debug evidence span substring matching and semantic relevance scoring (Module B).

# COMMAND ----------
from trading_databricks.agents.nodes.mechanism_validator import (  # noqa: E402
    _validate_evidence_spans,
    validate_extraction,
)


# Create a mock embedding function for sandbox testing
def mock_embed_fn(text: str) -> list[float]:
    import numpy as np

    return list(np.random.rand(1024))


try:
    # Sandbox: Create a problematic extraction missing exact evidence
    mock_extraction = {
        "treatment_class": "Macro.rate_decision",
        "mechanism": "liquidity_contraction",
        "expected_direction": "bearish",
        "trigger_event": "Fed raises rates",
        # Intentionally paraphrased:
        "evidence_spans": ["The Fed raised rates by 50 basis points"],
    }

    body = "The Federal Reserve increased interest rates by 50 basis points today."

    print(f"Evidence Spans: {mock_extraction['evidence_spans']}")
    print(f"Article Body: {body}")

    # Test strict substring matching
    passed = _validate_evidence_spans(mock_extraction["evidence_spans"], body)  # type: ignore
    print(f"\nStrict Substring Match Passed: {passed}")
    if not passed:
        print(
            "Data Scientist Action: The LLM paraphrased the evidence. "
            "Need to adjust the system prompt to enforce 'VERBATIM' copying."
        )

    # Run full validator
    validation_result = validate_extraction(
        extraction=mock_extraction,  # type: ignore
        article_body=body,
        whitelist=whitelist,  # from section 4
        config=MASConfig().validator,
        embed_fn=mock_embed_fn,
    )

    print("\nValidation Result:")
    print(pd.Series(validation_result))

except Exception as e:
    print(f"Skipping Validator execution: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Section 6: End-to-End Evaluation via MLflow
# MAGIC Execute the full compiled LangGraph on a test batch and log traces to MLflow.

# COMMAND ----------
import time  # noqa: E402

from trading_databricks.agents.graph import build_mas_graph  # noqa: E402

try:
    # Compile graph
    graph = build_mas_graph(config=MASConfig())

    # Run a small E2E batch (e.g., 2 articles) to avoid massive LLM costs in the sandbox
    batch_states = states[:2]

    e2e_results = []

    with mlflow.start_run(run_name="Interactive_QA_Batch"):
        for state in batch_states:
            print(f"Processing story_id: {state.get('story_id')}...")
            start_time = time.time()

            # Invoke Graph
            final_state = graph.invoke(state)

            latency = time.time() - start_time

            e2e_results.append({
                "story_id": final_state.get("story_id"),
                "headline": final_state.get("headline"),
                "extraction_path": final_state.get("extraction_path"),
                "validation_status": final_state.get("validation_status"),
                "latency_sec": round(latency, 2)
            })

    e2e_df = pd.DataFrame(e2e_results)
    print("\nE2E Evaluation Summary:")
    display(e2e_df) # type: ignore  # noqa: F821

    print("\nMetrics:")
    print(e2e_df["validation_status"].value_counts())
    print(
        "\nCheck the MLflow Experiment UI to view the "
        "full execution traces for these runs."
    )

except Exception as e:
    print(f"Skipping E2E execution (requires active dependencies): {e}")
