# Databricks notebook source
# MAGIC %md
# MAGIC # MAS Causal News Research
# MAGIC
# MAGIC This notebook evaluates a LangGraph-based agent pipeline for causal financial event extraction from news.

# COMMAND ----------
# MAGIC %pip install langgraph pandas mlflow

# COMMAND ----------
import json
import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
import pandas as pd
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from trading_databricks.common.uc_identifiers import (
    BRONZE_SCHEMA,
    DEFAULT_CATALOG,
    NEWS_AGENT_EVAL_DATASET_TABLE,
    quoted_table_fqn,
)

class NewsAgentState(TypedDict):
    article_id: str
    article_text: str
    is_relevant: Optional[bool]
    context: Optional[str]
    extracted_events: Optional[List[Dict[str, Any]]]

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load Evaluation Dataset

# COMMAND ----------
def load_eval_dataset(limit: int = 5) -> pd.DataFrame:
    """Loads the news agent eval dataset for testing."""
    df = spark.table(
        quoted_table_fqn(
            DEFAULT_CATALOG,
            BRONZE_SCHEMA,
            NEWS_AGENT_EVAL_DATASET_TABLE,
        )
    )
    return df.limit(limit).toPandas()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Define Agent Nodes

# COMMAND ----------
@mlflow.trace(name="gatekeeper_node")
def gatekeeper_node(state: NewsAgentState) -> NewsAgentState:
    """Classifies if the article is financially relevant using ai_classify."""
    print(f"Running Gatekeeper for article {state['article_id']}...")
    text = state['article_text'].replace("'", "''") # Escape single quotes
    
    # Using ai_classify via Spark SQL
    query = f"""
    SELECT ai_classify(
        '{text}',
        ARRAY('Financially Relevant', 'Not Financially Relevant')
    ) as relevance
    """
    try:
        df = spark.sql(query)
        result = df.collect()[0]['relevance']
        state['is_relevant'] = (result == 'Financially Relevant')
    except Exception as e:
        print(f"Gatekeeper failed: {e}")
        state['is_relevant'] = False
        
    return state

@mlflow.trace(name="context_retriever_node")
def context_retriever_node(state: NewsAgentState) -> NewsAgentState:
    """MVP Mock for Context Retriever."""
    print(f"Running Context Retriever for article {state['article_id']}...")
    state['context'] = "MOCKED_CONTEXT: No historical context needed for MVP."
    return state

@mlflow.trace(name="extractor_node")
def extractor_node(state: NewsAgentState) -> NewsAgentState:
    """Extracts causal events using ai_query."""
    print(f"Running Extractor for article {state['article_id']}...")
    text = state['article_text'].replace("'", "''")
    prompt = f"Extract causal financial events from this text. Return a JSON array of objects with 'cause' and 'effect' keys. Text: {text}"
    prompt = prompt.replace("'", "''")
    
    # Using ai_query with a standard model (e.g., databricks-meta-llama-3-1-70b-instruct)
    query = f"""
    SELECT ai_query(
        'databricks-meta-llama-3-1-70b-instruct',
        '{prompt}'
    ) as extracted
    """
    try:
        df = spark.sql(query)
        result_str = df.collect()[0]['extracted']
        # Try to parse JSON out of the response
        try:
            # Very basic extraction if wrapped in markdown
            if "```json" in result_str:
                result_str = result_str.split("```json")[1].split("```")[0].strip()
            elif "```" in result_str:
                result_str = result_str.split("```")[1].split("```")[0].strip()
            
            state['extracted_events'] = json.loads(result_str)
        except json.JSONDecodeError:
            state['extracted_events'] = [{"raw_response": result_str}]
            
    except Exception as e:
        print(f"Extractor failed: {e}")
        state['extracted_events'] = []
        
    return state

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Build LangGraph Workflow

# COMMAND ----------
def build_graph():
    """Builds and compiles the LangGraph."""
    workflow = StateGraph(NewsAgentState)
    
    workflow.add_node("gatekeeper", gatekeeper_node)
    workflow.add_node("context_retriever", context_retriever_node)
    workflow.add_node("extractor", extractor_node)
    
    workflow.set_entry_point("gatekeeper")
    
    def route_after_gatekeeper(state: NewsAgentState) -> str:
        if state.get("is_relevant", False):
            return "context_retriever"
        return END

    workflow.add_conditional_edges(
        "gatekeeper",
        route_after_gatekeeper,
        {
            "context_retriever": "context_retriever",
            END: END
        }
    )
    
    workflow.add_edge("context_retriever", "extractor")
    workflow.add_edge("extractor", END)
    
    return workflow.compile()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Evaluation Execution

# COMMAND ----------
@mlflow.trace(name="evaluate_article")
def evaluate_article(app, row):
    """Evaluates a single article through the compiled graph."""
    initial_state = NewsAgentState(
        article_id=str(row.get("story_id", "unknown")),
        article_text=str(row.get("body", "")),
        is_relevant=None,
        context=None,
        extracted_events=None
    )
    return app.invoke(initial_state)

if __name__ == "__main__":
    df_sample = load_eval_dataset(limit=3)
    app = build_graph()
    
    print(f"Evaluating {len(df_sample)} articles through LangGraph...")
    results = []
    for _, row in df_sample.iterrows():
        print(f"--- Processing {row.get('story_id', 'unknown')} ---")
        try:
            final_state = evaluate_article(app, row)
            results.append(final_state)
            print(f"Relevance: {final_state.get('is_relevant')}")
            if final_state.get('is_relevant'):
                print(f"Events: {json.dumps(final_state.get('extracted_events'), indent=2)}")
        except Exception as e:
            print(f"Error evaluating article: {e}")
