import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
import pandas as pd
from typing import TypedDict, List, Dict, Any, Optional

class NewsAgentState(TypedDict):
    article_id: str
    article_text: str
    is_relevant: Optional[bool]
    context: Optional[str]
    extracted_events: Optional[List[Dict[str, Any]]]

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

def load_eval_dataset(limit: int = 5) -> pd.DataFrame:
    """Loads the news agent eval dataset for testing."""
    df = spark.table("`trading-signal-gen-dev`.`bronze`.`news_agent_eval_dataset`")
    return df.limit(limit).toPandas()

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

if __name__ == "__main__":
    df_sample = load_eval_dataset(limit=1)
    if not df_sample.empty:
        sample_row = df_sample.iloc[0]
        initial_state = NewsAgentState(
            article_id=str(sample_row.get("story_id", "test_id")),
            article_text=str(sample_row.get("body", "Bitcoin hits new all time high.")),
            is_relevant=None,
            context=None,
            extracted_events=None
        )
        new_state = gatekeeper_node(initial_state)
        print(f"Gatekeeper Result: {new_state['is_relevant']}")
