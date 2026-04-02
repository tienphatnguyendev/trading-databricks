import mlflow
from pyspark.sql import SparkSession
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

if __name__ == "__main__":
    df_sample = load_eval_dataset()
    print(f"Loaded {len(df_sample)} rows for evaluation.")
