from unittest.mock import MagicMock
import pandas as pd

class MockSpark:
    def sql(self, query, args=None):
        print(f"Executing query: {query}")
        print(f"With args: {args}")
        # Return a mock DataFrame that supports collect() or toPandas() or toArrow()
        mock_df = MagicMock()
        mock_df.collect.return_value = [{"relevance": "Financially Relevant", "extracted": "{\"cause\": \"test\"}"}]
        mock_df.toPandas.return_value = pd.DataFrame([{"story_id": 1}])
        mock_df.toArrow.return_value = MagicMock()
        return mock_df

spark = MockSpark()

# Test cases based on my changes
# 1. notebooks/research_mas_causal_news.py
query1 = """
    SELECT ai_classify(
        :text,
        ARRAY('Financially Relevant', 'Not Financially Relevant')
    ) as relevance
    """
spark.sql(query1, args={"text": "some article text"})

# 2. notebooks/qa/utils.py
query2 = "SELECT * FROM `smart-trading-dev`.others.ml_training_data ORDER BY timestamp"
query2 += " LIMIT :limit"
spark.sql(query2, args={"limit": 10})

print("Verification script finished successfully (mocked).")
