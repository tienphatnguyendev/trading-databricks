"""Gold layer transforms for the news domain.

Pure PySpark functions — no SDP decorators, no SparkSession creation.
NLP library selection is deferred (brainstorm required before implementing).

Source: {catalog}.silver.news_cleaned
Target: {catalog}.gold.news_features (materialized view, full recompute)
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def add_sentiment(df: DataFrame) -> DataFrame:
    """Add sentiment_score column to the DataFrame.

    TODO: brainstorm NLP stack → implement sentiment scoring.
          Candidates: Databricks Foundation Model APIs, Hugging Face via
          Model Serving endpoint, or MLflow pyfunc UDF.
    """
    raise NotImplementedError("NLP stack not yet selected — see TRD-4")


def extract_entities(df: DataFrame) -> DataFrame:
    """Add named entity columns: tickers, companies, people.

    TODO: brainstorm NLP stack → implement entity extraction.
    """
    raise NotImplementedError("NLP stack not yet selected — see TRD-4")


def news_features(df: DataFrame) -> DataFrame:
    """Full gold transform chain for news features.

    TODO: implement after NLP stack is chosen.
    """
    raise NotImplementedError("NLP stack not yet selected — see TRD-4")
