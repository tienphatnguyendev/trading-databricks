"""Tests for trading_databricks.news.gold transforms.

Gold news feature functions are stubs (NotImplementedError) until the NLP
stack is selected in TRD-4.  These tests verify the stub contract.
"""
from __future__ import annotations

import pytest
from pyspark.sql import Row


def test_add_sentiment_raises_not_implemented(spark):
    from trading_databricks.news.gold import add_sentiment

    df = spark.createDataFrame([Row(story_id="s1", headline="Test")])
    with pytest.raises(NotImplementedError):
        add_sentiment(df)


def test_extract_entities_raises_not_implemented(spark):
    from trading_databricks.news.gold import extract_entities

    df = spark.createDataFrame([Row(story_id="s1", headline="Test")])
    with pytest.raises(NotImplementedError):
        extract_entities(df)


def test_news_features_raises_not_implemented(spark):
    from trading_databricks.news.gold import news_features

    df = spark.createDataFrame([Row(story_id="s1")])
    with pytest.raises(NotImplementedError):
        news_features(df)
