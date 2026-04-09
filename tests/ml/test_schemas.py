from trading_databricks.ml.schemas import (
    CUSUM_EVENTS_SCHEMA,
    FEATURES_NEWS_SCHEMA,
    SIGNALS_SCHEMA,
)


def test_cusum_events_schema_count() -> None:
    assert len(CUSUM_EVENTS_SCHEMA.fields) == 11

def test_signals_schema_count() -> None:
    assert len(SIGNALS_SCHEMA.fields) == 13

def test_features_news_schema_count() -> None:
    # Key columns (4) + 35 news features
    assert len(FEATURES_NEWS_SCHEMA.fields) == 39
