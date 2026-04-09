from trading_databricks.ml.schemas import (
    CUSUM_EVENTS_SCHEMA, CUSUM_EVENT_STATS_SCHEMA, TRIPLE_BARRIER_LABELS_SCHEMA,
    FEATURES_TECHNICAL_SCHEMA, FEATURES_VOLATILITY_SCHEMA, FEATURES_VOLUME_SCHEMA,
    FEATURES_NEWS_SCHEMA, TRAINING_DATASET_SCHEMA, SIGNALS_SCHEMA,
    POSITIONS_SCHEMA, PROMOTION_EVIDENCE_SCHEMA, TRIAL_REGISTRY_SCHEMA,
    DATA_QUALITY_SCHEMA
)

def test_cusum_events_schema_count():
    assert len(CUSUM_EVENTS_SCHEMA.fields) == 11

def test_signals_schema_count():
    assert len(SIGNALS_SCHEMA.fields) == 13

def test_features_news_schema_count():
    # Key columns (4) + 35 news features
    assert len(FEATURES_NEWS_SCHEMA.fields) == 39
