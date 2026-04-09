from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

CUSUM_EVENTS_SCHEMA = StructType([
    StructField("event_id", StringType(), False),
    StructField("symbol", StringType(), False),
    StructField("event_ts", TimestampType(), False),
    StructField("bar_index", LongType(), False),
    StructField("log_return", DoubleType(), False),
    StructField("cusum_side", StringType(), False),
    StructField("cusum_value", DoubleType(), False),
    StructField("threshold_used", DoubleType(), False),
    StructField("ewma_vol", DoubleType(), False),
    StructField("avg_bars_per_day", DoubleType(), False),
    StructField("_ingested_at", TimestampType(), False),
])

CUSUM_EVENT_STATS_SCHEMA = StructType([
    StructField("symbol", StringType(), False),
    StructField("date", DateType(), False),
    StructField("event_count", IntegerType(), False),
    StructField("threshold_used", DoubleType(), False),
    StructField("ewma_vol", DoubleType(), False),
    StructField("avg_bars_per_day", DoubleType(), False),
])

TRIPLE_BARRIER_LABELS_SCHEMA = StructType([
    StructField("event_id", StringType(), False),
    StructField("symbol", StringType(), False),
    StructField("event_ts", TimestampType(), False),
    StructField("label", IntegerType(), False),
    StructField("barrier_hit", StringType(), False),
    StructField("return_pct", DoubleType(), False),
    StructField("upper_level", DoubleType(), False),
    StructField("lower_level", DoubleType(), False),
    StructField("vertical_bar", LongType(), False),
    StructField("matured_at", TimestampType(), False),
    StructField("avg_uniqueness", DoubleType(), False),
])

FEATURES_TECHNICAL_SCHEMA = StructType([
    StructField("event_id", StringType(), False),
    StructField("symbol", StringType(), False),
    StructField("event_ts", TimestampType(), False),
    StructField("computed_at", TimestampType(), False),
    StructField("rsi_14", DoubleType(), True),
    StructField("macd_signal", DoubleType(), True),
    StructField("bb_pct", DoubleType(), True),
    StructField("sma_ratio_5_20", DoubleType(), True),
    StructField("sma_ratio_10_50", DoubleType(), True),
    StructField("log_return_5", DoubleType(), True),
    StructField("log_return_20", DoubleType(), True),
])

FEATURES_VOLATILITY_SCHEMA = StructType([
    StructField("event_id", StringType(), False),
    StructField("symbol", StringType(), False),
    StructField("event_ts", TimestampType(), False),
    StructField("computed_at", TimestampType(), False),
    StructField("realized_vol_20", DoubleType(), True),
    StructField("realized_vol_50", DoubleType(), True),
    StructField("parkinson_vol_20", DoubleType(), True),
    StructField("garman_klass_20", DoubleType(), True),
    StructField("vol_ratio_20_50", DoubleType(), True),
])

FEATURES_VOLUME_SCHEMA = StructType([
    StructField("event_id", StringType(), False),
    StructField("symbol", StringType(), False),
    StructField("event_ts", TimestampType(), False),
    StructField("computed_at", TimestampType(), False),
    StructField("vwap_dev_20", DoubleType(), True),
    StructField("relative_volume_20", DoubleType(), True),
    StructField("obv_slope_20", DoubleType(), True),
    StructField("dollar_volume_ratio", DoubleType(), True),
])

# Dynamically generate News features (4 metrics x 8 L1 categories + 3 global)
_news_fields = [
    StructField("event_id", StringType(), False),
    StructField("symbol", StringType(), False),
    StructField("event_ts", TimestampType(), False),
    StructField("computed_at", TimestampType(), False),
]
for l1 in (
    "Sentiment",
    "DeFi",
    "Supply",
    "Exchange",
    "Macro",
    "Adoption",
    "Security",
    "Regulation",
):
    _news_fields.extend([
        StructField(f"news_count_{l1}", IntegerType(), True),
        StructField(f"news_max_conf_{l1}", DoubleType(), True),
        StructField(f"news_direction_{l1}", IntegerType(), True),
        StructField(f"news_accepted_ratio_{l1}", DoubleType(), True),
    ])
_news_fields.extend([
    StructField("news_volume_24h", IntegerType(), True),
    StructField("news_sentiment_score", DoubleType(), True),
    StructField("has_news_24h", BooleanType(), True),
])
FEATURES_NEWS_SCHEMA = StructType(_news_fields)

TRAINING_DATASET_SCHEMA = StructType([
    StructField("dataset_version", StringType(), False),
    StructField("event_id", StringType(), False),
    StructField("symbol", StringType(), False),
    StructField("event_ts", TimestampType(), False),
    StructField("label", IntegerType(), True),
    StructField("avg_uniqueness", DoubleType(), False),
    StructField("cpcv_group", IntegerType(), False),
    # ... (Technical 7)
    StructField("rsi_14", DoubleType(), True),
    StructField("macd_signal", DoubleType(), True),
    StructField("bb_pct", DoubleType(), True),
    StructField("sma_ratio_5_20", DoubleType(), True),
    StructField("sma_ratio_10_50", DoubleType(), True),
    StructField("log_return_5", DoubleType(), True),
    StructField("log_return_20", DoubleType(), True),
    # ... (Volatility 5)
    StructField("realized_vol_20", DoubleType(), True),
    StructField("realized_vol_50", DoubleType(), True),
    StructField("parkinson_vol_20", DoubleType(), True),
    StructField("garman_klass_20", DoubleType(), True),
    StructField("vol_ratio_20_50", DoubleType(), True),
    # ... (Volume 4)
    StructField("vwap_dev_20", DoubleType(), True),
    StructField("relative_volume_20", DoubleType(), True),
    StructField("obv_slope_20", DoubleType(), True),
    StructField("dollar_volume_ratio", DoubleType(), True),
    # ... (News 35)
    *[
        _f
        for _f in _news_fields
        if _f.name not in ("event_id", "symbol", "event_ts", "computed_at")
    ],
    StructField("created_at", TimestampType(), False),
])

SIGNALS_SCHEMA = StructType([
    StructField("signal_id", StringType(), False),
    StructField("symbol", StringType(), False),
    StructField("event_id", StringType(), False),
    StructField("event_ts", TimestampType(), False),
    StructField("primary_side", IntegerType(), False),
    StructField("primary_confidence", DoubleType(), False),
    StructField("meta_label", IntegerType(), False),
    StructField("meta_confidence", DoubleType(), False),
    StructField("raw_bet_size", DoubleType(), False),
    StructField("discretized_weight", DoubleType(), False),
    StructField("primary_model_version", IntegerType(), False),
    StructField("meta_model_version", IntegerType(), False),
    StructField("created_at", TimestampType(), False),
])

POSITIONS_SCHEMA = StructType([
    StructField("symbol", StringType(), False),
    StructField("side", StringType(), False),
    StructField("weight", DoubleType(), False),
    StructField("confidence", DoubleType(), False),
    StructField("signal_id", StringType(), False),
    StructField("updated_at", TimestampType(), False),
])

PROMOTION_EVIDENCE_SCHEMA = StructType([
    StructField("run_id", StringType(), False),
    StructField("model_type", StringType(), False),
    StructField("model_version", IntegerType(), False),
    StructField("dataset_version", StringType(), False),
    StructField("psr", DoubleType(), False),
    StructField("dsr", DoubleType(), False),
    StructField("max_drawdown", DoubleType(), False),
    StructField("time_under_water", IntegerType(), False),
    StructField("cost_robust", BooleanType(), False),
    StructField("strategy_failure_prob", DoubleType(), False),
    StructField("meta_precision", DoubleType(), False),
    StructField("meta_recall", DoubleType(), False),
    StructField("meta_f1", DoubleType(), False),
    StructField("meta_logloss", DoubleType(), False),
    StructField("all_gates_pass", BooleanType(), False),
    StructField("failure_reasons", StringType(), True),
    StructField("evaluated_at", TimestampType(), False),
])

TRIAL_REGISTRY_SCHEMA = StructType([
    StructField("trial_id", StringType(), False),
    StructField("run_id", StringType(), False),
    StructField("model_type", StringType(), False),
    StructField("algorithm", StringType(), False),
    StructField("sharpe", DoubleType(), False),
    StructField("created_at", TimestampType(), False),
])

DATA_QUALITY_SCHEMA = StructType([
    StructField("check_id", StringType(), False),
    StructField("check_name", StringType(), False),
    StructField("symbol", StringType(), False),
    StructField("status", StringType(), False),
    StructField("metric_value", DoubleType(), False),
    StructField("threshold", DoubleType(), False),
    StructField("message", StringType(), False),
    StructField("checked_at", TimestampType(), False),
])
