"""Shared Unity Catalog identifiers and table path helpers.

This module is the single source of truth for default catalog/schema/table
names used across the repository.
"""

from __future__ import annotations


DEFAULT_CATALOG = "smart-trading-dev"

BRONZE_SCHEMA = "01_bronze"
SILVER_SCHEMA = "02_silver"
GOLD_SCHEMA = "03_gold"
OTHERS_SCHEMA = "others"

RAW_AGGTRADES_SAMPLES_TABLE = "raw_aggtrades_samples"
NEWS_AGENT_EVAL_DATASET_TABLE = "news_agent_eval_dataset"
DOLLAR_BARS_TABLE = "dollar_bars"
MECHANISM_WHITELIST_TABLE = "mechanism_whitelist"
TREATMENT_TAXONOMY_TABLE = "treatment_taxonomy"
NEWS_CAUSAL_FEATURES_TABLE = "news_causal_features"
MAS_TRAINING_DATASET_TABLE = "mas_training_dataset"
CAUSAL_CHAINS_INDEX_TABLE = "causal_chains_idx"
OHLC_GOLD_TABLE = "ohlc_gold"

MAS_VS_ENDPOINT_NAME = "mas-few-shot"
MAS_LLM_ENDPOINT_NAME = "databricks-meta-llama-3-1-8b-instruct"
EMBEDDING_ENDPOINT_NAME = "databricks-gte-large-en"


def table_fqn(catalog: str, schema: str, relation: str) -> str:
    """Build a Unity Catalog fully-qualified object path."""
    return f"{catalog}.{schema}.{relation}"


def quoted_table_fqn(catalog: str, schema: str, relation: str) -> str:
    """Build a quoted Unity Catalog fully-qualified object path."""
    return f"`{catalog}`.`{schema}`.{relation}"


RAW_AGGTRADES_SAMPLES_FQN = table_fqn(
    DEFAULT_CATALOG,
    BRONZE_SCHEMA,
    RAW_AGGTRADES_SAMPLES_TABLE,
)
NEWS_AGENT_EVAL_DATASET_FQN = table_fqn(
    DEFAULT_CATALOG,
    BRONZE_SCHEMA,
    NEWS_AGENT_EVAL_DATASET_TABLE,
)
DOLLAR_BARS_FQN = table_fqn(
    DEFAULT_CATALOG,
    GOLD_SCHEMA,
    DOLLAR_BARS_TABLE,
)
MECHANISM_WHITELIST_FQN = table_fqn(
    DEFAULT_CATALOG,
    GOLD_SCHEMA,
    MECHANISM_WHITELIST_TABLE,
)
TREATMENT_TAXONOMY_FQN = table_fqn(
    DEFAULT_CATALOG,
    GOLD_SCHEMA,
    TREATMENT_TAXONOMY_TABLE,
)
NEWS_CAUSAL_FEATURES_FQN = table_fqn(
    DEFAULT_CATALOG,
    GOLD_SCHEMA,
    NEWS_CAUSAL_FEATURES_TABLE,
)
MAS_TRAINING_DATASET_FQN = table_fqn(
    DEFAULT_CATALOG,
    OTHERS_SCHEMA,
    MAS_TRAINING_DATASET_TABLE,
)
MAS_VS_INDEX_NAME = table_fqn(
    DEFAULT_CATALOG,
    GOLD_SCHEMA,
    CAUSAL_CHAINS_INDEX_TABLE,
)

EVAL_SCHEMA = "04_eval"

NEWS_FIREHOSE_INDEX_TABLE = "news_firehose_idx"
MECHANISM_CANDIDATES_REVIEW_TABLE = "mechanism_candidates_review"

NEWS_FIREHOSE_INDEX_FQN = table_fqn(
    DEFAULT_CATALOG,
    GOLD_SCHEMA,
    NEWS_FIREHOSE_INDEX_TABLE,
)
MECHANISM_CANDIDATES_REVIEW_FQN = table_fqn(
    DEFAULT_CATALOG,
    EVAL_SCHEMA,
    MECHANISM_CANDIDATES_REVIEW_TABLE,
)
