"""Tests for UC identifier helpers."""
from trading_databricks.common.uc_identifiers import (
    NEWS_FIREHOSE_INDEX_TABLE,
    NEWS_FIREHOSE_INDEX_FQN,
    MECHANISM_CANDIDATES_REVIEW_TABLE,
    MECHANISM_CANDIDATES_REVIEW_FQN,
    EVAL_SCHEMA,
)


def test_news_firehose_index_identifiers():
    assert NEWS_FIREHOSE_INDEX_TABLE == "news_firehose_idx"
    assert NEWS_FIREHOSE_INDEX_FQN.endswith(".news_firehose_idx")
    assert "03_gold" in NEWS_FIREHOSE_INDEX_FQN


def test_mechanism_candidates_review_identifiers():
    assert MECHANISM_CANDIDATES_REVIEW_TABLE == "mechanism_candidates_review"
    assert MECHANISM_CANDIDATES_REVIEW_FQN.endswith(".mechanism_candidates_review")
    assert EVAL_SCHEMA == "04_eval"
    assert EVAL_SCHEMA in MECHANISM_CANDIDATES_REVIEW_FQN
