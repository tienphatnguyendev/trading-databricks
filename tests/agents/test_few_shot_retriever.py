"""Tests for the few-shot retriever node."""
from __future__ import annotations

import pytest

from trading_databricks.agents.nodes.few_shot_retriever import STATIC_FEW_SHOT_EXAMPLES
from trading_databricks.agents.taxonomy import MECHANISM_WHITELIST_SEED


class TestStaticExamples:
    """Verify static example data quality."""

    def test_static_examples_count(self):
        """STATIC_FEW_SHOT_EXAMPLES has exactly 5 examples."""
        assert len(STATIC_FEW_SHOT_EXAMPLES) == 5

    def test_static_examples_schema(self):
        """Each example has required fields matching VS result format."""
        required_fields = {
            "story_id",
            "treatment_class",
            "mechanism",
            "expected_direction",
            "confidence",
            "trigger_event",
            "causal_chain_json",
        }

        for example in STATIC_FEW_SHOT_EXAMPLES:
            assert set(example.keys()) == required_fields
            assert isinstance(example["story_id"], str)
            assert example["story_id"].startswith("static-example-")
            assert isinstance(example["confidence"], float)
            assert 0.0 <= example["confidence"] <= 1.0

    def test_static_examples_coverage(self):
        """Static examples cover at least 4 distinct L1 categories."""
        l1_categories = {
            example["treatment_class"].split(".")[0]
            for example in STATIC_FEW_SHOT_EXAMPLES
        }
        assert len(l1_categories) >= 4

    def test_static_examples_whitelist_compliance(self):
        """All (L1, mechanism) pairs exist in MECHANISM_WHITELIST_SEED."""
        whitelist_pairs = {
            (entry["treatment_l1"], entry["mechanism"])
            for entry in MECHANISM_WHITELIST_SEED
        }

        for example in STATIC_FEW_SHOT_EXAMPLES:
            l1_category = example["treatment_class"].split(".")[0]
            mechanism = example["mechanism"]
            assert (l1_category, mechanism) in whitelist_pairs


import time
from unittest.mock import MagicMock, patch

from trading_databricks.agents.nodes.few_shot_retriever import (
    get_accepted_chain_count,
    _accepted_count_cache,
)


class TestAcceptedChainCount:
    """Test count query and caching."""

    def test_count_query_with_accepted_chains(self):
        """Returns correct count when accepted chains exist."""
        # Clear cache
        _accepted_count_cache.clear()

        # Mock Spark session
        mock_spark = MagicMock()
        mock_result = MagicMock()
        mock_result.collect.return_value = [(42,)]
        mock_spark.sql.return_value = mock_result

        count = get_accepted_chain_count(
            mock_spark,
            catalog="test_catalog",
            schema="test_schema",
        )

        assert count == 42
        mock_spark.sql.assert_called_once()
        assert "test_catalog" in mock_spark.sql.call_args[0][0]
        assert "validation_status = 'accepted'" in mock_spark.sql.call_args[0][0]

        # Clear cache
        _accepted_count_cache.clear()

    def test_count_query_with_zero_chains(self):
        """Returns 0 when no accepted chains exist."""
        # Clear cache
        _accepted_count_cache.clear()

        mock_spark = MagicMock()
        mock_result = MagicMock()
        mock_result.collect.return_value = [(0,)]
        mock_spark.sql.return_value = mock_result

        count = get_accepted_chain_count(
            mock_spark,
            catalog="test_catalog",
            schema="test_schema",
        )

        assert count == 0

        # Clear cache
        _accepted_count_cache.clear()

    def test_caching_behavior(self):
        """Second call within TTL uses cached value, not DB query."""
        # Clear cache
        _accepted_count_cache.clear()

        mock_spark = MagicMock()
        mock_result = MagicMock()
        mock_result.collect.return_value = [(100,)]
        mock_spark.sql.return_value = mock_result

        # First call - should query DB
        count1 = get_accepted_chain_count(
            mock_spark,
            catalog="test_catalog",
            schema="test_schema",
            cache_ttl_seconds=300,
        )
        assert count1 == 100
        assert mock_spark.sql.call_count == 1

        # Second call within TTL - should use cache
        count2 = get_accepted_chain_count(
            mock_spark,
            catalog="test_catalog",
            schema="test_schema",
            cache_ttl_seconds=300,
        )
        assert count2 == 100
        assert mock_spark.sql.call_count == 1  # Still 1, not 2

        # Clear cache
        _accepted_count_cache.clear()

    def test_cache_expiry(self):
        """Cache expires after TTL."""
        _accepted_count_cache.clear()

        mock_spark = MagicMock()
        mock_result = MagicMock()
        mock_result.collect.return_value = [(200,)]
        mock_spark.sql.return_value = mock_result

        # First call
        count1 = get_accepted_chain_count(
            mock_spark,
            catalog="test_catalog",
            schema="test_schema",
            cache_ttl_seconds=1,  # 1 second TTL
        )
        assert count1 == 200
        assert mock_spark.sql.call_count == 1

        # Wait for cache to expire
        time.sleep(1.1)

        # Second call after TTL - should query DB again
        count2 = get_accepted_chain_count(
            mock_spark,
            catalog="test_catalog",
            schema="test_schema",
            cache_ttl_seconds=1,
        )
        assert count2 == 200
        assert mock_spark.sql.call_count == 2

        _accepted_count_cache.clear()
