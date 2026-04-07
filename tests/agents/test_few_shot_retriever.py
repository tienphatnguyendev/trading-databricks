"""Tests for the few-shot retriever node."""
from __future__ import annotations

import logging
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from trading_databricks.agents.nodes.few_shot_retriever import (
    STATIC_FEW_SHOT_EXAMPLES,
    _accepted_count_cache,
    get_accepted_chain_count,
    merge_examples,
    query_vector_search,
    MASState,
)
from trading_databricks.agents.config import MASConfig
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


class TestVectorSearchQuery:
    """Test VS query logic (mocked MCP)."""

    def test_query_returns_results(self, monkeypatch):
        """VS query returns expected format with story_ids."""
        def mock_query_vs_index(*args, **kwargs):
            return [
                {
                    "story_id": "test-123",
                    "treatment_class": "Macro.rate_decision",
                    "mechanism": "liquidity_contraction",
                    "expected_direction": "bearish",
                    "confidence": 0.85,
                    "trigger_event": "Fed raises rates",
                    "causal_chain_json": '{"trigger_event": "Fed raises rates"}',
                },
                {
                    "story_id": "test-456",
                    "treatment_class": "Regulation.etf_ruling",
                    "mechanism": "institutional_access",
                    "expected_direction": "bullish",
                    "confidence": 0.90,
                    "trigger_event": "SEC approves ETF",
                    "causal_chain_json": '{"trigger_event": "SEC approves ETF"}',
                },
            ]

        # Mock the MCP tool
        import sys
        few_shot_retriever_module = sys.modules["trading_databricks.agents.nodes.few_shot_retriever"]
        monkeypatch.setattr(few_shot_retriever_module, "query_vs_index", mock_query_vs_index)

        results = query_vector_search(
            "Bitcoin ETF approved",
            index_name="test.gold.causal_chains_idx",
            num_results=5,
        )

        assert len(results) == 2
        assert results[0]["story_id"] == "test-123"
        assert results[1]["story_id"] == "test-456"

    def test_query_handles_empty_results(self, monkeypatch):
        """Empty VS results return empty list (no error)."""
        def mock_query_vs_index(*args, **kwargs):
            return []

        import sys
        few_shot_retriever_module = sys.modules["trading_databricks.agents.nodes.few_shot_retriever"]
        monkeypatch.setattr(few_shot_retriever_module, "query_vs_index", mock_query_vs_index)

        results = query_vector_search(
            "Obscure altcoin news",
            index_name="test.gold.causal_chains_idx",
        )

        assert results == []

    def test_query_handles_vs_error(self, monkeypatch, caplog):
        """VS query failure returns empty list, logs warning."""
        def mock_query_vs_index(*args, **kwargs):
            raise RuntimeError("VS endpoint unavailable")

        import sys
        few_shot_retriever_module = sys.modules["trading_databricks.agents.nodes.few_shot_retriever"]
        monkeypatch.setattr(few_shot_retriever_module, "query_vs_index", mock_query_vs_index)

        with caplog.at_level(logging.WARNING):
            results = query_vector_search(
                "Test headline",
                index_name="test.gold.causal_chains_idx",
            )

        assert results == []
        assert "VS query failed" in caplog.text
        assert "returning empty results" in caplog.text


class TestMergeExamples:
    """Test dynamic + static merging logic."""

    def test_merge_dynamic_only(self):
        """5 dynamic + static -> returns first 5 dynamic."""
        dynamic = [{"story_id": f"dyn-{i}"} for i in range(5)]
        static = [{"story_id": f"static-{i}"} for i in range(5)]

        result = merge_examples(dynamic, static, max_total=5)

        assert len(result) == 5
        assert all(ex["story_id"].startswith("dyn-") for ex in result)

    def test_merge_dynamic_plus_static(self):
        """2 dynamic + static -> returns 2 dynamic + 3 static."""
        dynamic = [{"story_id": "dyn-0"}, {"story_id": "dyn-1"}]
        static = [{"story_id": f"static-{i}"} for i in range(5)]

        result = merge_examples(dynamic, static, max_total=5)

        assert len(result) == 5
        assert result[0]["story_id"] == "dyn-0"
        assert result[1]["story_id"] == "dyn-1"
        assert result[2]["story_id"] == "static-0"
        assert result[3]["story_id"] == "static-1"
        assert result[4]["story_id"] == "static-2"

    def test_merge_static_fallback(self):
        """0 dynamic + static -> returns 5 static."""
        dynamic = []
        static = [{"story_id": f"static-{i}"} for i in range(5)]

        result = merge_examples(dynamic, static, max_total=5)

        assert len(result) == 5
        assert all(ex["story_id"].startswith("static-") for ex in result)

    def test_merge_respects_max_total(self):
        """Result never exceeds max_total."""
        dynamic = [{"story_id": f"dyn-{i}"} for i in range(10)]
        static = [{"story_id": f"static-{i}"} for i in range(10)]

        result = merge_examples(dynamic, static, max_total=3)

        assert len(result) == 3
        assert all(ex["story_id"].startswith("dyn-") for ex in result)


class TestFewShotRetrieverNode:
    """Integration tests for the full few_shot_retriever node."""

    def test_cold_start_returns_static(self, monkeypatch):
        """count < min_index_size -> returns static examples only."""
        # Clear cache
        _accepted_count_cache.clear()

        # Mock Spark to return count < 50
        mock_spark = MagicMock()
        mock_result = MagicMock()
        mock_result.collect.return_value = [(25,)]
        mock_spark.sql.return_value = mock_result

        def mock_get_spark():
            return mock_spark

        few_shot_retriever_module = sys.modules["trading_databricks.agents.nodes.few_shot_retriever"]
        monkeypatch.setattr(few_shot_retriever_module, "get_spark_session", mock_get_spark)

        # Import after patching to get the patched function
        from trading_databricks.agents.nodes.few_shot_retriever import few_shot_retriever

        # Input state
        state: MASState = {
            "headline": "Bitcoin price surges",
            "few_shot_examples": [],
            "retrieval_story_ids": [],
        }
        config = MASConfig()

        # Call node
        result = few_shot_retriever(state, config)

        # Verify cold-start: returns static only
        assert len(result["few_shot_examples"]) == 5
        assert all("static-example" in ex["story_id"] for ex in result["few_shot_examples"])
        assert result["retrieval_story_ids"] == []

        _accepted_count_cache.clear()

    def test_warm_start_queries_vs(self, monkeypatch):
        """count >= min_index_size -> queries VS, merges dynamic + static."""
        _accepted_count_cache.clear()

        # Mock Spark to return count >= 50
        mock_spark = MagicMock()
        mock_result = MagicMock()
        mock_result.collect.return_value = [(75,)]
        mock_spark.sql.return_value = mock_result

        def mock_get_spark():
            return mock_spark

        # Mock VS query to return 2 dynamic results
        def mock_query_vs_index(*args, **kwargs):
            return [
                {"story_id": "dyn-1", "treatment_class": "Macro.rate_decision"},
                {"story_id": "dyn-2", "treatment_class": "Regulation.etf_ruling"},
            ]

        few_shot_retriever_module = sys.modules["trading_databricks.agents.nodes.few_shot_retriever"]
        monkeypatch.setattr(few_shot_retriever_module, "get_spark_session", mock_get_spark)
        monkeypatch.setattr(few_shot_retriever_module, "query_vs_index", mock_query_vs_index)

        # Import after patching
        from trading_databricks.agents.nodes.few_shot_retriever import few_shot_retriever

        # Input state
        state: MASState = {
            "headline": "Fed raises interest rates",
            "few_shot_examples": [],
            "retrieval_story_ids": [],
        }
        config = MASConfig()

        # Call node
        result = few_shot_retriever(state, config)

        # Verify warm-start: 2 dynamic + 3 static
        assert len(result["few_shot_examples"]) == 5
        assert result["few_shot_examples"][0]["story_id"] == "dyn-1"
        assert result["few_shot_examples"][1]["story_id"] == "dyn-2"
        assert "static-example" in result["few_shot_examples"][2]["story_id"]

        # Verify lineage tracking (only dynamic story IDs)
        assert result["retrieval_story_ids"] == ["dyn-1", "dyn-2"]

        _accepted_count_cache.clear()
