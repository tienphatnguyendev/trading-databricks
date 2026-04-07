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
