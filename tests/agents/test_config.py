"""Tests for trading_databricks.agents.config frozen dataclasses."""
from __future__ import annotations

import pytest

from trading_databricks.agents.config import (
    GatekeeperConfig,
    MASConfig,
    MechanismValidatorConfig,
    UncertaintyRouterConfig,
)


class TestGatekeeperConfig:
    def test_defaults(self):
        cfg = GatekeeperConfig()
        assert cfg.model_uri == "models:/news-gatekeeper/Production"
        assert cfg.confidence_threshold == 0.5
        assert cfg.ner_only_mode is True
        assert cfg.text_window == 500

    def test_frozen(self):
        cfg = GatekeeperConfig()
        with pytest.raises(Exception):
            cfg.model_uri = "other"  # type: ignore[misc]

    def test_custom_values(self):
        cfg = GatekeeperConfig(confidence_threshold=0.7, ner_only_mode=False)
        assert cfg.confidence_threshold == 0.7
        assert cfg.ner_only_mode is False
        assert cfg.text_window == 500  # other defaults preserved


class TestUncertaintyRouterConfig:
    def test_defaults(self):
        cfg = UncertaintyRouterConfig()
        assert cfg.high_similarity_threshold == 0.85
        assert cfg.low_similarity_threshold == 0.50
        assert cfg.self_consistency_k == 3
        assert cfg.min_vs_results == 2
        assert cfg.min_index_size == 50

    def test_frozen(self):
        cfg = UncertaintyRouterConfig()
        with pytest.raises(Exception):
            cfg.self_consistency_k = 10  # type: ignore[misc]

    def test_custom_thresholds(self):
        cfg = UncertaintyRouterConfig(high_similarity_threshold=0.9, self_consistency_k=5)
        assert cfg.high_similarity_threshold == 0.9
        assert cfg.self_consistency_k == 5
        assert cfg.low_similarity_threshold == 0.50  # default preserved


class TestMechanismValidatorConfig:
    def test_defaults(self):
        cfg = MechanismValidatorConfig()
        assert cfg.evidence_grounding_reject_threshold is None
        assert cfg.embedding_model == "databricks-gte-large-en"

    def test_frozen(self):
        cfg = MechanismValidatorConfig()
        with pytest.raises(Exception):
            cfg.embedding_model = "other"  # type: ignore[misc]

    def test_none_threshold_means_log_only(self):
        # None = informational only in V3; rejection threshold added in V4
        cfg = MechanismValidatorConfig()
        assert cfg.evidence_grounding_reject_threshold is None

    def test_custom_threshold(self):
        cfg = MechanismValidatorConfig(evidence_grounding_reject_threshold=0.3)
        assert cfg.evidence_grounding_reject_threshold == 0.3


class TestMASConfig:
    def test_constructs_with_all_nested_defaults(self):
        cfg = MASConfig()
        assert isinstance(cfg.gatekeeper, GatekeeperConfig)
        assert isinstance(cfg.router, UncertaintyRouterConfig)
        assert isinstance(cfg.validator, MechanismValidatorConfig)

    def test_nested_default_values_correct(self):
        cfg = MASConfig()
        assert cfg.gatekeeper.confidence_threshold == 0.5
        assert cfg.gatekeeper.ner_only_mode is True
        assert cfg.router.high_similarity_threshold == 0.85
        assert cfg.router.min_index_size == 50
        assert cfg.validator.evidence_grounding_reject_threshold is None
        assert cfg.validator.embedding_model == "databricks-gte-large-en"

    def test_scalar_defaults(self):
        cfg = MASConfig()
        assert cfg.catalog == "trading-signal-gen-dev"
        assert cfg.schema == "gold"
        assert cfg.vs_index_name == "trading-signal-gen-dev.gold.causal_chains_idx"
        assert cfg.vs_endpoint_name == "mas-few-shot"
        assert cfg.llm_endpoint == "databricks-claude-sonnet"
        assert cfg.agent_model_version == "v1.0"

    def test_frozen(self):
        cfg = MASConfig()
        with pytest.raises(Exception):
            cfg.catalog = "other"  # type: ignore[misc]

    def test_custom_gatekeeper_preserves_other_defaults(self):
        cfg = MASConfig(gatekeeper=GatekeeperConfig(ner_only_mode=False))
        assert cfg.gatekeeper.ner_only_mode is False
        assert cfg.gatekeeper.confidence_threshold == 0.5
        assert cfg.router.high_similarity_threshold == 0.85  # unrelated config unaffected

    def test_catalog_override(self):
        cfg = MASConfig(catalog="trading-signal-gen-prod")
        assert cfg.catalog == "trading-signal-gen-prod"
        assert cfg.gatekeeper.confidence_threshold == 0.5  # nested unaffected

    def test_two_instances_are_independent(self):
        cfg_a = MASConfig(catalog="trading-signal-gen-dev")
        cfg_b = MASConfig(catalog="trading-signal-gen-prod")
        assert cfg_a.catalog != cfg_b.catalog
        # nested objects are independent instances
        assert cfg_a.gatekeeper is not cfg_b.gatekeeper
