"""Frozen configuration dataclasses for all MAS components.

All values match the spec defaults exactly. Pass instances through
agent nodes rather than threading raw parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from trading_databricks.common.uc_identifiers import (
    EMBEDDING_ENDPOINT_NAME,
    GOLD_SCHEMA,
    DEFAULT_CATALOG,
    MAS_LLM_ENDPOINT_NAME,
    MAS_VS_ENDPOINT_NAME,
    MAS_VS_INDEX_NAME,
    NEWS_FIREHOSE_INDEX_FQN,
    OHLC_GOLD_TABLE,
)


@dataclass(frozen=True)
class GatekeeperConfig:
    model_uri: str = "models:/news-gatekeeper/Production"
    confidence_threshold: float = 0.5
    ner_only_mode: bool = True
    text_window: int = 500


@dataclass(frozen=True)
class UncertaintyRouterConfig:
    high_similarity_threshold: float = 0.85
    low_similarity_threshold: float = 0.50
    self_consistency_k: int = 3
    min_vs_results: int = 2
    min_index_size: int = 50


@dataclass(frozen=True)
class MechanismValidatorConfig:
    evidence_grounding_reject_threshold: float | None = None
    embedding_model: str = EMBEDDING_ENDPOINT_NAME


@dataclass(frozen=True)
class ReflectionConfig:
    """Bounded reflection loop config for the causality sub-pipeline.

    max_reflections=1 means at most one retry of hypothesis_generator after a
    judge_panel failure (worst case = 2 hypothesis_generator calls per article).
    """
    max_reflections: int = 1


@dataclass(frozen=True)
class JudgePanelConfig:
    """Online LLM judge panel config.

    Three judges (direction, mechanism, grounding) run in parallel via
    LangGraph Send API. Strict-AND aggregation: panel passes only if all three pass.
    Per-judge timeout prevents one slow judge from blocking the panel.
    """
    judge_endpoint: str = MAS_LLM_ENDPOINT_NAME
    timeout_s: float = 10.0
    grounding_threshold: float = 0.5  # grounding judge returns float; ≥ this → pass


@dataclass(frozen=True)
class RAGRetrieverConfig:
    """Deep RAG config: precedent news + market outcome join."""
    firehose_index_name: str = NEWS_FIREHOSE_INDEX_FQN
    timeout_s: float = 5.0
    num_precedents: int = 5
    outcome_window_hours: int = 24
    ohlc_table: str = OHLC_GOLD_TABLE


@dataclass(frozen=True)
class MASConfig:
    gatekeeper: GatekeeperConfig = field(default_factory=GatekeeperConfig)
    router: UncertaintyRouterConfig = field(default_factory=UncertaintyRouterConfig)
    validator: MechanismValidatorConfig = field(default_factory=MechanismValidatorConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    judge_panel: JudgePanelConfig = field(default_factory=JudgePanelConfig)
    rag: RAGRetrieverConfig = field(default_factory=RAGRetrieverConfig)
    catalog: str = DEFAULT_CATALOG
    schema: str = GOLD_SCHEMA
    vs_index_name: str = MAS_VS_INDEX_NAME
    vs_endpoint_name: str = MAS_VS_ENDPOINT_NAME
    llm_endpoint: str = MAS_LLM_ENDPOINT_NAME
    agent_model_version: str = "v1.0"
