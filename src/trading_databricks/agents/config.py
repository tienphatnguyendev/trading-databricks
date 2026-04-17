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
)


@dataclass(frozen=True)
class GatekeeperConfig:
    model_uri: str = "models:/news-gatekeeper/Production"
    confidence_threshold: float = 0.5
    ner_only_mode: bool = True   # V1: NER only, no MLflow dependency
    text_window: int = 500       # chars of body concatenated with headline


@dataclass(frozen=True)
class UncertaintyRouterConfig:
    high_similarity_threshold: float = 0.85   # fast path (VS cosine ≥ this)
    low_similarity_threshold: float = 0.50    # careful path (VS cosine < this)
    self_consistency_k: int = 3               # LLM calls on careful path
    min_vs_results: int = 2                   # minimum VS matches for fast path
    min_index_size: int = 50                  # below this: force careful regardless


@dataclass(frozen=True)
class MechanismValidatorConfig:
    evidence_grounding_reject_threshold: float | None = None  # None = log only (V3)
    embedding_model: str = EMBEDDING_ENDPOINT_NAME


@dataclass(frozen=True)
class MASConfig:
    gatekeeper: GatekeeperConfig = field(default_factory=GatekeeperConfig)
    router: UncertaintyRouterConfig = field(default_factory=UncertaintyRouterConfig)
    validator: MechanismValidatorConfig = field(default_factory=MechanismValidatorConfig)
    catalog: str = DEFAULT_CATALOG
    schema: str = GOLD_SCHEMA
    vs_index_name: str = MAS_VS_INDEX_NAME
    vs_endpoint_name: str = MAS_VS_ENDPOINT_NAME
    llm_endpoint: str = MAS_LLM_ENDPOINT_NAME
    agent_model_version: str = "v1.0"
