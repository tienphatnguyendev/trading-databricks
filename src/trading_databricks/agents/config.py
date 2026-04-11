"""Frozen configuration dataclasses for all MAS components.

All values match the spec defaults exactly. Pass instances through
agent nodes rather than threading raw parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field


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
    embedding_model: str = "databricks-gte-large-en"


@dataclass(frozen=True)
class MASConfig:
    gatekeeper: GatekeeperConfig = field(default_factory=GatekeeperConfig)
    router: UncertaintyRouterConfig = field(default_factory=UncertaintyRouterConfig)
    validator: MechanismValidatorConfig = field(default_factory=MechanismValidatorConfig)
    catalog: str = "trading-signal-gen-dev"
    schema: str = "gold"
    vs_index_name: str = "trading-signal-gen-dev.gold.causal_chains_idx"
    vs_endpoint_name: str = "mas-few-shot"
    llm_endpoint: str = "databricks-meta-llama-3-1-8b-instruct"
    agent_model_version: str = "v1.0"
