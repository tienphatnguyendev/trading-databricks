"""Mechanism validator with 5-step validation + Module B evidence grounding.

Wave 2 (TRD-30)

Implements:
- 2026-04-04-mas-final-spec.md §7.1-7.2 (5-step validator logic)
- 2026-04-04-robust-trustworthy-mas-spec.md §3.1-3.5 (Module B grounding)

Exports:
    - ExtractionInput (TypedDict)
    - ValidationResult (TypedDict)
    - validate_extraction (main function)
"""
from __future__ import annotations

import logging
from typing import Callable, TypedDict

import numpy as np

from trading_databricks.agents.config import MechanismValidatorConfig

logger = logging.getLogger(__name__)


class ExtractionInput(TypedDict):
    """LLM extraction output consumed by the validator.

    Matches the subset of CAUSAL_FEATURES_SCHEMA fields that the
    extractor produces. The validator needs these 5 fields to run
    the 5-step validation pipeline.
    """

    treatment_class: str  # "L1.l2_subcategory"
    trigger_event: str  # max 200 chars, for Module B claim embedding
    mechanism: str  # lowercase snake_case
    expected_direction: str  # "bullish" | "bearish" | "volatile"
    evidence_spans: list[str]  # 1-3 quoted text fragments from article


class ValidationResult(TypedDict):
    """Validation-only fields returned by validate_extraction().

    Caller merges these into the full extraction dict before writing
    to gold.news_causal_features.
    """

    validation_status: str  # "accepted" | "rejected_mechanism" | "rejected_direction" | "rejected_evidence"
    rejected_mechanism_raw: str | None  # populated only when validation_status = "rejected_mechanism"
    evidence_relevance_score: float | None  # [0.0, 1.0] cosine similarity, None on embed failure or empty evidence


def validate_extraction(
    extraction: ExtractionInput,
    article_body: str,
    whitelist: dict[tuple[str, str], list[str]],
    *,
    config: MechanismValidatorConfig,
    embed_fn: Callable[[str], list[float]],
) -> ValidationResult:
    """Validate an LLM extraction through 5-step validation pipeline.

    Stub implementation. Full implementation follows in Tasks 2-6.

    Args:
        extraction: LLM extraction output
        article_body: Full article text for evidence matching
        whitelist: Taxonomy whitelist from load_whitelist()
        config: MechanismValidatorConfig settings
        embed_fn: Embedding function (str) -> list[float]

    Returns:
        Validation result with status and optional rejection reasons

    Raises:
        NotImplementedError: Tasks 2-6 pending implementation
    """
    raise NotImplementedError("validate_extraction: Tasks 2-6 pending")
