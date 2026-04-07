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


def _parse_l1_category(treatment_class: str) -> str:
    """Extract L1 from treatment_class using best-effort parsing.

    Examples:
        "Regulation.etf_ruling"     → "Regulation"
        "Regulation"                → "Regulation"  (no dot, return as-is)
        "Regulation."               → "Regulation"  (trailing dot)
        "Regulation.etf.extra"      → "Regulation"  (split on first dot)
        ".etf_ruling"               → ""            (malformed, empty L1)
        ""                          → ""            (empty input)

    Malformed cases naturally fail at step 2 whitelist lookup.
    No exceptions raised — let downstream validation catch it.
    """
    if "." not in treatment_class:
        return treatment_class
    return treatment_class.split(".", maxsplit=1)[0]


def _validate_evidence_spans(
    evidence_spans: list[str],
    article_body: str,
) -> bool:
    """Check that all evidence spans exist in article body (normalized).

    Normalization: casefold() + strip() — prevents false rejections from
    LLM whitespace/casing variations while catching fabricated evidence.

    Returns False (rejection) when:
    - evidence_spans is empty []
    - article_body is empty ""
    - Any span is empty or whitespace-only after normalization
    - Any span not found as substring of normalized article body
    """
    normed_body = article_body.casefold().strip()
    normed_spans = [s.casefold().strip() for s in evidence_spans]

    # Degenerate cases: empty evidence, empty body, or whitespace-only spans
    if not normed_spans or not normed_body or any(not s for s in normed_spans):
        return False

    # All spans must exist in body
    return all(span in normed_body for span in normed_spans)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors.

    Formula: cos(θ) = (A · B) / (||A|| × ||B||)

    Returns:
        float in [0.0, 1.0] - clamped to match schema contract

    Clamping rationale:
        Raw cosine similarity is [-1, 1], but negative values have no
        meaningful interpretation for "evidence relevance." Evidence
        semantically opposed to the claim has zero relevance, not
        negative relevance.

    Edge cases:
        - Zero-length vectors → 0.0 (no similarity)
        - Negative raw similarity → 0.0 (clamped)
    """
    a = np.array(vec_a)
    b = np.array(vec_b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    raw_similarity = np.dot(a, b) / (norm_a * norm_b)
    return float(max(0.0, raw_similarity))  # Clamp to [0.0, 1.0]


def _compute_evidence_relevance(
    trigger_event: str,
    mechanism: str,
    evidence_spans: list[str],
    *,
    embed_fn: Callable[[str], list[float]],
) -> float | None:
    """Score semantic relevance between causal claim and evidence.

    ANM-inspired check: if evidence truly grounds the claim, embeddings
    should be semantically similar. Low similarity suggests evidence is
    filler text or the LLM is hallucinating.

    Implementation: Cosine similarity between:
    - Claim: "{trigger_event} causes {mechanism}"
    - Evidence: " ".join(evidence_spans)

    Returns:
        float in [0.0, 1.0] - cosine similarity score
        None - on edge cases (empty evidence) or embedding failures

    Error handling:
        Module B is informational only in V3. Embedding failures are
        caught, logged, and return None. Validation proceeds.
    """
    # Edge case: empty or whitespace-only evidence
    if not evidence_spans or all(not s.strip() for s in evidence_spans):
        return None  # Cannot score — no evidence to embed

    try:
        # Construct claim and evidence text
        claim_text = f"{trigger_event} causes {mechanism}"
        evidence_text = " ".join(evidence_spans)

        # Call embedding endpoint (2 calls, ~50 tokens each)
        claim_embedding = embed_fn(claim_text)
        evidence_embedding = embed_fn(evidence_text)

        # Compute cosine similarity
        similarity = _cosine_similarity(claim_embedding, evidence_embedding)

        return float(similarity)

    except Exception as e:
        # Graceful degradation: log and continue without score
        logger.warning(
            "Module B embedding failed, continuing without evidence_relevance_score",
            exc_info=True,
            extra={
                "trigger_event": trigger_event[:50],  # truncate for log size
                "mechanism": mechanism,
                "error": str(e),
            },
        )
        return None


def validate_extraction(
    extraction: ExtractionInput,
    article_body: str,
    whitelist: dict[tuple[str, str], list[str]],
    *,
    config: MechanismValidatorConfig,
    embed_fn: Callable[[str], list[float]],
) -> ValidationResult:
    """Run 5-step validation. Steps 1-4 deterministic. Step 5 (Module B) informational.

    Args:
        extraction: LLM output with treatment_class, mechanism, direction, evidence
        article_body: Full article text for evidence substring matching
        whitelist: {(treatment_l1, mechanism): [valid_directions]} from load_whitelist()
        config: MechanismValidatorConfig (contains evidence_grounding_reject_threshold, unused in V3)
        embed_fn: Embedding function, signature (str) -> list[float], injected by caller

    Returns:
        ValidationResult with 3 fields:
            - validation_status: "accepted" | "rejected_mechanism" | "rejected_direction" | "rejected_evidence"
            - rejected_mechanism_raw: str | None (only for rejected_mechanism)
            - evidence_relevance_score: float [0, 1] | None (None on embed failure or empty evidence)

    Error handling:
        Module B embedding failures are caught and logged. The function always
        returns a ValidationResult — no uncaught exceptions propagate to caller.
    """
    # Step 1: Parse L1 from treatment_class
    l1 = _parse_l1_category(extraction["treatment_class"])
    mechanism = extraction["mechanism"]

    # Steps 2-4: Determine validation status (NO early returns)
    if (l1, mechanism) not in whitelist:
        status = "rejected_mechanism"
        rejected_raw = mechanism
    elif extraction["expected_direction"] not in whitelist[(l1, mechanism)]:
        status = "rejected_direction"
        rejected_raw = None
    elif not _validate_evidence_spans(extraction["evidence_spans"], article_body):
        status = "rejected_evidence"
        rejected_raw = None
    else:
        status = "accepted"
        rejected_raw = None

    # Step 5: ALWAYS runs (Module B — informational, even for rejected chains)
    score = _compute_evidence_relevance(
        extraction["trigger_event"],
        mechanism,
        extraction["evidence_spans"],
        embed_fn=embed_fn,
    )

    return {
        "validation_status": status,
        "rejected_mechanism_raw": rejected_raw,
        "evidence_relevance_score": score,
    }
