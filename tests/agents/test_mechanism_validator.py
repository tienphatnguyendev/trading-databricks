"""Tests for mechanism validator (TRD-30)."""
from __future__ import annotations

import numpy as np
import pytest

from trading_databricks.agents.config import MechanismValidatorConfig
from trading_databricks.agents.nodes.mechanism_validator import (
    ExtractionInput,
    ValidationResult,
    _compute_evidence_relevance,
    _cosine_similarity,
    _parse_l1_category,
    _validate_evidence_spans,
    validate_extraction,
)


@pytest.fixture
def mock_whitelist() -> dict[tuple[str, str], list[str]]:
    """Sample whitelist matching Wave 1 taxonomy."""
    return {
        ("Regulation", "regulatory_clarity"): ["bullish"],
        ("Regulation", "regulatory_restriction"): ["bearish"],
        ("Security", "sell_pressure"): ["bearish"],
        ("Adoption", "demand_increase"): ["bullish"],
        ("Macro", "risk_on"): ["bullish"],
        ("Macro", "risk_off"): ["bearish"],
    }


@pytest.fixture
def mock_embed_fn():
    """Deterministic embedding function for testing."""

    def _embed(text: str) -> list[float]:
        # Simple hash-based mock: same text → same vector
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.randn(384).tolist()  # GTE-large dimension

    return _embed


@pytest.mark.parametrize(
    "treatment_class,expected",
    [
        ("Regulation.etf_ruling", "Regulation"),
        ("Regulation", "Regulation"),
        ("Regulation.", "Regulation"),
        ("Regulation.etf.extra", "Regulation"),
        (".etf_ruling", ""),
        ("", ""),
    ],
)
def test_parse_l1_category(treatment_class: str, expected: str):
    """Parse L1 from treatment_class with best-effort handling."""
    assert _parse_l1_category(treatment_class) == expected


def test_validate_evidence_spans_degenerate_cases():
    """Reject empty evidence, empty body, whitespace-only spans."""
    # Empty spans
    assert not _validate_evidence_spans([], "article body")

    # Empty body
    assert not _validate_evidence_spans(["span"], "")

    # Whitespace-only span
    assert not _validate_evidence_spans(["   "], "article body")

    # Span normalizes to empty
    assert not _validate_evidence_spans(["  \n  "], "article")


def test_validate_evidence_spans_normalized_matching():
    """Casefold + strip allows case/whitespace variations."""
    # Case and whitespace variation
    assert _validate_evidence_spans(["  Bitcoin PRICE  "], "The bitcoin price is rising")

    # Substring match (not word boundaries)
    assert _validate_evidence_spans(["bitcoin"], "The bitcoin price")

    # Multiple spans, all must match
    assert _validate_evidence_spans(
        ["SEC approved", "spot ETF"], "The SEC approved the first spot ETF"
    )

    # One span doesn't match → reject
    assert not _validate_evidence_spans(
        ["SEC approved", "ethereum"], "The SEC approved Bitcoin ETF"
    )


def test_validate_evidence_spans_fabricated_evidence():
    """Evidence not in article body → reject."""
    assert not _validate_evidence_spans(
        ["ethereum price drops"], "The SEC approved the Bitcoin ETF"
    )


# ===== _cosine_similarity tests =====


def test_cosine_similarity_identical_vectors():
    """Identical vectors → similarity = 1.0."""
    vec = [1.0, 2.0, 3.0]
    assert _cosine_similarity(vec, vec) == 1.0


def test_cosine_similarity_orthogonal_vectors():
    """Orthogonal vectors → similarity = 0.0."""
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0]
    assert _cosine_similarity(vec_a, vec_b) == 0.0


def test_cosine_similarity_opposite_vectors():
    """Opposite vectors → clamped to 0.0 (not -1.0)."""
    vec_a = [1.0, 2.0, 3.0]
    vec_b = [-1.0, -2.0, -3.0]
    result = _cosine_similarity(vec_a, vec_b)
    assert result == 0.0  # Clamped


def test_cosine_similarity_zero_length_vectors():
    """Zero-length vectors → 0.0 (no similarity)."""
    assert _cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0
    assert _cosine_similarity([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == 0.0


def test_cosine_similarity_similar_vectors():
    """Similar vectors → 0.0 < similarity < 1.0."""
    vec_a = [1.0, 2.0, 3.0]
    vec_b = [1.1, 2.1, 2.9]
    result = _cosine_similarity(vec_a, vec_b)
    assert 0.9 < result < 1.0  # Very similar


# ===== _compute_evidence_relevance tests =====


def test_compute_evidence_relevance_success(mock_embed_fn):
    """Module B computes score when embeddings succeed."""
    result = _compute_evidence_relevance(
        trigger_event="SEC approves ETF",
        mechanism="regulatory_clarity",
        evidence_spans=["The SEC approved the first spot ETF"],
        embed_fn=mock_embed_fn,
    )

    assert result is not None
    assert 0.0 <= result <= 1.0


def test_compute_evidence_relevance_empty_evidence():
    """Empty evidence → None immediately, no embed call."""
    call_count = 0

    def counting_embed_fn(text: str) -> list[float]:
        nonlocal call_count
        call_count += 1
        return [0.0] * 384

    # Empty list
    result = _compute_evidence_relevance(
        trigger_event="event",
        mechanism="mechanism",
        evidence_spans=[],
        embed_fn=counting_embed_fn,
    )
    assert result is None
    assert call_count == 0  # Never called

    # Whitespace-only spans
    result = _compute_evidence_relevance(
        trigger_event="event",
        mechanism="mechanism",
        evidence_spans=["   ", "  \n  "],
        embed_fn=counting_embed_fn,
    )
    assert result is None
    assert call_count == 0  # Never called


def test_compute_evidence_relevance_embed_failure():
    """When embed_fn raises, return None and log warning."""

    def failing_embed_fn(text: str) -> list[float]:
        raise TimeoutError("Model endpoint unavailable")

    result = _compute_evidence_relevance(
        trigger_event="SEC approves ETF",
        mechanism="regulatory_clarity",
        evidence_spans=["The SEC approved..."],
        embed_fn=failing_embed_fn,
    )

    assert result is None  # Graceful degradation


# ===== validate_extraction integration tests =====


def test_validate_extraction_accepted(mock_whitelist, mock_embed_fn):
    """Valid extraction passes all checks."""
    extraction: ExtractionInput = {
        "treatment_class": "Regulation.etf_ruling",
        "trigger_event": "SEC approves ETF",
        "mechanism": "regulatory_clarity",
        "expected_direction": "bullish",
        "evidence_spans": ["The SEC approved the first spot ETF"],
    }

    result = validate_extraction(
        extraction,
        article_body="The SEC approved the first spot ETF for Bitcoin",
        whitelist=mock_whitelist,
        config=MechanismValidatorConfig(),
        embed_fn=mock_embed_fn,
    )

    assert result["validation_status"] == "accepted"
    assert result["rejected_mechanism_raw"] is None
    assert result["evidence_relevance_score"] is not None
    assert 0.0 <= result["evidence_relevance_score"] <= 1.0


def test_validate_extraction_rejected_mechanism(mock_whitelist, mock_embed_fn):
    """Unknown mechanism → rejected_mechanism, Module B still runs."""
    extraction: ExtractionInput = {
        "treatment_class": "Regulation.etf_ruling",
        "trigger_event": "SEC approves",
        "mechanism": "unknown_mechanism",  # Not in whitelist
        "expected_direction": "bullish",
        "evidence_spans": ["The SEC approved the first spot ETF"],
    }

    result = validate_extraction(
        extraction,
        article_body="The SEC approved the first spot ETF for Bitcoin",
        whitelist=mock_whitelist,
        config=MechanismValidatorConfig(),
        embed_fn=mock_embed_fn,
    )

    assert result["validation_status"] == "rejected_mechanism"
    assert result["rejected_mechanism_raw"] == "unknown_mechanism"
    assert result["evidence_relevance_score"] is not None  # Module B still ran!
    assert 0.0 <= result["evidence_relevance_score"] <= 1.0


def test_validate_extraction_rejected_direction(mock_whitelist, mock_embed_fn):
    """Invalid direction → rejected_direction, Module B runs."""
    extraction: ExtractionInput = {
        "treatment_class": "Regulation.etf_ruling",
        "trigger_event": "SEC approves",
        "mechanism": "regulatory_clarity",
        "expected_direction": "bearish",  # Not in valid_directions
        "evidence_spans": ["The SEC approved"],
    }

    result = validate_extraction(
        extraction,
        article_body="The SEC approved the ETF",
        whitelist=mock_whitelist,
        config=MechanismValidatorConfig(),
        embed_fn=mock_embed_fn,
    )

    assert result["validation_status"] == "rejected_direction"
    assert result["rejected_mechanism_raw"] is None
    assert result["evidence_relevance_score"] is not None  # Module B runs


def test_validate_extraction_rejected_evidence(mock_whitelist, mock_embed_fn):
    """Evidence not in article body → rejected_evidence, Module B tries."""
    extraction: ExtractionInput = {
        "treatment_class": "Regulation.etf_ruling",
        "trigger_event": "SEC approves",
        "mechanism": "regulatory_clarity",
        "expected_direction": "bullish",
        "evidence_spans": ["ethereum price drops"],  # Not in article
    }

    result = validate_extraction(
        extraction,
        article_body="The SEC approved the Bitcoin ETF",
        whitelist=mock_whitelist,
        config=MechanismValidatorConfig(),
        embed_fn=mock_embed_fn,
    )

    assert result["validation_status"] == "rejected_evidence"
    assert result["rejected_mechanism_raw"] is None
    # Module B tries to score, likely succeeds (spans exist as strings)
    # or returns None if embed_fn rejects the text


def test_validate_extraction_malformed_treatment_class(mock_whitelist, mock_embed_fn):
    """Malformed treatment_class → fails at whitelist lookup."""
    extraction: ExtractionInput = {
        "treatment_class": ".malformed",  # Parses to empty L1
        "trigger_event": "event",
        "mechanism": "regulatory_clarity",
        "expected_direction": "bullish",
        "evidence_spans": ["evidence"],
    }

    result = validate_extraction(
        extraction,
        article_body="evidence text",
        whitelist=mock_whitelist,
        config=MechanismValidatorConfig(),
        embed_fn=mock_embed_fn,
    )

    assert result["validation_status"] == "rejected_mechanism"
    assert result["rejected_mechanism_raw"] == "regulatory_clarity"
