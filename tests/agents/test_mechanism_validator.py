"""Tests for mechanism validator (TRD-30)."""
from __future__ import annotations

import numpy as np
import pytest

from trading_databricks.agents.config import MechanismValidatorConfig
from trading_databricks.agents.nodes.mechanism_validator import (
    ExtractionInput,
    ValidationResult,
    _parse_l1_category,
    _validate_evidence_spans,
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
