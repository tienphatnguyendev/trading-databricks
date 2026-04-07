"""Tests for mechanism validator (TRD-30)."""
from __future__ import annotations

import numpy as np
import pytest

from trading_databricks.agents.config import MechanismValidatorConfig
from trading_databricks.agents.nodes.mechanism_validator import (
    ExtractionInput,
    ValidationResult,
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
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(384).tolist()  # GTE-large dimension

    return _embed
