"""Tests for Gatekeeper Stage 1 — spaCy NER filter (TRD-25)."""

from trading_databricks.agents.config import GatekeeperConfig
from trading_databricks.agents.gatekeeper import (
    CRYPTO_ENTITY_PATTERNS,
    build_nlp,
    extract_ner_features,
    is_crypto_relevant,
)

_CONFIG = GatekeeperConfig()
_EXPECTED_LABELS = {
    "CRYPTO_ASSET",
    "CRYPTO_EXCHANGE",
    "CRYPTO_STABLECOIN",
    "CRYPTO_PROTOCOL",
    "CRYPTO_CONCEPT",
}


def test_ner_patterns() -> None:
    """All 50 patterns present, all 5 category labels covered."""
    assert len(CRYPTO_ENTITY_PATTERNS) == 50
    labels = {p["label"] for p in CRYPTO_ENTITY_PATTERNS}
    assert labels == _EXPECTED_LABELS


def test_extract_ner_features_all_keys() -> None:
    """extract_ner_features returns all 6 expected keys."""
    nlp = build_nlp()
    doc = nlp("Bitcoin staking on Uniswap")
    features = extract_ner_features(doc)
    assert set(features.keys()) == {
        "has_crypto_asset",
        "has_exchange",
        "has_stablecoin",
        "has_protocol",
        "has_concept",
        "entity_count",
    }


def test_extract_ner_features_multi_category() -> None:
    """Multi-category text yields correct booleans and entity_count >= 3."""
    nlp = build_nlp()
    doc = nlp("Bitcoin staking on Uniswap")  # ASSET + CONCEPT + PROTOCOL
    features = extract_ner_features(doc)
    assert features["has_crypto_asset"] is True
    assert features["has_concept"] is True
    assert features["has_protocol"] is True
    assert features["has_exchange"] is False
    assert features["has_stablecoin"] is False
    assert features["entity_count"] >= 3


def test_build_nlp_cached() -> None:
    """build_nlp() returns the same object on repeated calls (cache works)."""
    assert build_nlp() is build_nlp()


def test_bitcoin_etf_approved() -> None:
    """Acceptance criterion #2: CRYPTO_ASSET in headline → impactful."""
    result, conf = is_crypto_relevant("Bitcoin ETF approved", "", config=_CONFIG)
    assert result is True
    assert conf == 1.0


def test_uk_inflation_not_impactful() -> None:
    """Acceptance criterion #3: no crypto entity → not impactful."""
    result, conf = is_crypto_relevant(
        "UK inflation data surprises upward", "", config=_CONFIG
    )
    assert result is False
    assert conf == 0.0


def test_fed_defi_yields() -> None:
    """Acceptance criterion #4: DeFi (CRYPTO_CONCEPT) → impactful."""
    result, conf = is_crypto_relevant(
        "Fed rate decision affects DeFi yields", "", config=_CONFIG
    )
    assert result is True
    assert conf == 1.0


def test_body_truncated_to_text_window() -> None:
    """Entity found only beyond text_window chars in body is NOT detected."""
    # body_prefix is exactly 500 chars (text_window default), no crypto entity
    body_prefix = "a " * 250  # 250 * 2 chars = 500 chars exactly
    body_suffix = "Bitcoin surges."  # would match, but falls beyond window
    body = body_prefix + body_suffix
    result, _ = is_crypto_relevant("stocks rally", body, config=_CONFIG)
    assert result is False
