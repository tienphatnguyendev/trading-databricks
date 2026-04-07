"""Gatekeeper Stage 1 — spaCy EntityRuler crypto NER filter.

Stage 1 is rule-based only: no model download, no MLflow dependency.
Runs on 100% of articles. Short-circuit: no crypto entity → is_impactful=False.
"""
from __future__ import annotations

import functools

import spacy

from trading_databricks.agents.config import GatekeeperConfig

CRYPTO_ENTITY_PATTERNS: list[dict] = [
    # ── CRYPTO_ASSET: Primary tracked assets (11) ──
    {"label": "CRYPTO_ASSET", "pattern": [{"LOWER": "bitcoin"}]},
    {"label": "CRYPTO_ASSET", "pattern": [{"TEXT": "BTC"}]},
    {"label": "CRYPTO_ASSET", "pattern": [{"LOWER": "ethereum"}]},
    {"label": "CRYPTO_ASSET", "pattern": [{"TEXT": "ETH"}]},
    {"label": "CRYPTO_ASSET", "pattern": [{"LOWER": "ether"}]},
    {"label": "CRYPTO_ASSET", "pattern": [{"LOWER": "crypto"}]},
    {"label": "CRYPTO_ASSET", "pattern": [{"LOWER": "cryptocurrency"}]},
    {"label": "CRYPTO_ASSET", "pattern": [{"LOWER": "altcoin"}]},
    {"label": "CRYPTO_ASSET", "pattern": [{"TEXT": "DOGE"}]},
    {"label": "CRYPTO_ASSET", "pattern": [{"TEXT": "XRP"}]},
    {"label": "CRYPTO_ASSET", "pattern": [{"LOWER": "solana"}]},
    # ── CRYPTO_EXCHANGE (8) ──
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "binance"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "coinbase"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "kraken"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "bybit"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"TEXT": "OKX"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "ftx"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "gemini"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "bitstamp"}]},
    # ── CRYPTO_STABLECOIN (5) ──
    {"label": "CRYPTO_STABLECOIN", "pattern": [{"TEXT": "USDC"}]},
    {"label": "CRYPTO_STABLECOIN", "pattern": [{"TEXT": "USDT"}]},
    {"label": "CRYPTO_STABLECOIN", "pattern": [{"LOWER": "tether"}]},
    {"label": "CRYPTO_STABLECOIN", "pattern": [{"LOWER": "stablecoin"}]},
    {
        "label": "CRYPTO_STABLECOIN",
        "pattern": [{"LOWER": "dai"}, {"LOWER": "stablecoin"}],
    },
    # ── CRYPTO_PROTOCOL (10) ──
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "uniswap"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "aave"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "makerdao"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "chainlink"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "arbitrum"}]},
    {
        "label": "CRYPTO_PROTOCOL",
        "pattern": [{"LOWER": "compound"}, {"LOWER": "finance"}],
    },
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"TEXT": "COMP"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"TEXT": "OP"}, {"LOWER": "token"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "lido"}, {"LOWER": "finance"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "steth"}]},
    # ── CRYPTO_CONCEPT (16) ──
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "blockchain"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "defi"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "nft"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "halving"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "staking"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "tokenomics"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "layer"}, {"TEXT": "2"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"TEXT": "L2"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "smart"}, {"LOWER": "contract"}]},
    {
        "label": "CRYPTO_CONCEPT",
        "pattern": [{"LOWER": "proof"}, {"LOWER": "of"}, {"LOWER": "work"}],
    },
    {
        "label": "CRYPTO_CONCEPT",
        "pattern": [{"LOWER": "proof"}, {"LOWER": "of"}, {"LOWER": "stake"}],
    },
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "decentralized"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "on-chain"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "whitepaper"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "bitcoin"}, {"LOWER": "mining"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "crypto"}, {"LOWER": "mining"}]},
]


@functools.cache
def build_nlp() -> spacy.Language:
    """Build blank spaCy pipeline with EntityRuler. No model download needed.

    Cached per process — safe for repeated calls in tests and non-UDF contexts.
    UDF callers (TRD-26) should call build_nlp() inside the UDF body; the cache
    is per-executor-process, not serialized, so no serialization trap.
    """
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(CRYPTO_ENTITY_PATTERNS)
    return nlp


def extract_ner_features(doc: spacy.tokens.Doc) -> dict[str, bool | int]:
    """Return boolean entity category flags and total entity count.

    Args:
        doc: A spaCy Doc processed by build_nlp().

    Returns:
        Dict with keys: has_crypto_asset, has_exchange, has_stablecoin,
        has_protocol, has_concept, entity_count.
    """
    labels = {ent.label_ for ent in doc.ents}
    return {
        "has_crypto_asset": "CRYPTO_ASSET" in labels,
        "has_exchange":     "CRYPTO_EXCHANGE" in labels,
        "has_stablecoin":   "CRYPTO_STABLECOIN" in labels,
        "has_protocol":     "CRYPTO_PROTOCOL" in labels,
        "has_concept":      "CRYPTO_CONCEPT" in labels,
        "entity_count":     len(doc.ents),
    }


def is_crypto_relevant(
    headline: str,
    body: str,
    *,
    config: GatekeeperConfig,
) -> tuple[bool, float]:
    """Stage 1 NER filter. Returns (is_impactful, confidence).

    Concatenates headline + body[:config.text_window], runs EntityRuler.
    No entity found → (False, 0.0). Any entity found → (True, 1.0).

    Note: ner_only_mode=True assumed (V1). Stage 2 sklearn (TRD-26) extends this.
    """
    text = headline + " " + body[: config.text_window]
    doc = build_nlp()(text)
    features = extract_ner_features(doc)
    if features["entity_count"] == 0:
        return False, 0.0
    return True, 1.0
