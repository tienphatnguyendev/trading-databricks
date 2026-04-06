"""Tests for trading_databricks.agents.taxonomy seed data (no Spark required)."""
from __future__ import annotations

from trading_databricks.agents.taxonomy import (
    MECHANISM_WHITELIST_SEED,
    TREATMENT_TAXONOMY_SEED,
)

_VALID_DIRECTIONS = {"bullish", "bearish", "volatile"}
_REQUIRED_L1_CATEGORIES = {
    "Regulation",
    "Security",
    "Adoption",
    "Macro",
    "Exchange",
    "Supply",
    "DeFi",
    "Sentiment",
}


def test_taxonomy_seed_row_count():
    assert len(TREATMENT_TAXONOMY_SEED) == 34


def test_whitelist_seed_row_count():
    assert len(MECHANISM_WHITELIST_SEED) == 40


def test_taxonomy_pk_unique():
    pks = [(r["l1_category"], r["l2_subcategory"]) for r in TREATMENT_TAXONOMY_SEED]
    assert len(pks) == len(set(pks))


def test_whitelist_pk_unique():
    pks = [(r["treatment_l1"], r["mechanism"]) for r in MECHANISM_WHITELIST_SEED]
    assert len(pks) == len(set(pks))


def test_whitelist_valid_directions_non_empty():
    for row in MECHANISM_WHITELIST_SEED:
        assert len(row["valid_directions"]) >= 1, (
            f"Empty valid_directions for {row['treatment_l1']}/{row['mechanism']}"
        )


def test_whitelist_valid_directions_subset():
    for row in MECHANISM_WHITELIST_SEED:
        for direction in row["valid_directions"]:
            assert direction in _VALID_DIRECTIONS, (
                f"Invalid direction '{direction}' in {row['treatment_l1']}/{row['mechanism']}"
            )


def test_taxonomy_l1_alignment():
    taxonomy_l1s = {r["l1_category"] for r in TREATMENT_TAXONOMY_SEED}
    whitelist_l1s = {r["treatment_l1"] for r in MECHANISM_WHITELIST_SEED}
    assert taxonomy_l1s == whitelist_l1s


def test_taxonomy_required_l1_categories():
    l1s = {r["l1_category"] for r in TREATMENT_TAXONOMY_SEED}
    assert _REQUIRED_L1_CATEGORIES == l1s


def test_taxonomy_all_rows_have_required_keys():
    required = {"l1_category", "l2_subcategory", "description", "example"}
    for row in TREATMENT_TAXONOMY_SEED:
        assert required <= set(row.keys()), f"Missing keys in row: {row}"


def test_whitelist_all_rows_have_required_keys():
    required = {"treatment_l1", "mechanism", "valid_directions", "description"}
    for row in MECHANISM_WHITELIST_SEED:
        assert required <= set(row.keys()), f"Missing keys in row: {row}"
