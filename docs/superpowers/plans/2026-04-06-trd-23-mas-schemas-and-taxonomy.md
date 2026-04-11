# TRD-23: MAS Schemas and Taxonomy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement five `StructType` schema constants in `schemas.py`, the 34-row treatment taxonomy and 40-entry mechanism whitelist seed data with load/seed functions in `taxonomy.py`, and a one-shot `scripts/seed_taxonomy.py` seeder — giving all downstream MAS Wave 2+ agents their data contracts and reference data.

**Architecture:** Two new source files (`schemas.py`, `taxonomy.py`) and one script (`scripts/seed_taxonomy.py`). Schemas are module-level constants; taxonomy seed data is module-level lists; functions use keyword-only params. Parallel to TRD-24 (separate worktree); this branch merges second (rebases onto main after TRD-24 merges).

**Tech Stack:** Python 3.12, PySpark `StructType` (no Spark session needed for schemas), `databricks-connect` for seed script, `pytest`, `uv`

**Spec:** `docs/superpowers/specs/2026-04-06-mas-w1-schema-config-design.md` §3

---

## File Map

| Action | Path |
|--------|------|
| Create | `src/trading_databricks/agents/schemas.py` |
| Create | `src/trading_databricks/agents/taxonomy.py` |
| Create | `scripts/seed_taxonomy.py` |
| Create | `tests/agents/__init__.py` |
| Create | `tests/agents/test_schemas.py` |
| Create | `tests/agents/test_taxonomy.py` |

---

## Task 1: Branch Setup

- [ ] **Step 1.1: Create branch from main**

```bash
git checkout main && git pull
git checkout -b feat/TRD-23-mas-schemas-and-taxonomy
```

Expected: Switched to a new branch 'feat/TRD-23-mas-schemas-and-taxonomy'

- [ ] **Step 1.2: Create tests/agents package**

```bash
touch tests/agents/__init__.py
```

- [ ] **Step 1.3: Create empty test files**

```bash
touch tests/agents/test_schemas.py tests/agents/test_taxonomy.py
```

- [ ] **Step 1.4: Verify test discovery works**

```bash
uv run pytest tests/agents/ --collect-only
```

Expected: `no tests ran` (files are empty) — no errors

---

## Task 2: CAUSAL_FEATURES_SCHEMA

- [ ] **Step 2.1: Write failing tests for CAUSAL_FEATURES_SCHEMA**

Write `tests/agents/test_schemas.py`:

```python
"""Tests for trading_databricks.agents.schemas StructType definitions."""
from __future__ import annotations

from pyspark.sql.types import (
    ArrayType,
    DecimalType,
    DoubleType,
    StringType,
)

from trading_databricks.agents.schemas import (
    CAUSAL_FEATURES_SCHEMA,
    CONFOUNDING_AUDIT_SCHEMA,
    DAGNESS_AUDIT_SCHEMA,
    MECHANISM_WHITELIST_SCHEMA,
    TREATMENT_TAXONOMY_SCHEMA,
)


class TestCausalFeaturesSchema:
    def test_field_count(self):
        assert len(CAUSAL_FEATURES_SCHEMA.fields) == 23

    def test_non_nullable_fields_exactly(self):
        non_nullable = {f.name for f in CAUSAL_FEATURES_SCHEMA.fields if not f.nullable}
        assert non_nullable == {
            "story_id",
            "published_at",
            "is_impactful",
            "agent_model_version",
            "processed_at",
        }

    def test_dollar_bar_close_is_decimal_18_8(self):
        field = next(f for f in CAUSAL_FEATURES_SCHEMA.fields if f.name == "dollar_bar_close")
        assert isinstance(field.dataType, DecimalType)
        assert field.dataType == DecimalType(18, 8)
        assert field.nullable is True

    def test_evidence_spans_is_array_of_string(self):
        field = next(f for f in CAUSAL_FEATURES_SCHEMA.fields if f.name == "evidence_spans")
        assert isinstance(field.dataType, ArrayType)
        assert isinstance(field.dataType.elementType, StringType)
        assert field.nullable is True

    def test_retrieval_story_ids_is_array_of_string(self):
        field = next(f for f in CAUSAL_FEATURES_SCHEMA.fields if f.name == "retrieval_story_ids")
        assert isinstance(field.dataType, ArrayType)
        assert isinstance(field.dataType.elementType, StringType)

    def test_v3_field_extraction_path_exists_and_nullable(self):
        field = next(f for f in CAUSAL_FEATURES_SCHEMA.fields if f.name == "extraction_path")
        assert isinstance(field.dataType, StringType)
        assert field.nullable is True

    def test_v3_field_evidence_relevance_score_is_double_nullable(self):
        field = next(
            f for f in CAUSAL_FEATURES_SCHEMA.fields if f.name == "evidence_relevance_score"
        )
        assert isinstance(field.dataType, DoubleType)
        assert field.nullable is True

    def test_v3_field_confounding_cluster_id_exists_and_nullable(self):
        field = next(
            f for f in CAUSAL_FEATURES_SCHEMA.fields if f.name == "confounding_cluster_id"
        )
        assert isinstance(field.dataType, StringType)
        assert field.nullable is True

    def test_field_order_story_id_first(self):
        assert CAUSAL_FEATURES_SCHEMA.fields[0].name == "story_id"

    def test_field_order_v3_fields_last(self):
        names = [f.name for f in CAUSAL_FEATURES_SCHEMA.fields]
        assert names[-3] == "extraction_path"
        assert names[-2] == "evidence_relevance_score"
        assert names[-1] == "confounding_cluster_id"
```

- [ ] **Step 2.2: Run to confirm ImportError**

```bash
uv run pytest tests/agents/test_schemas.py::TestCausalFeaturesSchema -v
```

Expected: `ModuleNotFoundError: No module named 'trading_databricks.agents.schemas'`

- [ ] **Step 2.3: Implement schemas.py with CAUSAL_FEATURES_SCHEMA**

Create `src/trading_databricks/agents/schemas.py`:

```python
"""StructType definitions for the MAS (Multi-Agent System) domain.

All schemas are module-level constants and represent Delta table contracts.
Gold layer schemas are frozen — field renames or removals require explicit approval.
"""
from __future__ import annotations

from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DateType,
    DecimalType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ── gold.news_causal_features (23 fields) ─────────────────────────────────────
# Fields 1-6: always populated
# Fields 7-17: NULL when is_impactful=False
# Fields 18-20: NULL until TRD-5 (dollar bars market context)
# Fields 21-23: V3 additions (nullable, backward compatible)
CAUSAL_FEATURES_SCHEMA = StructType(
    [
        # Always populated
        StructField("story_id",                 StringType(),            nullable=False),  # 1  PK
        StructField("published_at",             TimestampType(),         nullable=False),  # 2
        StructField("is_impactful",             BooleanType(),           nullable=False),  # 3
        StructField("gatekeeper_confidence",    DoubleType(),            nullable=True),   # 4
        StructField("agent_model_version",      StringType(),            nullable=False),  # 5
        StructField("processed_at",             TimestampType(),         nullable=False),  # 6
        # NULL when is_impactful=False
        StructField("treatment_class",          StringType(),            nullable=True),   # 7
        StructField("trigger_event",            StringType(),            nullable=True),   # 8
        StructField("mechanism",                StringType(),            nullable=True),   # 9
        StructField("asset",                    StringType(),            nullable=True),   # 10
        StructField("expected_direction",       StringType(),            nullable=True),   # 11
        StructField("confidence",               DoubleType(),            nullable=True),   # 12
        StructField("causal_chain_json",        StringType(),            nullable=True),   # 13
        StructField("evidence_spans",           ArrayType(StringType()), nullable=True),   # 14
        StructField("validation_status",        StringType(),            nullable=True),   # 15
        StructField("rejected_mechanism_raw",   StringType(),            nullable=True),   # 16
        StructField("retrieval_story_ids",      ArrayType(StringType()), nullable=True),   # 17
        # NULL until TRD-5
        StructField("market_vol_at_publish",    DoubleType(),            nullable=True),   # 18
        StructField("market_trend_at_publish",  StringType(),            nullable=True),   # 19
        StructField("dollar_bar_close",         DecimalType(18, 8),      nullable=True),   # 20
        # V3 additions
        StructField("extraction_path",          StringType(),            nullable=True),   # 21
        StructField("evidence_relevance_score", DoubleType(),            nullable=True),   # 22
        StructField("confounding_cluster_id",   StringType(),            nullable=True),   # 23
    ]
)
```

- [ ] **Step 2.4: Run to confirm CAUSAL_FEATURES_SCHEMA tests pass**

```bash
uv run pytest tests/agents/test_schemas.py::TestCausalFeaturesSchema -v
```

Expected: 10 passed

- [ ] **Step 2.5: Commit**

```bash
git add src/trading_databricks/agents/schemas.py tests/agents/__init__.py tests/agents/test_schemas.py
git commit -m "TRD-23 feat: add 23-field CAUSAL_FEATURES_SCHEMA"
```

---

## Task 3: Remaining Four Schemas

- [ ] **Step 3.1: Add tests for remaining schemas in test_schemas.py**

Append to `tests/agents/test_schemas.py`:

```python
class TestTreatmentTaxonomySchema:
    def test_field_count(self):
        assert len(TREATMENT_TAXONOMY_SCHEMA.fields) == 4

    def test_field_names_in_order(self):
        names = [f.name for f in TREATMENT_TAXONOMY_SCHEMA.fields]
        assert names == ["l1_category", "l2_subcategory", "description", "example"]

    def test_all_fields_not_nullable(self):
        assert all(not f.nullable for f in TREATMENT_TAXONOMY_SCHEMA.fields)

    def test_all_fields_string_type(self):
        assert all(isinstance(f.dataType, StringType) for f in TREATMENT_TAXONOMY_SCHEMA.fields)


class TestMechanismWhitelistSchema:
    def test_field_count(self):
        assert len(MECHANISM_WHITELIST_SCHEMA.fields) == 4

    def test_field_names_in_order(self):
        names = [f.name for f in MECHANISM_WHITELIST_SCHEMA.fields]
        assert names == ["treatment_l1", "mechanism", "valid_directions", "description"]

    def test_valid_directions_is_array_of_string_not_nullable(self):
        field = next(f for f in MECHANISM_WHITELIST_SCHEMA.fields if f.name == "valid_directions")
        assert isinstance(field.dataType, ArrayType)
        assert isinstance(field.dataType.elementType, StringType)
        assert field.nullable is False

    def test_scalar_fields_not_nullable(self):
        scalars = [f for f in MECHANISM_WHITELIST_SCHEMA.fields if f.name != "valid_directions"]
        assert all(not f.nullable for f in scalars)


class TestDagnessAuditSchema:
    def test_field_count(self):
        assert len(DAGNESS_AUDIT_SCHEMA.fields) == 8

    def test_dagness_score_is_double_not_nullable(self):
        field = next(f for f in DAGNESS_AUDIT_SCHEMA.fields if f.name == "dagness_score")
        assert isinstance(field.dataType, DoubleType)
        assert field.nullable is False

    def test_is_dag_is_boolean_not_nullable(self):
        field = next(f for f in DAGNESS_AUDIT_SCHEMA.fields if f.name == "is_dag")
        assert isinstance(field.dataType, BooleanType)
        assert field.nullable is False

    def test_top_cycle_contributors_is_string_nullable(self):
        field = next(
            f for f in DAGNESS_AUDIT_SCHEMA.fields if f.name == "top_cycle_contributors"
        )
        assert isinstance(field.dataType, StringType)
        assert field.nullable is True

    def test_flagged_story_ids_is_array_nullable(self):
        field = next(f for f in DAGNESS_AUDIT_SCHEMA.fields if f.name == "flagged_story_ids")
        assert isinstance(field.dataType, ArrayType)
        assert field.nullable is True


class TestConfoundingAuditSchema:
    def test_field_count(self):
        assert len(CONFOUNDING_AUDIT_SCHEMA.fields) == 10

    def test_treatment_classes_is_array_not_nullable(self):
        field = next(
            f for f in CONFOUNDING_AUDIT_SCHEMA.fields if f.name == "treatment_classes"
        )
        assert isinstance(field.dataType, ArrayType)
        assert field.nullable is False

    def test_story_ids_is_array_not_nullable(self):
        field = next(f for f in CONFOUNDING_AUDIT_SCHEMA.fields if f.name == "story_ids")
        assert isinstance(field.dataType, ArrayType)
        assert field.nullable is False
```

Also add `BooleanType` to the import block at the top of `test_schemas.py`:

```python
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DecimalType,
    DoubleType,
    StringType,
)
```

- [ ] **Step 3.2: Run to confirm new tests fail**

```bash
uv run pytest tests/agents/test_schemas.py -v -k "not TestCausalFeatures"
```

Expected: `ImportError` (remaining schemas not yet defined)

- [ ] **Step 3.3: Add remaining schemas to schemas.py**

Append to `src/trading_databricks/agents/schemas.py`:

```python

# ── gold.treatment_taxonomy (34 rows) ─────────────────────────────────────────
# PK: (l1_category, l2_subcategory)
TREATMENT_TAXONOMY_SCHEMA = StructType(
    [
        StructField("l1_category",    StringType(), nullable=False),
        StructField("l2_subcategory", StringType(), nullable=False),
        StructField("description",    StringType(), nullable=False),
        StructField("example",        StringType(), nullable=False),
    ]
)

# ── gold.mechanism_whitelist (40 entries) ─────────────────────────────────────
# PK: (treatment_l1, mechanism)
# valid_directions ⊆ {"bullish", "bearish", "volatile"}
MECHANISM_WHITELIST_SCHEMA = StructType(
    [
        StructField("treatment_l1",     StringType(),            nullable=False),
        StructField("mechanism",        StringType(),            nullable=False),
        StructField("valid_directions", ArrayType(StringType()), nullable=False),
        StructField("description",      StringType(),            nullable=False),
    ]
)

# ── gold.dagness_audit_log ────────────────────────────────────────────────────
# Output of nightly NOTEARS acyclicity audit (Module C)
# dagness_score = 0.0 iff aggregate knowledge graph is a DAG
DAGNESS_AUDIT_SCHEMA = StructType(
    [
        StructField("audit_date",             DateType(),              nullable=False),
        StructField("total_accepted_chains",  IntegerType(),           nullable=False),
        StructField("num_nodes",              IntegerType(),           nullable=False),
        StructField("num_edges",              IntegerType(),           nullable=False),
        StructField("dagness_score",          DoubleType(),            nullable=False),
        StructField("is_dag",                 BooleanType(),           nullable=False),
        StructField("top_cycle_contributors", StringType(),            nullable=True),
        StructField("flagged_story_ids",      ArrayType(StringType()), nullable=True),
    ]
)

# ── gold.confounding_audit_log ────────────────────────────────────────────────
# Output of FCI-inspired temporal confounding detector (Module E)
CONFOUNDING_AUDIT_SCHEMA = StructType(
    [
        StructField("audit_date",          DateType(),              nullable=False),
        StructField("cluster_start",       TimestampType(),         nullable=False),
        StructField("cluster_end",         TimestampType(),         nullable=False),
        StructField("asset",               StringType(),            nullable=False),
        StructField("expected_direction",  StringType(),            nullable=False),
        StructField("distinct_treatments", IntegerType(),           nullable=False),
        StructField("article_count",       IntegerType(),           nullable=False),
        StructField("treatment_classes",   ArrayType(StringType()), nullable=False),
        StructField("story_ids",           ArrayType(StringType()), nullable=False),
        StructField("confounding_flag",    StringType(),            nullable=False),
    ]
)
```

- [ ] **Step 3.4: Run all schema tests**

```bash
uv run pytest tests/agents/test_schemas.py -v
```

Expected: All tests pass (10 from Task 2 + new tests)

- [ ] **Step 3.5: Commit**

```bash
git add src/trading_databricks/agents/schemas.py tests/agents/test_schemas.py
git commit -m "TRD-23 feat: add TREATMENT_TAXONOMY, MECHANISM_WHITELIST, DAGNESS_AUDIT, CONFOUNDING_AUDIT schemas"
```

---

## Task 4: Treatment Taxonomy Seed Data

- [ ] **Step 4.1: Write failing tests for seed data in test_taxonomy.py**

Write `tests/agents/test_taxonomy.py`:

```python
"""Tests for MAS taxonomy seed data integrity.

These tests run without a Spark session — they validate the
Python-level seed lists before they are written to Delta.
"""
from __future__ import annotations

from collections import Counter

import pytest

from trading_databricks.agents.taxonomy import (
    MECHANISM_WHITELIST_SEED,
    TREATMENT_TAXONOMY_SEED,
)

_VALID_DIRECTIONS = {"bullish", "bearish", "volatile"}
_EXPECTED_L1S = {
    "Regulation",
    "Security",
    "Adoption",
    "Macro",
    "Exchange",
    "Supply",
    "DeFi",
    "Sentiment",
}


class TestTreatmentTaxonomySeed:
    def test_row_count(self):
        assert len(TREATMENT_TAXONOMY_SEED) == 34

    def test_required_fields_present(self):
        required = {"l1_category", "l2_subcategory", "description", "example"}
        for row in TREATMENT_TAXONOMY_SEED:
            assert required.issubset(row.keys()), f"Missing fields in: {row}"

    def test_pk_unique(self):
        pks = [(r["l1_category"], r["l2_subcategory"]) for r in TREATMENT_TAXONOMY_SEED]
        assert len(pks) == len(set(pks)), "Duplicate (l1_category, l2_subcategory) pairs"

    def test_all_eight_l1_categories_present(self):
        actual = {r["l1_category"] for r in TREATMENT_TAXONOMY_SEED}
        assert actual == _EXPECTED_L1S

    def test_l1_row_counts(self):
        counts = Counter(r["l1_category"] for r in TREATMENT_TAXONOMY_SEED)
        assert counts["Regulation"] == 5
        assert counts["Macro"] == 5
        assert counts["Security"] == 4
        assert counts["Adoption"] == 4
        assert counts["Exchange"] == 4
        assert counts["Supply"] == 4
        assert counts["DeFi"] == 4
        assert counts["Sentiment"] == 4

    def test_no_empty_string_values(self):
        for row in TREATMENT_TAXONOMY_SEED:
            for key, val in row.items():
                assert isinstance(val, str) and val.strip(), (
                    f"Empty or non-string value for '{key}' in row {row}"
                )


class TestMechanismWhitelistSeed:
    def test_row_count(self):
        assert len(MECHANISM_WHITELIST_SEED) == 40

    def test_required_fields_present(self):
        required = {"treatment_l1", "mechanism", "valid_directions", "description"}
        for row in MECHANISM_WHITELIST_SEED:
            assert required.issubset(row.keys()), f"Missing fields in: {row}"

    def test_pk_unique(self):
        pks = [(r["treatment_l1"], r["mechanism"]) for r in MECHANISM_WHITELIST_SEED]
        assert len(pks) == len(set(pks)), "Duplicate (treatment_l1, mechanism) pairs"

    def test_five_entries_per_l1(self):
        counts = Counter(r["treatment_l1"] for r in MECHANISM_WHITELIST_SEED)
        for l1, count in counts.items():
            assert count == 5, f"{l1} has {count} entries, expected 5"

    def test_valid_directions_non_empty(self):
        for row in MECHANISM_WHITELIST_SEED:
            assert len(row["valid_directions"]) > 0, (
                f"Empty valid_directions for ({row['treatment_l1']}, {row['mechanism']})"
            )

    def test_valid_directions_subset_of_allowed(self):
        for row in MECHANISM_WHITELIST_SEED:
            for d in row["valid_directions"]:
                assert d in _VALID_DIRECTIONS, (
                    f"Invalid direction '{d}' for ({row['treatment_l1']}, {row['mechanism']})"
                )

    def test_l1_alignment_with_taxonomy(self):
        taxonomy_l1s = {r["l1_category"] for r in TREATMENT_TAXONOMY_SEED}
        whitelist_l1s = {r["treatment_l1"] for r in MECHANISM_WHITELIST_SEED}
        assert taxonomy_l1s == whitelist_l1s, (
            f"L1 mismatch — taxonomy: {taxonomy_l1s}, whitelist: {whitelist_l1s}"
        )

    def test_description_non_empty(self):
        for row in MECHANISM_WHITELIST_SEED:
            assert row["description"].strip(), (
                f"Empty description for ({row['treatment_l1']}, {row['mechanism']})"
            )
```

- [ ] **Step 4.2: Run to confirm ImportError**

```bash
uv run pytest tests/agents/test_taxonomy.py -v
```

Expected: `ModuleNotFoundError: No module named 'trading_databricks.agents.taxonomy'`

- [ ] **Step 4.3: Create taxonomy.py with seed data**

Create `src/trading_databricks/agents/taxonomy.py`:

```python
"""Treatment taxonomy and mechanism whitelist — seed data and Delta accessors.

Seed lists are the single source of truth for the 34-row treatment taxonomy
and 40-entry mechanism whitelist. Written to Delta once per environment via
seed_taxonomy(). Read at agent runtime via load_taxonomy() / load_whitelist().
"""
from __future__ import annotations

from trading_databricks.agents.schemas import (
    MECHANISM_WHITELIST_SCHEMA,
    TREATMENT_TAXONOMY_SCHEMA,
)

# ── Treatment Taxonomy Seed (34 rows) ─────────────────────────────────────────
# 8 L1 categories: Regulation(5), Macro(5), Security(4), Adoption(4),
#                  Exchange(4), Supply(4), DeFi(4), Sentiment(4) = 34 total
TREATMENT_TAXONOMY_SEED: list[dict] = [
    # Sentiment (4)
    {
        "l1_category": "Sentiment",
        "l2_subcategory": "whale_movement",
        "description": "Massive dormant cryptocurrency holdings are transferred on-chain, signaling potential imminent market activity.",
        "example": "Dormant whale address moves $250M in Bitcoin to centralized exchange",
    },
    {
        "l1_category": "Sentiment",
        "l2_subcategory": "influencer_signal",
        "description": "High-profile individuals make public statements or social media posts that trigger immediate speculative trading frenzies.",
        "example": "Elon Musk tweets support for Dogecoin, triggering 15% price surge within minutes",
    },
    {
        "l1_category": "Sentiment",
        "l2_subcategory": "fund_flow",
        "description": "Asset managers report the net inflows and outflows for publicly traded crypto investment products.",
        "example": "Spot Bitcoin ETFs see record $500M weekly inflow, led by institutional allocators",
    },
    {
        "l1_category": "Sentiment",
        "l2_subcategory": "options_expiry",
        "description": "The settlement of massive derivatives contracts forces market makers to hedge, anchoring spot prices to max pain levels.",
        "example": "Bitcoin braces for volatility as $14 billion in options contracts expire today",
    },
    # DeFi (4)
    {
        "l1_category": "DeFi",
        "l2_subcategory": "tvl_shift",
        "description": "Capital rapidly migrates between decentralized finance sectors, altering the Total Value Locked (TVL) landscape.",
        "example": "Capital rotates from complex restaking into tokenized Real-World Assets as DeFi matures",
    },
    {
        "l1_category": "DeFi",
        "l2_subcategory": "protocol_upgrade",
        "description": "A decentralized protocol successfully executes a governance vote or technical hard fork to improve economic mechanics.",
        "example": "Uniswap activates fee switch, distributing protocol revenue to token holders",
    },
    {
        "l1_category": "DeFi",
        "l2_subcategory": "stablecoin_depeg",
        "description": "A fiat-pegged digital asset loses parity with its underlying reserve currency due to market panic or collateral failure.",
        "example": "USDC drops to 89 cents amid panic over exposure to collapsed Silicon Valley Bank",
    },
    {
        "l1_category": "DeFi",
        "l2_subcategory": "yield_event",
        "description": "Structural changes in on-chain borrowing/lending rates alter the risk-free return profile for decentralized capital.",
        "example": "Fixed-rate lending platform Yield Protocol winds down operations amid declining DeFi demand",
    },
    # Supply (4)
    {
        "l1_category": "Supply",
        "l2_subcategory": "halving",
        "description": "Bitcoin's block reward is cut in half approximately every four years, permanently reducing new BTC issuance rate.",
        "example": "Bitcoin halving complete: block reward drops from 6.25 to 3.125 BTC",
    },
    {
        "l1_category": "Supply",
        "l2_subcategory": "unlock_distribution",
        "description": "Previously locked tokens are released, or bankruptcy estates distribute recovered assets to creditors.",
        "example": "Mt. Gox estate transfers 10,000 BTC to exchanges, signaling impending creditor distributions",
    },
    {
        "l1_category": "Supply",
        "l2_subcategory": "mining_difficulty",
        "description": "Network mining difficulty adjusts to accommodate changes in total computational hash rate, impacting profitability.",
        "example": "Bitcoin mining difficulty hits new all-time high of 150 trillion, squeezing miner margins",
    },
    {
        "l1_category": "Supply",
        "l2_subcategory": "burn_event",
        "description": "A protocol permanently removes a portion of its token supply from circulation, increasing asset scarcity.",
        "example": "Ethereum network burns $1.1B in transaction fees following successful EIP-1559 implementation",
    },
    # Exchange (4)
    {
        "l1_category": "Exchange",
        "l2_subcategory": "listing",
        "description": "A major exchange announces support for a new token, instantly expanding its liquidity and retail accessibility.",
        "example": "Binance announces spot listing for emerging Layer-2 token, driving immediate price surge",
    },
    {
        "l1_category": "Exchange",
        "l2_subcategory": "delisting",
        "description": "Trading platforms permanently remove a token due to low liquidity, compliance issues, or network instability.",
        "example": "OKX announces delisting of six altcoin pairs to maintain a robust trading environment",
    },
    {
        "l1_category": "Exchange",
        "l2_subcategory": "insolvency",
        "description": "An exchange or lending platform files for bankruptcy or halts withdrawals due to an inability to cover liabilities.",
        "example": "Top crypto lending platform halts all withdrawals amid rumors of critical liquidity crisis",
    },
    {
        "l1_category": "Exchange",
        "l2_subcategory": "reserve_proof",
        "description": "Exchanges publish cryptographic Merkle tree audits to verify customer deposits are fully backed by on-chain assets.",
        "example": "Kraken publishes updated Proof of Reserves, verifying 100% asset backing for all customer funds",
    },
    # Macro (5)
    {
        "l1_category": "Macro",
        "l2_subcategory": "rate_decision",
        "description": "Central banks alter benchmark interest rates, directly impacting the cost of capital for risk assets.",
        "example": "Fed cuts interest rates by 50 basis points, signaling a shift toward monetary easing",
    },
    {
        "l1_category": "Macro",
        "l2_subcategory": "inflation_data",
        "description": "Government agencies release consumer or producer price indices, altering expectations for fiat debasement.",
        "example": "US CPI inflation cools to 2.8%, missing analyst expectations and driving risk-on sentiment",
    },
    {
        "l1_category": "Macro",
        "l2_subcategory": "employment_data",
        "description": "Labor market statistics are published, serving as leading economic indicators for central bank policy.",
        "example": "US adds 130,000 jobs in January, outperforming expectations and dampening rate cut hopes",
    },
    {
        "l1_category": "Macro",
        "l2_subcategory": "gdp",
        "description": "National economic output data is revised or released, indicating macroeconomic expansion or contraction.",
        "example": "US GDP growth surges past estimates in Q3, bolstering confidence in economic resilience",
    },
    {
        "l1_category": "Macro",
        "l2_subcategory": "geopolitical",
        "description": "Significant international conflicts, trade wars, or tariff implementations generate macroeconomic uncertainty.",
        "example": "Global markets reel as new US tariffs trigger massive Bitcoin liquidation cascade",
    },
    # Adoption (4)
    {
        "l1_category": "Adoption",
        "l2_subcategory": "institutional_buy",
        "description": "Large asset managers or hedge funds publicly disclose significant allocations to BTC or ETH.",
        "example": "Major Wall Street asset manager announces 5% Bitcoin allocation for flagship macro fund",
    },
    {
        "l1_category": "Adoption",
        "l2_subcategory": "corporate_treasury",
        "description": "Public or private corporations add digital assets to their balance sheets as a primary treasury reserve.",
        "example": "MicroStrategy acquires an additional 50,000 BTC, expanding total corporate treasury holdings",
    },
    {
        "l1_category": "Adoption",
        "l2_subcategory": "payment_integration",
        "description": "Global payment processors or merchant networks enable crypto settlement for commercial transactions.",
        "example": "Visa expands stablecoin settlement capabilities to the Solana network for global merchants",
    },
    {
        "l1_category": "Adoption",
        "l2_subcategory": "country_adoption",
        "description": "Sovereign nations formally adopt cryptocurrency as legal tender or establish state-backed reserve funds.",
        "example": "Central African Republic follows El Salvador in adopting Bitcoin as official legal tender",
    },
    # Security (4)
    {
        "l1_category": "Security",
        "l2_subcategory": "exchange_hack",
        "description": "A centralised crypto exchange is exploited, resulting in theft of user funds and loss of market confidence.",
        "example": "Bybit confirms $1.4B hack; withdrawals suspended as investigation begins",
    },
    {
        "l1_category": "Security",
        "l2_subcategory": "protocol_exploit",
        "description": "Decentralized smart contracts or cross-chain bridges are compromised due to code vulnerabilities.",
        "example": "Drift Protocol drained of $285M in highly coordinated smart contract exploit",
    },
    {
        "l1_category": "Security",
        "l2_subcategory": "rug_pull",
        "description": "Project founders abruptly abandon a protocol, draining liquidity pools and leaving investors with worthless tokens.",
        "example": "Vault of Gems founders vanish with $22M in classic NFT rug pull scheme",
    },
    {
        "l1_category": "Security",
        "l2_subcategory": "wallet_compromise",
        "description": "Private keys for institutional hot/cold wallets are stolen via social engineering or physical coercion.",
        "example": "Major market maker loses $50M as hot wallet private keys are compromised",
    },
    # Regulation (5)
    {
        "l1_category": "Regulation",
        "l2_subcategory": "etf_ruling",
        "description": "A regulatory authority approves, rejects, or delays a Bitcoin or Ethereum spot/futures ETF application.",
        "example": "SEC approves first spot Bitcoin ETF, opening market to institutional investors",
    },
    {
        "l1_category": "Regulation",
        "l2_subcategory": "legislation",
        "description": "Legislative bodies pass or propose laws establishing statutory frameworks for digital assets.",
        "example": "Congress passes GENIUS Act, establishing federal regulatory framework for payment stablecoins",
    },
    {
        "l1_category": "Regulation",
        "l2_subcategory": "enforcement_action",
        "description": "Regulators bring civil or criminal charges against entities for securities, commodities, or AML violations.",
        "example": "SEC issues Wells notice to major US exchange over alleged unregistered securities offerings",
    },
    {
        "l1_category": "Regulation",
        "l2_subcategory": "cbdc_policy",
        "description": "Central banks announce the development, piloting, or cancellation of sovereign digital currencies.",
        "example": "Federal Reserve publishes whitepaper exploring privacy frameworks for potential US CBDC",
    },
    {
        "l1_category": "Regulation",
        "l2_subcategory": "tax_policy",
        "description": "Tax authorities issue new guidelines or enforcement mechanisms for reporting crypto capital gains.",
        "example": "Treasury Department proposes new tax reporting rules for digital asset brokers to close compliance gap",
    },
]

# ── Mechanism Whitelist Seed (40 entries) ─────────────────────────────────────
# 8 L1 categories × 5 mechanisms = 40 entries
# valid_directions ⊆ {"bullish", "bearish", "volatile"}
# Cross-L1 duplicates (sell_pressure, trust_erosion) are intentional.
MECHANISM_WHITELIST_SEED: list[dict] = [
    # Regulation (5)
    {
        "treatment_l1": "Regulation",
        "mechanism": "regulatory_clarity",
        "valid_directions": ["bullish"],
        "description": "Removal of regulatory ambiguity reduces uncertainty premium, attracting institutional and retail capital inflows.",
    },
    {
        "treatment_l1": "Regulation",
        "mechanism": "regulatory_restriction",
        "valid_directions": ["bearish"],
        "description": "Prohibitions or strict licensing requirements sever capital access and force structural reductions in market participation.",
    },
    {
        "treatment_l1": "Regulation",
        "mechanism": "institutional_access",
        "valid_directions": ["bullish"],
        "description": "Legal frameworks enabling traditional finance entities to custody or trade crypto drastically expand the available capital base.",
    },
    {
        "treatment_l1": "Regulation",
        "mechanism": "compliance_burden",
        "valid_directions": ["bearish"],
        "description": "Onerous tax or AML reporting requirements increase operational friction, reducing trading velocity and market maker profitability.",
    },
    {
        "treatment_l1": "Regulation",
        "mechanism": "market_legitimization",
        "valid_directions": ["bullish"],
        "description": "Sovereign or tier-one regulatory endorsement lowers the psychological risk profile of digital assets, inviting risk-averse capital.",
    },
    # Security (5)
    {
        "treatment_l1": "Security",
        "mechanism": "sell_pressure",
        "valid_directions": ["bearish"],
        "description": "Hackers liquidating stolen digital assets for fiat create immediate, overwhelming supply gluts on the open market.",
    },
    {
        "treatment_l1": "Security",
        "mechanism": "trust_erosion",
        "valid_directions": ["bearish"],
        "description": "A security breach destroys user confidence in the affected platform or asset class, triggering withdrawals and sell-offs.",
    },
    {
        "treatment_l1": "Security",
        "mechanism": "market_exit",
        "valid_directions": ["bearish"],
        "description": "Severe hacks render platforms insolvent, leading to permanent operational shutdown and complete loss of associated asset value.",
    },
    {
        "treatment_l1": "Security",
        "mechanism": "contagion_spread",
        "valid_directions": ["bearish"],
        "description": "A vulnerability in one DeFi primitive cascades across interconnected lending protocols, threatening systemic collateral health.",
    },
    {
        "treatment_l1": "Security",
        "mechanism": "user_withdrawal",
        "valid_directions": ["bearish"],
        "description": "Fear of exposure to a breach triggers a localized bank run, depleting reserve liquidity and halting operational capacity.",
    },
    # Adoption (5)
    {
        "treatment_l1": "Adoption",
        "mechanism": "demand_increase",
        "valid_directions": ["bullish"],
        "description": "Direct purchasing by new demographic cohorts or commercial entities absorbs circulating supply, elevating the clearing price.",
    },
    {
        "treatment_l1": "Adoption",
        "mechanism": "institutional_inflow",
        "valid_directions": ["bullish"],
        "description": "Sustained capital allocation from macro funds and ETFs creates a robust price floor and drastically improves market depth.",
    },
    {
        "treatment_l1": "Adoption",
        "mechanism": "network_expansion",
        "valid_directions": ["bullish"],
        "description": "Growth in active wallet addresses and integrations validates Metcalfe's Law, fundamentally driving protocol valuation.",
    },
    {
        "treatment_l1": "Adoption",
        "mechanism": "utility_unlock",
        "valid_directions": ["bullish"],
        "description": "Integration into legacy payment networks transitions a token from a speculative instrument to a productive commercial currency.",
    },
    {
        "treatment_l1": "Adoption",
        "mechanism": "strategic_accumulation",
        "valid_directions": ["bullish"],
        "description": "Corporate or sovereign treasuries aggressively hoarding assets permanently reduces free float, magnifying future demand shocks.",
    },
    # Macro (5)
    {
        "treatment_l1": "Macro",
        "mechanism": "risk_on",
        "valid_directions": ["bullish"],
        "description": "Favorable macroeconomic conditions encourage capital to seek higher yields out on the risk curve, benefiting high-beta assets.",
    },
    {
        "treatment_l1": "Macro",
        "mechanism": "risk_off",
        "valid_directions": ["bearish"],
        "description": "Economic uncertainty or rising yields drive massive capital flight from speculative networks into traditional safe havens.",
    },
    {
        "treatment_l1": "Macro",
        "mechanism": "liquidity_expansion",
        "valid_directions": ["bullish"],
        "description": "Central bank quantitative easing injects excess fiat liquidity into the global economy, inherently inflating finite digital assets.",
    },
    {
        "treatment_l1": "Macro",
        "mechanism": "liquidity_contraction",
        "valid_directions": ["bearish"],
        "description": "Quantitative tightening drains speculative fiat liquidity from the financial system, starving crypto markets of upward momentum.",
    },
    {
        "treatment_l1": "Macro",
        "mechanism": "inflation_hedge",
        "valid_directions": ["bullish"],
        "description": "Investors rotate capital into Bitcoin as a store-of-value hedge when fiat purchasing power erodes due to persistent inflation.",
    },
    # Exchange (5)
    {
        "treatment_l1": "Exchange",
        "mechanism": "liquidity_increase",
        "valid_directions": ["bullish"],
        "description": "Tier-1 exchange listings drastically lower trading friction and grant exposure to massive new retail and institutional pools.",
    },
    {
        "treatment_l1": "Exchange",
        "mechanism": "trust_erosion",
        "valid_directions": ["bearish"],
        "description": "Delistings or rumors of exchange instability degrade faith in centralized market structure, prompting widespread capital flight.",
    },
    {
        "treatment_l1": "Exchange",
        "mechanism": "market_access_expansion",
        "valid_directions": ["bullish"],
        "description": "Securing operational licenses in new jurisdictions unlocks previously inaccessible geographic sectors of retail liquidity.",
    },
    {
        "treatment_l1": "Exchange",
        "mechanism": "trading_suspension",
        "valid_directions": ["bearish"],
        "description": "Halting withdrawals traps user capital, instantly signaling critical insolvency and sparking panic selling on secondary markets.",
    },
    {
        "treatment_l1": "Exchange",
        "mechanism": "counterparty_risk",
        "valid_directions": ["bearish"],
        "description": "Fears of fractional reserves or commingled funds destroy the risk-reward ratio of centralized custody, accelerating insolvency.",
    },
    # Supply (5)
    {
        "treatment_l1": "Supply",
        "mechanism": "sell_pressure",
        "valid_directions": ["bearish"],
        "description": "The mass distribution of recovered bankruptcy assets or token unlocks floods the market with highly liquid selling volume.",
    },
    {
        "treatment_l1": "Supply",
        "mechanism": "supply_reduction",
        "valid_directions": ["bullish"],
        "description": "Programmatic halvings slash the structural daily issuance of new coins, starving the market of required seller liquidity.",
    },
    {
        "treatment_l1": "Supply",
        "mechanism": "miner_capitulation",
        "valid_directions": ["bearish", "volatile"],
        "description": "Operating losses force miners to aggressively dump BTC treasuries, creating sharp drawdowns that historically mark cycle bottoms.",
    },
    {
        "treatment_l1": "Supply",
        "mechanism": "scarcity_signal",
        "valid_directions": ["bullish"],
        "description": "Deflationary fee-burn mechanics programmatically destroy circulating supply, reinforcing the psychological hard-money narrative.",
    },
    {
        "treatment_l1": "Supply",
        "mechanism": "vesting_unlock",
        "valid_directions": ["bearish", "volatile"],
        "description": "The scheduled release of venture capital tokens drastically inflates the float, inviting aggressive front-run short selling.",
    },
    # DeFi (5)
    {
        "treatment_l1": "DeFi",
        "mechanism": "contagion_risk",
        "valid_directions": ["bearish"],
        "description": "The failure of a foundational stablecoin or primitive triggers cascading debt liquidations across dependent DeFi protocols.",
    },
    {
        "treatment_l1": "DeFi",
        "mechanism": "yield_migration",
        "valid_directions": ["volatile"],
        "description": "Mercenary capital rapidly rotates between chains chasing high APYs, causing severe localized liquidity crunches and price swings.",
    },
    {
        "treatment_l1": "DeFi",
        "mechanism": "liquidity_drain",
        "valid_directions": ["bearish"],
        "description": "The rapid exit of TVL from an automated market maker creates extreme slippage and mathematically ensures token death spirals.",
    },
    {
        "treatment_l1": "DeFi",
        "mechanism": "protocol_confidence",
        "valid_directions": ["bullish"],
        "description": "The implementation of institutional-grade risk frameworks and successful technical upgrades attracts durable, sticky capital.",
    },
    {
        "treatment_l1": "DeFi",
        "mechanism": "stablecoin_risk",
        "valid_directions": ["bearish", "volatile"],
        "description": "Depegging fears force users to dump stablecoins for volatile assets, temporarily pumping prices before a structural macro crash.",
    },
    # Sentiment (5)
    {
        "treatment_l1": "Sentiment",
        "mechanism": "momentum_signal",
        "valid_directions": ["volatile"],
        "description": "Trend-following algorithmic traders exacerbate directional moves, violently extending both speculative rallies and panic crashes.",
    },
    {
        "treatment_l1": "Sentiment",
        "mechanism": "liquidation_risk",
        "valid_directions": ["bearish", "volatile"],
        "description": "High systemic leverage ensures that minor price drops trigger cascading margin calls, resulting in violent downside wicks.",
    },
    {
        "treatment_l1": "Sentiment",
        "mechanism": "fear_signal",
        "valid_directions": ["bearish"],
        "description": "Overwhelmingly negative sentiment indices and media cycles prompt retail capitulation and defensive institutional de-risking.",
    },
    {
        "treatment_l1": "Sentiment",
        "mechanism": "greed_signal",
        "valid_directions": ["bullish", "volatile"],
        "description": "Euphoric retail speculation drives asset bubbles with exponential price appreciation, inevitably preceding sharp mean-reversions.",
    },
    {
        "treatment_l1": "Sentiment",
        "mechanism": "narrative_shift",
        "valid_directions": ["volatile"],
        "description": "Social velocity unpredictably rotates capital between meta-themes, draining established sectors to fuel localized asset frenzies.",
    },
]
```

- [ ] **Step 4.4: Run taxonomy integrity tests**

```bash
uv run pytest tests/agents/test_taxonomy.py -v
```

Expected: All tests pass

- [ ] **Step 4.5: Commit**

```bash
git add src/trading_databricks/agents/taxonomy.py tests/agents/test_taxonomy.py
git commit -m "TRD-23 feat: add treatment taxonomy (34 rows) and mechanism whitelist (40 entries) seed data"
```

---

## Task 5: Taxonomy Functions

- [ ] **Step 5.1: Add seed/load functions to taxonomy.py**

Append to the bottom of `src/trading_databricks/agents/taxonomy.py`:

```python


def seed_taxonomy(spark, *, catalog: str, schema: str) -> None:
    """Write treatment_taxonomy and mechanism_whitelist tables to Delta.

    Overwrites existing tables and schema. Run once per environment.
    Requires a Databricks Connect or cluster Spark session.

    Args:
        spark: Active SparkSession.
        catalog: Unity Catalog catalog name (e.g. "smart-trading-dev").
        schema: Schema name (e.g. "gold").
    """
    taxonomy_df = spark.createDataFrame(TREATMENT_TAXONOMY_SEED, schema=TREATMENT_TAXONOMY_SCHEMA)
    (
        taxonomy_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"`{catalog}`.`{schema}`.treatment_taxonomy")
    )

    whitelist_df = spark.createDataFrame(
        MECHANISM_WHITELIST_SEED, schema=MECHANISM_WHITELIST_SCHEMA
    )
    (
        whitelist_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"`{catalog}`.`{schema}`.mechanism_whitelist")
    )


def load_taxonomy(spark, *, catalog: str, schema: str) -> dict[str, list[str]]:
    """Load treatment taxonomy from Delta as a Python dict.

    Args:
        spark: Active SparkSession.
        catalog: Unity Catalog catalog name.
        schema: Schema name.

    Returns:
        Mapping of l1_category → [l2_subcategory, ...] (insertion order preserved).
    """
    rows = spark.table(f"`{catalog}`.`{schema}`.treatment_taxonomy").collect()
    result: dict[str, list[str]] = {}
    for row in rows:
        result.setdefault(row.l1_category, []).append(row.l2_subcategory)
    return result


def load_whitelist(spark, *, catalog: str, schema: str) -> dict[tuple[str, str], list[str]]:
    """Load mechanism whitelist from Delta as a Python dict.

    Args:
        spark: Active SparkSession.
        catalog: Unity Catalog catalog name.
        schema: Schema name.

    Returns:
        Mapping of (treatment_l1, mechanism) → [valid_directions].
    """
    rows = spark.table(f"`{catalog}`.`{schema}`.mechanism_whitelist").collect()
    return {
        (row.treatment_l1, row.mechanism): list(row.valid_directions) for row in rows
    }
```

- [ ] **Step 5.2: Run full test suite to confirm no regressions**

```bash
uv run pytest tests/agents/ -v
```

Expected: All tests pass (taxonomy + schema tests)

- [ ] **Step 5.3: Commit**

```bash
git add src/trading_databricks/agents/taxonomy.py
git commit -m "TRD-23 feat: add seed_taxonomy, load_taxonomy, load_whitelist functions"
```

---

## Task 6: Seed Script

- [ ] **Step 6.1: Create scripts/seed_taxonomy.py**

```python
"""One-shot script to seed treatment taxonomy and mechanism whitelist Delta tables.

Run once per environment after deploying to a new catalog. The script
verifies the written row counts and exits non-zero on mismatch.

Usage:
    uv run python scripts/seed_taxonomy.py
    uv run python scripts/seed_taxonomy.py --catalog trading-signal-gen-prod --schema gold
"""
from __future__ import annotations

import argparse
import sys

from databricks.connect import DatabricksSession

from trading_databricks.agents.taxonomy import seed_taxonomy


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed MAS taxonomy tables to Delta.")
    parser.add_argument(
        "--catalog",
        default="smart-trading-dev",
        help="Unity Catalog catalog name (default: smart-trading-dev)",
    )
    parser.add_argument(
        "--schema",
        default="gold",
        help="Schema name (default: gold)",
    )
    args = parser.parse_args()

    print(f"Seeding {args.catalog}.{args.schema} ...")
    spark = DatabricksSession.builder.getOrCreate()
    seed_taxonomy(spark, catalog=args.catalog, schema=args.schema)

    taxonomy_count = spark.table(f"`{args.catalog}`.`{args.schema}`.treatment_taxonomy").count()
    whitelist_count = spark.table(
        f"`{args.catalog}`.`{args.schema}`.mechanism_whitelist"
    ).count()

    print(f"  treatment_taxonomy: {taxonomy_count} rows (expected 34)")
    print(f"  mechanism_whitelist: {whitelist_count} rows (expected 40)")

    if taxonomy_count != 34 or whitelist_count != 40:
        print("ERROR: Row count mismatch — seed may have failed.", file=sys.stderr)
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Commit**

```bash
git add scripts/seed_taxonomy.py
git commit -m "TRD-23 feat: add seed_taxonomy one-shot Databricks Connect script"
```

---

## Task 7: Final Validation and PR

- [ ] **Step 7.1: Run full project test suite**

```bash
uv run pytest -v
```

Expected: All existing tests pass + new tests pass. No failures.

- [ ] **Step 7.2: Check if TRD-24 has merged to main**

```bash
git fetch origin && git log origin/main --oneline -5
```

If TRD-24 is merged (you see `feat/TRD-24` or `TRD-24 feat:` commits on main), rebase:

```bash
git rebase origin/main
```

If there's a conflict on `tests/agents/__init__.py` (both branches created it as an empty file), accept either version:

```bash
git checkout --ours tests/agents/__init__.py
git add tests/agents/__init__.py
git rebase --continue
```

If TRD-24 is not yet merged, proceed — the PR can be created and will rebase cleanly once TRD-24 merges.

- [ ] **Step 7.3: Update coordination file Section 1**

Edit `docs/coordination/TRD-6-gen-ai-agent-layer.md` Section 1:

```markdown
## Section 1: Schema & Config — TRD-23, TRD-24 (Wave 1)
- **Status:** in_progress
- **Branch:** feat/TRD-23-mas-schemas-and-taxonomy
- **Modified files:**
  - src/trading_databricks/agents/schemas.py (NEW)
  - src/trading_databricks/agents/taxonomy.py (NEW)
  - scripts/seed_taxonomy.py (NEW)
  - tests/agents/__init__.py (NEW)
  - tests/agents/test_schemas.py (NEW)
  - tests/agents/test_taxonomy.py (NEW)
- **Exports (public APIs):**
  - schemas.py: CAUSAL_FEATURES_SCHEMA, TREATMENT_TAXONOMY_SCHEMA, MECHANISM_WHITELIST_SCHEMA, DAGNESS_AUDIT_SCHEMA, CONFOUNDING_AUDIT_SCHEMA
  - taxonomy.py: seed_taxonomy(), load_taxonomy(), load_whitelist(), TREATMENT_TAXONOMY_SEED, MECHANISM_WHITELIST_SEED
- **Needs from others:** None (foundation)
- **Notes:** Merges second, after TRD-24. Run seed_taxonomy.py on cluster 0331-043814-ghylcr28 post-merge.
```

- [ ] **Step 7.4: Commit coordination update**

```bash
git add docs/coordination/TRD-6-gen-ai-agent-layer.md
git commit -m "TRD-23 chore: update coordination file with in_progress status"
```

- [ ] **Step 7.5: Push and create PR**

```bash
git push -u origin feat/TRD-23-mas-schemas-and-taxonomy
```

```bash
gh pr create \
  --title "TRD-23: Add 23-field causal features schema, treatment taxonomy, and mechanism whitelist" \
  --body "$(cat <<'EOF'
## Summary

- Adds 5 `StructType` constants to `src/trading_databricks/agents/schemas.py`: `CAUSAL_FEATURES_SCHEMA` (23 fields), `TREATMENT_TAXONOMY_SCHEMA`, `MECHANISM_WHITELIST_SCHEMA`, `DAGNESS_AUDIT_SCHEMA`, `CONFOUNDING_AUDIT_SCHEMA`
- Adds `TREATMENT_TAXONOMY_SEED` (34 rows) and `MECHANISM_WHITELIST_SEED` (40 entries) with confirmed canonical data in `src/trading_databricks/agents/taxonomy.py`
- Adds `seed_taxonomy()`, `load_taxonomy()`, `load_whitelist()` functions
- Adds `scripts/seed_taxonomy.py` one-shot seeder with row count assertions

## Test plan

- [x] `uv run pytest tests/agents/test_schemas.py -v` — schema structure tests pass (no Spark session)
- [x] `uv run pytest tests/agents/test_taxonomy.py -v` — seed data integrity tests pass (no Spark session)
- [x] `uv run pytest -v` — full suite, no regressions
- [ ] Post-merge: `uv run python scripts/seed_taxonomy.py` on cluster `0331-043814-ghylcr28` — verify 34 + 40 rows in dev catalog

## Note

Merges after TRD-24 (`feat/TRD-24-mas-config-dataclasses`). Will rebase cleanly — only potential conflict is `tests/agents/__init__.py` (empty file in both branches).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
