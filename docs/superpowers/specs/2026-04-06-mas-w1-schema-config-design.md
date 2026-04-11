# MAS Wave 1 — Schema, Taxonomy & Config Design

**Date:** 2026-04-06
**Version:** 1.0
**Status:** Approved
**Tickets:** TRD-23 (Schema + Taxonomy), TRD-24 (Config Dataclasses)
**Parent:** TRD-6 (Build Gen-AI Agent Layer)

**Extends:**
- [`2026-04-04-mas-final-spec.md`](./2026-04-04-mas-final-spec.md) — schemas §8, §9, §10; gatekeeper §5.5
- [`2026-04-04-robust-trustworthy-mas-spec.md`](./2026-04-04-robust-trustworthy-mas-spec.md) — fields 21–23; router §2.3; audit schemas §4.4, §6.4
- [`2026-04-05-mas-spec.md`](./2026-04-05-mas-spec.md) — V3.1 project structure

---

## 1. Context

Wave 1 is the **foundation** for the entire MAS pipeline. All Wave 2+ agents (TRD-25, 27–30) import from the files produced here. No agent code begins until both TRD-23 and TRD-24 are merged.

The two tickets are fully non-overlapping in files and are developed in parallel in separate git worktrees:

| Ticket | Branch | Merges |
|--------|--------|--------|
| TRD-24 | `feat/TRD-24-mas-config-dataclasses` | First (pure Python, no Spark) |
| TRD-23 | `feat/TRD-23-mas-schemas-and-taxonomy` | Second (rebases on TRD-24) |

---

## 2. TRD-24 — Frozen Config Dataclasses

**File:** `src/trading_databricks/agents/config.py`

Four `@dataclass(frozen=True)` classes. All defaults match spec values exactly. `MASConfig` uses `field(default_factory=...)` for nested config objects.

### GatekeeperConfig

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `model_uri` | `str` | `"models:/news-gatekeeper/Production"` | MLflow model URI |
| `confidence_threshold` | `float` | `0.5` | sklearn classifier cutoff |
| `ner_only_mode` | `bool` | `True` | V1: NER only, no MLflow dependency |
| `text_window` | `int` | `500` | chars of body concatenated with headline |

### UncertaintyRouterConfig

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `high_similarity_threshold` | `float` | `0.85` | VS cosine → fast path |
| `low_similarity_threshold` | `float` | `0.50` | VS cosine → careful path |
| `self_consistency_k` | `int` | `3` | LLM calls on careful path |
| `min_vs_results` | `int` | `2` | minimum VS matches for fast path |
| `min_index_size` | `int` | `50` | index too small → force careful regardless |

### MechanismValidatorConfig

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `evidence_grounding_reject_threshold` | `float \| None` | `None` | None = informational only (V3); set threshold in V4 |
| `embedding_model` | `str` | `"databricks-gte-large-en"` | Same endpoint as VS index embedding |

### MASConfig

| Field | Type | Default |
|-------|------|---------|
| `gatekeeper` | `GatekeeperConfig` | `field(default_factory=GatekeeperConfig)` |
| `router` | `UncertaintyRouterConfig` | `field(default_factory=UncertaintyRouterConfig)` |
| `validator` | `MechanismValidatorConfig` | `field(default_factory=MechanismValidatorConfig)` |
| `catalog` | `str` | `"smart-trading-dev"` |
| `schema` | `str` | `"gold"` |
| `vs_index_name` | `str` | `"smart-trading-dev.03_gold.causal_chains_idx"` |
| `vs_endpoint_name` | `str` | `"mas-few-shot"` |
| `llm_endpoint` | `str` | `"databricks-claude-sonnet"` |
| `agent_model_version` | `str` | `"v1.0"` |

**Tests:** `tests/agents/test_config.py` — defaults, frozen enforcement, nested construction, override patterns. No Spark required.

---

## 3. TRD-23 — Schemas, Taxonomy & Whitelist

### 3.1 Five StructType Schemas

**File:** `src/trading_databricks/agents/schemas.py`

#### CAUSAL_FEATURES_SCHEMA (23 fields)

`gold.news_causal_features` output schema. Fields 21–23 are V3 additions (nullable, backward compatible).

| # | Field | Type | Nullable | Notes |
|---|-------|------|----------|-------|
| 1 | `story_id` | STRING | **NOT NULL** | PK, FK → silver.news_cleaned |
| 2 | `published_at` | TIMESTAMP | **NOT NULL** | |
| 3 | `is_impactful` | BOOLEAN | **NOT NULL** | Gatekeeper decision |
| 4 | `gatekeeper_confidence` | DOUBLE | nullable | 0.0/1.0 (NER mode) or [0,1] (sklearn) |
| 5 | `agent_model_version` | STRING | **NOT NULL** | MLflow lineage |
| 6 | `processed_at` | TIMESTAMP | **NOT NULL** | |
| 7 | `treatment_class` | STRING | nullable | `"L1.l2_subcategory"` |
| 8 | `trigger_event` | STRING | nullable | max 200 chars |
| 9 | `mechanism` | STRING | nullable | must exist in whitelist |
| 10 | `asset` | STRING | nullable | `BTC` \| `ETH` \| `BOTH` |
| 11 | `expected_direction` | STRING | nullable | `bullish` \| `bearish` \| `volatile` |
| 12 | `confidence` | DOUBLE | nullable | LLM self-assessed [0.0, 1.0] |
| 13 | `causal_chain_json` | STRING | nullable | full extraction JSON |
| 14 | `evidence_spans` | ARRAY\<STRING\> | nullable | 1–3 normalized substrings |
| 15 | `validation_status` | STRING | nullable | `accepted` \| `rejected_*` |
| 16 | `rejected_mechanism_raw` | STRING | nullable | LLM output before rejection |
| 17 | `retrieval_story_ids` | ARRAY\<STRING\> | nullable | VS few-shot source IDs |
| 18 | `market_vol_at_publish` | DOUBLE | nullable | NULL until TRD-5 |
| 19 | `market_trend_at_publish` | STRING | nullable | NULL until TRD-5 |
| 20 | `dollar_bar_close` | DECIMAL(18,8) | nullable | NULL until TRD-5; **not** DoubleType |
| 21 | `extraction_path` | STRING | nullable | `fast` \| `standard` \| `careful` (Module A) |
| 22 | `evidence_relevance_score` | DOUBLE | nullable | cosine similarity [0,1] (Module B) |
| 23 | `confounding_cluster_id` | STRING | nullable | UUID if confounded (Module E) |

**Nullability rule:** `is_impactful=False` → fields 7–17 and 21–23 are ALL NULL.

#### TREATMENT_TAXONOMY_SCHEMA (4 fields)

`gold.treatment_taxonomy` — 34 rows, PK: `(l1_category, l2_subcategory)`.

```python
StructType([
    StructField("l1_category",    StringType(), nullable=False),
    StructField("l2_subcategory", StringType(), nullable=False),
    StructField("description",    StringType(), nullable=False),
    StructField("example",        StringType(), nullable=False),
])
```

#### MECHANISM_WHITELIST_SCHEMA (4 fields)

`gold.mechanism_whitelist` — 40 entries, PK: `(treatment_l1, mechanism)`.

```python
StructType([
    StructField("treatment_l1",     StringType(),            nullable=False),
    StructField("mechanism",        StringType(),            nullable=False),
    StructField("valid_directions", ArrayType(StringType()), nullable=False),
    StructField("description",      StringType(),            nullable=False),
])
```

#### DAGNESS_AUDIT_SCHEMA (8 fields)

`gold.dagness_audit_log` — nightly NOTEARS acyclicity audit output.

```python
StructType([
    StructField("audit_date",              DateType(),              nullable=False),
    StructField("total_accepted_chains",   IntegerType(),           nullable=False),
    StructField("num_nodes",               IntegerType(),           nullable=False),
    StructField("num_edges",               IntegerType(),           nullable=False),
    StructField("dagness_score",           DoubleType(),            nullable=False),
    StructField("is_dag",                  BooleanType(),           nullable=False),
    StructField("top_cycle_contributors",  StringType(),            nullable=True),
    StructField("flagged_story_ids",       ArrayType(StringType()), nullable=True),
])
```

#### CONFOUNDING_AUDIT_SCHEMA (10 fields)

`gold.confounding_audit_log` — FCI-inspired temporal cluster output.

```python
StructType([
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
])
```

---

### 3.2 Treatment Taxonomy Seed Data (34 rows)

**File:** `src/trading_databricks/agents/taxonomy.py` — `TREATMENT_TAXONOMY_SEED: list[dict]`

| L1 | L2 | Description | Example headline |
|----|-----|-------------|-----------------|
| Sentiment | whale_movement | Massive dormant cryptocurrency holdings are transferred on-chain, signaling potential imminent market activity. | Dormant whale address moves $250M in Bitcoin to centralized exchange |
| Sentiment | influencer_signal | High-profile individuals make public statements or social media posts that trigger immediate speculative trading frenzies. | Elon Musk tweets support for Dogecoin, triggering 15% price surge within minutes |
| Sentiment | fund_flow | Asset managers report the net inflows and outflows for publicly traded crypto investment products. | Spot Bitcoin ETFs see record $500M weekly inflow, led by institutional allocators |
| Sentiment | options_expiry | The settlement of massive derivatives contracts forces market makers to hedge, anchoring spot prices to max pain levels. | Bitcoin braces for volatility as $14 billion in options contracts expire today |
| DeFi | tvl_shift | Capital rapidly migrates between decentralized finance sectors, altering the Total Value Locked (TVL) landscape. | Capital rotates from complex restaking into tokenized Real-World Assets as DeFi matures |
| DeFi | protocol_upgrade | A decentralized protocol successfully executes a governance vote or technical hard fork to improve economic mechanics. | Uniswap activates fee switch, distributing protocol revenue to token holders |
| DeFi | stablecoin_depeg | A fiat-pegged digital asset loses parity with its underlying reserve currency due to market panic or collateral failure. | USDC drops to 89 cents amid panic over exposure to collapsed Silicon Valley Bank |
| DeFi | yield_event | Structural changes in on-chain borrowing/lending rates alter the risk-free return profile for decentralized capital. | Fixed-rate lending platform Yield Protocol winds down operations amid declining DeFi demand |
| Supply | halving | Bitcoin's block reward is cut in half approximately every four years, permanently reducing new BTC issuance rate. | Bitcoin halving complete: block reward drops from 6.25 to 3.125 BTC |
| Supply | unlock_distribution | Previously locked tokens are released, or bankruptcy estates distribute recovered assets to creditors. | Mt. Gox estate transfers 10,000 BTC to exchanges, signaling impending creditor distributions |
| Supply | mining_difficulty | Network mining difficulty adjusts to accommodate changes in total computational hash rate, impacting profitability. | Bitcoin mining difficulty hits new all-time high of 150 trillion, squeezing miner margins |
| Supply | burn_event | A protocol permanently removes a portion of its token supply from circulation, increasing asset scarcity. | Ethereum network burns $1.1B in transaction fees following successful EIP-1559 implementation |
| Exchange | listing | A major exchange announces support for a new token, instantly expanding its liquidity and retail accessibility. | Binance announces spot listing for emerging Layer-2 token, driving immediate price surge |
| Exchange | delisting | Trading platforms permanently remove a token due to low liquidity, compliance issues, or network instability. | OKX announces delisting of six altcoin pairs to maintain a robust trading environment |
| Exchange | insolvency | An exchange or lending platform files for bankruptcy or halts withdrawals due to an inability to cover liabilities. | Top crypto lending platform halts all withdrawals amid rumors of critical liquidity crisis |
| Exchange | reserve_proof | Exchanges publish cryptographic Merkle tree audits to verify customer deposits are fully backed by on-chain assets. | Kraken publishes updated Proof of Reserves, verifying 100% asset backing for all customer funds |
| Macro | rate_decision | Central banks alter benchmark interest rates, directly impacting the cost of capital for risk assets. | Fed cuts interest rates by 50 basis points, signaling a shift toward monetary easing |
| Macro | inflation_data | Government agencies release consumer or producer price indices, altering expectations for fiat debasement. | US CPI inflation cools to 2.8%, missing analyst expectations and driving risk-on sentiment |
| Macro | employment_data | Labor market statistics are published, serving as leading economic indicators for central bank policy. | US adds 130,000 jobs in January, outperforming expectations and dampening rate cut hopes |
| Macro | gdp | National economic output data is revised or released, indicating macroeconomic expansion or contraction. | US GDP growth surges past estimates in Q3, bolstering confidence in economic resilience |
| Macro | geopolitical | Significant international conflicts, trade wars, or tariff implementations generate macroeconomic uncertainty. | Global markets reel as new US tariffs trigger massive Bitcoin liquidation cascade |
| Adoption | institutional_buy | Large asset managers or hedge funds publicly disclose significant allocations to BTC or ETH. | Major Wall Street asset manager announces 5% Bitcoin allocation for flagship macro fund |
| Adoption | corporate_treasury | Public or private corporations add digital assets to their balance sheets as a primary treasury reserve. | MicroStrategy acquires an additional 50,000 BTC, expanding total corporate treasury holdings |
| Adoption | payment_integration | Global payment processors or merchant networks enable crypto settlement for commercial transactions. | Visa expands stablecoin settlement capabilities to the Solana network for global merchants |
| Adoption | country_adoption | Sovereign nations formally adopt cryptocurrency as legal tender or establish state-backed reserve funds. | Central African Republic follows El Salvador in adopting Bitcoin as official legal tender |
| Security | exchange_hack | A centralised crypto exchange is exploited, resulting in theft of user funds and loss of market confidence. | Bybit confirms $1.4B hack; withdrawals suspended as investigation begins |
| Security | protocol_exploit | Decentralized smart contracts or cross-chain bridges are compromised due to code vulnerabilities. | Drift Protocol drained of $285M in highly coordinated smart contract exploit |
| Security | rug_pull | Project founders abruptly abandon a protocol, draining liquidity pools and leaving investors with worthless tokens. | Vault of Gems founders vanish with $22M in classic NFT rug pull scheme |
| Security | wallet_compromise | Private keys for institutional hot/cold wallets are stolen via social engineering or physical coercion. | Major market maker loses $50M as hot wallet private keys are compromised |
| Regulation | etf_ruling | A regulatory authority approves, rejects, or delays a Bitcoin or Ethereum spot/futures ETF application. | SEC approves first spot Bitcoin ETF, opening market to institutional investors |
| Regulation | legislation | Legislative bodies pass or propose laws establishing statutory frameworks for digital assets. | Congress passes GENIUS Act, establishing federal regulatory framework for payment stablecoins |
| Regulation | enforcement_action | Regulators bring civil or criminal charges against entities for securities, commodities, or AML violations. | SEC issues Wells notice to major US exchange over alleged unregistered securities offerings |
| Regulation | cbdc_policy | Central banks announce the development, piloting, or cancellation of sovereign digital currencies. | Federal Reserve publishes whitepaper exploring privacy frameworks for potential US CBDC |
| Regulation | tax_policy | Tax authorities issue new guidelines or enforcement mechanisms for reporting crypto capital gains. | Treasury Department proposes new tax reporting rules for digital asset brokers to close compliance gap |

---

### 3.3 Mechanism Whitelist Seed Data (40 entries)

**File:** `src/trading_databricks/agents/taxonomy.py` — `MECHANISM_WHITELIST_SEED: list[dict]`

| L1 | Mechanism | Valid Directions | Description |
|----|-----------|-----------------|-------------|
| Regulation | regulatory_clarity | bullish | Removal of regulatory ambiguity reduces uncertainty premium, attracting institutional and retail capital inflows. |
| Regulation | regulatory_restriction | bearish | Prohibitions or strict licensing requirements sever capital access and force structural reductions in market participation. |
| Regulation | institutional_access | bullish | Legal frameworks enabling traditional finance entities to custody or trade crypto drastically expand the available capital base. |
| Regulation | compliance_burden | bearish | Onerous tax or AML reporting requirements increase operational friction, reducing trading velocity and market maker profitability. |
| Regulation | market_legitimization | bullish | Sovereign or tier-one regulatory endorsement lowers the psychological risk profile of digital assets, inviting risk-averse capital. |
| Security | sell_pressure | bearish | Hackers liquidating stolen digital assets for fiat create immediate, overwhelming supply gluts on the open market. |
| Security | trust_erosion | bearish | A security breach destroys user confidence in the affected platform or asset class, triggering withdrawals and sell-offs. |
| Security | market_exit | bearish | Severe hacks render platforms insolvent, leading to permanent operational shutdown and complete loss of associated asset value. |
| Security | contagion_spread | bearish | A vulnerability in one DeFi primitive cascades across interconnected lending protocols, threatening systemic collateral health. |
| Security | user_withdrawal | bearish | Fear of exposure to a breach triggers a localized bank run, depleting reserve liquidity and halting operational capacity. |
| Adoption | demand_increase | bullish | Direct purchasing by new demographic cohorts or commercial entities absorbs circulating supply, elevating the clearing price. |
| Adoption | institutional_inflow | bullish | Sustained capital allocation from macro funds and ETFs creates a robust price floor and drastically improves market depth. |
| Adoption | network_expansion | bullish | Growth in active wallet addresses and integrations validates Metcalfe's Law, fundamentally driving protocol valuation. |
| Adoption | utility_unlock | bullish | Integration into legacy payment networks transitions a token from a speculative instrument to a productive commercial currency. |
| Adoption | strategic_accumulation | bullish | Corporate or sovereign treasuries aggressively hoarding assets permanently reduces free float, magnifying future demand shocks. |
| Macro | risk_on | bullish | Favorable macroeconomic conditions encourage capital to seek higher yields out on the risk curve, benefiting high-beta assets. |
| Macro | risk_off | bearish | Economic uncertainty or rising yields drive massive capital flight from speculative networks into traditional safe havens. |
| Macro | liquidity_expansion | bullish | Central bank quantitative easing injects excess fiat liquidity into the global economy, inherently inflating finite digital assets. |
| Macro | liquidity_contraction | bearish | Quantitative tightening drains speculative fiat liquidity from the financial system, starving crypto markets of upward momentum. |
| Macro | inflation_hedge | bullish | Investors rotate capital into Bitcoin as a store-of-value hedge when fiat purchasing power erodes due to persistent inflation. |
| Exchange | liquidity_increase | bullish | Tier-1 exchange listings drastically lower trading friction and grant exposure to massive new retail and institutional pools. |
| Exchange | trust_erosion | bearish | Delistings or rumors of exchange instability degrade faith in centralized market structure, prompting widespread capital flight. |
| Exchange | market_access_expansion | bullish | Securing operational licenses in new jurisdictions unlocks previously inaccessible geographic sectors of retail liquidity. |
| Exchange | trading_suspension | bearish | Halting withdrawals traps user capital, instantly signaling critical insolvency and sparking panic selling on secondary markets. |
| Exchange | counterparty_risk | bearish | Fears of fractional reserves or commingled funds destroy the risk-reward ratio of centralized custody, accelerating insolvency. |
| Supply | sell_pressure | bearish | The mass distribution of recovered bankruptcy assets or token unlocks floods the market with highly liquid selling volume. |
| Supply | supply_reduction | bullish | Programmatic halvings slash the structural daily issuance of new coins, starving the market of required seller liquidity. |
| Supply | miner_capitulation | bearish, volatile | Operating losses force miners to aggressively dump BTC treasuries, creating sharp drawdowns that historically mark cycle bottoms. |
| Supply | scarcity_signal | bullish | Deflationary fee-burn mechanics programmatically destroy circulating supply, reinforcing the psychological hard-money narrative. |
| Supply | vesting_unlock | bearish, volatile | The scheduled release of venture capital tokens drastically inflates the float, inviting aggressive front-run short selling. |
| DeFi | contagion_risk | bearish | The failure of a foundational stablecoin or primitive triggers cascading debt liquidations across dependent DeFi protocols. |
| DeFi | yield_migration | volatile | Mercenary capital rapidly rotates between chains chasing high APYs, causing severe localized liquidity crunches and price swings. |
| DeFi | liquidity_drain | bearish | The rapid exit of TVL from an automated market maker creates extreme slippage and mathematically ensures token death spirals. |
| DeFi | protocol_confidence | bullish | The implementation of institutional-grade risk frameworks and successful technical upgrades attracts durable, sticky capital. |
| DeFi | stablecoin_risk | bearish, volatile | Depegging fears force users to dump stablecoins for volatile assets, temporarily pumping prices before a structural macro crash. |
| Sentiment | momentum_signal | volatile | Trend-following algorithmic traders exacerbate directional moves, violently extending both speculative rallies and panic crashes. |
| Sentiment | liquidation_risk | bearish, volatile | High systemic leverage ensures that minor price drops trigger cascading margin calls, resulting in violent downside wicks. |
| Sentiment | fear_signal | bearish | Overwhelmingly negative sentiment indices and media cycles prompt retail capitulation and defensive institutional de-risking. |
| Sentiment | greed_signal | bullish, volatile | Euphoric retail speculation drives asset bubbles with exponential price appreciation, inevitably preceding sharp mean-reversions. |
| Sentiment | narrative_shift | volatile | Social velocity unpredictably rotates capital between meta-themes, draining established sectors to fuel localized asset frenzies. |

**Cross-L1 mechanisms** (`sell_pressure`, `trust_erosion`) intentionally appear under multiple L1s with distinct descriptions and the same valid direction set.

---

### 3.4 Taxonomy Functions

**File:** `src/trading_databricks/agents/taxonomy.py`

```python
def seed_taxonomy(spark, *, catalog: str, schema: str) -> None:
    """Write treatment_taxonomy and mechanism_whitelist to Delta (overwrite)."""

def load_taxonomy(spark, *, catalog: str, schema: str) -> dict[str, list[str]]:
    """Return {l1_category: [l2_subcategory, ...]} from Delta."""

def load_whitelist(spark, *, catalog: str, schema: str) -> dict[tuple[str, str], list[str]]:
    """Return {(treatment_l1, mechanism): [valid_directions]} from Delta."""
```

Both `load_*` functions call `.collect()` — tables are small reference data, never streamed.

**One-shot seeder:** `scripts/seed_taxonomy.py` — argparse (`--catalog`, `--schema`), calls `seed_taxonomy()`, asserts row counts.

**Tests:** `tests/agents/test_schemas.py` (schema structure, no Spark) + `tests/agents/test_taxonomy.py` (seed list integrity, no Spark).

---

## 4. Exports (Public API for Wave 2+)

| Module | Exported names | Consumed by |
|--------|---------------|-------------|
| `agents.schemas` | `CAUSAL_FEATURES_SCHEMA`, `TREATMENT_TAXONOMY_SCHEMA`, `MECHANISM_WHITELIST_SCHEMA`, `DAGNESS_AUDIT_SCHEMA`, `CONFOUNDING_AUDIT_SCHEMA` | TRD-25, 27, 28, 29, 30 |
| `agents.taxonomy` | `seed_taxonomy`, `load_taxonomy`, `load_whitelist`, `TREATMENT_TAXONOMY_SEED`, `MECHANISM_WHITELIST_SEED` | TRD-29 (extractor prompt injection), TRD-30 (validator) |
| `agents.config` | `GatekeeperConfig`, `UncertaintyRouterConfig`, `MechanismValidatorConfig`, `MASConfig` | TRD-25 (gatekeeper), TRD-27 (context retriever), TRD-28 (few-shot), TRD-29 (extractor), TRD-30 (validator) |

---

## 5. Verification

```bash
# Local — no cluster required
uv run pytest tests/agents/ -v

# Databricks Connect — seeds dev catalog
uv run python scripts/seed_taxonomy.py --catalog smart-trading-dev --schema gold

# SQL verification post-seed
SELECT COUNT(*) FROM `smart-trading-dev`.03_gold.treatment_taxonomy;   -- 34
SELECT COUNT(*) FROM `smart-trading-dev`.03_gold.mechanism_whitelist;  -- 40
SELECT treatment_l1, COUNT(*) FROM `smart-trading-dev`.03_gold.mechanism_whitelist
  GROUP BY 1 ORDER BY 1;                                                  -- 5 per L1
```
