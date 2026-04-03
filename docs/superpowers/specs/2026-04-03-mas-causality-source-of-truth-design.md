# MAS Causality Layer — Source-of-Truth Design

**Date:** 2026-04-03  
**Status:** Approved  
**Implements:** Tasks 1.2, 1.3, 1.4 of `docs/superpowers/plans/2026-04-03-mas-causality-layer-redesign.md`  
**Feeds into:** Agent prompts, Mechanism Validator, `gold.news_causal_features` SDP pipeline

---

## Purpose

This spec defines the three deterministic data artifacts that the MAS causality pipeline treats as ground truth:

1. **`gold.treatment_taxonomy`** — the closed set of valid treatment classes the Causality Extractor may assign
2. **`gold.mechanism_whitelist`** — the constraint table the Mechanism Validator checks against deterministically
3. **`CAUSAL_FEATURES_SCHEMA`** — the Spark StructType contract for `gold.news_causal_features`

All agent prompts, validator logic, and ML consumers are downstream of these definitions. Mechanism name changes after deployment are schema migrations.

---

## Architecture

```
src/trading_databricks/agents/taxonomy.py     (this spec — Tasks 1.2, 1.3)
  TREATMENT_TAXONOMY_SCHEMA
  MECHANISM_WHITELIST_SCHEMA
  TREATMENT_TAXONOMY_SEED   (34 rows)
  MECHANISM_WHITELIST_SEED  (40 entries)

src/trading_databricks/news/schemas.py        (this spec — Task 1.4)
  CAUSAL_FEATURES_SCHEMA    (17 fields)

src/trading_databricks/agents/pipeline.py     (Task 3.4 — not in scope here)
  @dp.table → gold.treatment_taxonomy
  @dp.table → gold.mechanism_whitelist
```

`taxonomy.py` is a pure data module — no Spark session, no I/O. The SDP pipeline (Task 3.4) imports it and wraps its seed data in `@dp.table` decorators to write Delta tables.

**V1 governance note:** SDP pipeline overwrites both tables on each run. New entries require updating `taxonomy.py` + pipeline redeploy. V2 can promote to live Delta `MERGE INTO` for updates without code deploy.

---

## `gold.treatment_taxonomy`

### Schema

```python
TREATMENT_TAXONOMY_SCHEMA = StructType([
    StructField("l1_category",    StringType(), nullable=False),  # L1 bucket (frozen in V1)
    StructField("l2_subcategory", StringType(), nullable=False),  # Tag used in treatment_class output
    StructField("description",    StringType(), nullable=False),  # Human-readable definition
    StructField("example",        StringType(), nullable=False),  # Concrete news headline example
])
```

**Primary key:** `(l1_category, l2_subcategory)` — uniqueness enforced at seed time.  
**Consumers:** Causality Extractor system prompt (injected reference list), human auditing.  
**Output format in pipeline:** `treatment_class = "L1.l2_subcategory"` (e.g., `"Regulation.etf_ruling"`)

### Seed Data — 34 Rows

#### Regulation (5)

| l2_subcategory | description | example |
|----------------|-------------|---------|
| `etf_ruling` | Regulatory decision on ETF applications for crypto assets, directly affecting institutional access to spot or futures markets. | "SEC approves spot BTC ETF, enabling institutional investment via Fidelity and BlackRock" |
| `legislation` | Passage or proposal of laws governing crypto ownership, trading, or use that reshape the legal operating environment. | "US Congress passes Digital Asset Market Structure Act establishing SEC/CFTC jurisdiction" |
| `enforcement_action` | Regulatory agencies initiate enforcement proceedings against crypto entities, creating legal uncertainty and reputational damage. | "SEC files suit against Coinbase alleging unregistered securities exchange" |
| `cbdc_policy` | Central bank decisions on digital currency design or rollout that may compete with or complement private crypto assets. | "China accelerates digital yuan rollout to 500M retail users, restricting private stablecoins" |
| `tax_policy` | Changes to tax treatment of crypto transactions, gains, or holdings that alter investor behavior and on-chain activity. | "IRS mandates mandatory crypto cost-basis reporting starting 2025 tax year" |

#### Security (4)

| l2_subcategory | description | example |
|----------------|-------------|---------|
| `exchange_hack` | Unauthorized access to exchange hot or cold wallets resulting in theft of user funds, threatening solvency. | "Bybit hot wallet drained of $200M in ETH via private key compromise" |
| `protocol_exploit` | Exploitation of a smart contract vulnerability draining protocol funds or enabling unauthorized token minting. | "Euler Finance exploited for $197M via flash loan re-entrancy attack" |
| `rug_pull` | Deliberate project abandonment by founders who withdraw liquidity, leaving investors with worthless tokens. | "New DeFi protocol founders disappear with $50M in user deposits" |
| `wallet_compromise` | Theft of funds from individual or institutional wallets via private key compromise, phishing, or malware. | "Institutional custody provider reports $30M in hot wallet theft from key management breach" |

#### Adoption (4)

| l2_subcategory | description | example |
|----------------|-------------|---------|
| `institutional_buy` | Publicly disclosed acquisition of crypto assets by institutional investors, hedge funds, or asset managers. | "ARK Invest discloses $200M BTC position in quarterly SEC filing" |
| `corporate_treasury` | Corporate decision to add crypto assets to balance sheet as a treasury reserve asset. | "MicroStrategy announces purchase of additional 10,000 BTC for corporate treasury" |
| `payment_integration` | Integration of crypto payments into major consumer or enterprise platforms, expanding real-world utility. | "Stripe announces native USDC settlement rails for global merchant payments" |
| `country_adoption` | National government decision to make crypto a legal tender, reserve asset, or officially sanctioned payment method. | "El Salvador announces $500M BTC national reserve accumulation program" |

#### Macro (5)

| l2_subcategory | description | example |
|----------------|-------------|---------|
| `rate_decision` | Central bank interest rate decisions that alter the risk-free rate and the opportunity cost of holding speculative assets. | "Fed holds rates at 5.25% for sixth consecutive meeting, signaling prolonged tight conditions" |
| `inflation_data` | CPI or PPI releases that shape expectations for monetary policy trajectory and store-of-value demand for BTC. | "US CPI comes in at 4.2%, above expectations of 3.8%, raising rate-hike fears" |
| `employment_data` | Labor market releases (payrolls, unemployment rate) that drive Fed policy expectations and risk asset sentiment. | "US adds 350K jobs in March, crushing estimates of 180K, cementing hawkish Fed stance" |
| `gdp` | Gross domestic product releases that signal economic health and influence monetary policy and risk appetite. | "US Q4 GDP revised down to 1.2% annualized, raising recession probability estimates" |
| `geopolitical` | International conflicts, sanctions, or geopolitical events that disrupt global financial systems or drive safe-haven demand. | "Russia announces BTC accepted for oil settlement contracts with friendly nations" |

#### Exchange (4)

| l2_subcategory | description | example |
|----------------|-------------|---------|
| `listing` | Crypto asset added to major exchange spot or derivatives trading, increasing liquidity and market access. | "Coinbase lists BTC-EUR and BTC-GBP spot pairs for European retail users" |
| `delisting` | Crypto asset removed from exchange trading, reducing liquidity and signaling regulatory or reputational concerns. | "Binance US delists BTC-USDT pair following SEC enforcement pressure" |
| `insolvency` | Exchange becomes insolvent, halts withdrawals, or enters bankruptcy, threatening user funds and market confidence. | "FTX files Chapter 11 bankruptcy with $8B customer shortfall disclosed" |
| `reserve_proof` | Exchange publishes proof-of-reserves audit or on-chain Merkle tree verification to demonstrate solvency. | "Binance releases Merkle tree proof of full 1:1 BTC reserves to users" |

#### Supply (4)

| l2_subcategory | description | example |
|----------------|-------------|---------|
| `halving` | Bitcoin's programmatic block reward reduction event cutting new BTC supply by 50%, occurring every ~210,000 blocks. | "Bitcoin halving reduces miner block reward from 3.125 to 1.5625 BTC" |
| `unlock_distribution` | Large-scale release of previously locked or encumbered BTC to creditors, investors, or open market. | "Mt. Gox trustee confirms schedule for distributing 142,000 BTC to creditors starting July 2026" |
| `mining_difficulty` | Bitcoin network difficulty adjustment that affects miner profitability, hashrate, and sell pressure dynamics. | "Bitcoin mining difficulty hits all-time high, forcing high-cost miners to sell BTC to fund operations" |
| `burn_event` | Deliberate or protocol-mandated destruction of tokens that permanently reduces circulating supply. | "ETH burn rate exceeds issuance for 30 consecutive days, making ETH net deflationary" |

#### DeFi (4)

| l2_subcategory | description | example |
|----------------|-------------|---------|
| `tvl_shift` | Significant movement of total value locked into or out of DeFi protocols, signaling macro capital flow trends. | "Aave TVL surges 40% as DeFi yield rates outpace TradFi money market funds" |
| `protocol_upgrade` | Major smart contract upgrade or fork that changes protocol mechanics, fee structures, or tokenomics. | "Uniswap v4 launch introduces hook architecture and native ETH routing, slashing gas costs" |
| `stablecoin_depeg` | Stablecoin loses its peg to the reference asset, triggering cascading liquidations and protocol failures. | "USDC temporarily trades at $0.87 following SVB bank run and Circle reserve concerns" |
| `yield_event` | Change in DeFi yields that drives capital rotation between protocols or between DeFi and traditional finance. | "Compound offers 12% APY on USDC, attracting $2B TVL migration from Aave" |

#### Sentiment (4)

| l2_subcategory | description | example |
|----------------|-------------|---------|
| `whale_movement` | On-chain transfer of large BTC or ETH amounts between wallets, signaling potential buy or sell pressure. | "Dormant BTC wallet from 2012 transfers 5,000 BTC to Binance deposit address" |
| `influencer_signal` | Public statement by prominent market participants, executives, or influencers that shifts retail expectations. | "Elon Musk tweets 'BTC to the moon' with laser eyes, triggering retail FOMO buying" |
| `fund_flow` | Large directional capital flow into or out of crypto investment vehicles (ETFs, funds, trusts) as tracked by analytics firms. | "Bitcoin spot ETFs record $800M single-day net inflow, largest since launch" |
| `options_expiry` | Large options or futures contract expiration event that can trigger hedging, gamma squeezes, or volatility clustering. | "$6B notional BTC options expire Friday with max pain at $45K, below spot" |

**Total: 34 rows** (Regulation: 5, Macro: 5, Security/Adoption/Exchange/Supply/DeFi/Sentiment: 4 each)

---

## `gold.mechanism_whitelist`

### Schema

```python
MECHANISM_WHITELIST_SCHEMA = StructType([
    StructField("treatment_l1",     StringType(),              nullable=False),  # FK → l1_category in taxonomy
    StructField("mechanism",        StringType(),              nullable=False),  # Snake_case causal channel
    StructField("valid_directions", ArrayType(StringType()),   nullable=False),  # Subset of {bullish,bearish,volatile}
    StructField("description",      StringType(),              nullable=False),  # What this mechanism means
])
```

**Primary key:** `(treatment_l1, mechanism)`.  
A mechanism may appear under multiple L1s (e.g., `sell_pressure` appears under Security and Supply — different root causes, same transmission channel).  
**Consumers:** Mechanism Validator (deterministic lookup), Causality Extractor system prompt (injected whitelist).

### Validator Logic

```
1. Parse treatment_class → extract L1  (e.g., "Supply.halving" → "Supply")
2. Lookup (L1, mechanism) in whitelist
3. NOT FOUND       → validation_status = "rejected_mechanism"
4. direction ∉ valid_directions → validation_status = "rejected_direction"
5. any evidence_span ∉ substrings(article.body) → validation_status = "rejected_evidence"
6. all pass        → validation_status = "accepted"
```

### Seed Data — 40 Entries (5 per L1)

#### Regulation

| mechanism | valid_directions | description |
|-----------|-----------------|-------------|
| `regulatory_clarity` | bullish, volatile | Regulatory ruling provides legal certainty, enabling new market participants or removing legal risk premiums from crypto assets. |
| `regulatory_restriction` | bearish, volatile | Regulatory action restricts trading, bans crypto activity, or removes market access for key participant classes. |
| `institutional_access` | bullish | Regulatory change unlocks a new class of institutional investors to legally hold or trade crypto assets at scale. |
| `compliance_burden` | bearish, volatile | New regulatory requirements increase operational costs or friction for crypto service providers and end users. |
| `market_legitimization` | bullish | Formal sovereign recognition of crypto as a legal asset class, legal tender, or sanctioned payment method. |

#### Security

| mechanism | valid_directions | description |
|-----------|-----------------|-------------|
| `sell_pressure` | bearish | Forced or reactive sale of crypto assets following a security breach to cover losses or halt contagion spread. |
| `trust_erosion` | bearish, volatile | Reduction in user confidence in a platform or protocol following a security incident, triggering exits and fund withdrawals. |
| `market_exit` | bearish | Platform ceases operations following a breach, forcing user fund withdrawals and removing liquidity from the market. |
| `contagion_spread` | bearish, volatile | Security incident in one entity creates fear and systemic contagion across correlated DeFi or CeFi systems. |
| `user_withdrawal` | bearish | Mass withdrawal of funds from a platform following news of a security vulnerability or suspected insolvency. |

#### Adoption

| mechanism | valid_directions | description |
|-----------|-----------------|-------------|
| `demand_increase` | bullish | Organic growth in demand for crypto assets driven by new user segments, products, or real-world use cases. |
| `institutional_inflow` | bullish | Measurable capital allocation from institutional investors into crypto via ETFs, funds, or direct spot purchase. |
| `network_expansion` | bullish | Growth of the active user base, developer ecosystem, or integrations that expand a crypto network's utility. |
| `utility_unlock` | bullish | New technical or commercial use case increases the practical utility and sustained demand for a crypto asset. |
| `strategic_accumulation` | bullish | Announced ongoing purchase program by a sovereign, corporate, or institutional entity as a declared strategic reserve. |

#### Macro

| mechanism | valid_directions | description |
|-----------|-----------------|-------------|
| `risk_on` | bullish | Macro environment shifts to risk-on, increasing speculative appetite and directing capital flows toward crypto assets. |
| `risk_off` | bearish | Macro environment shifts to risk-off, triggering flight to safety and systematic selling of speculative assets. |
| `liquidity_expansion` | bullish | Central bank or government policy increases monetary liquidity, raising the floor for risk assets including crypto. |
| `liquidity_contraction` | bearish | Central bank tightening or QT reduces available monetary liquidity, putting downward pressure on speculative assets. |
| `inflation_hedge` | bullish | Rising or persistent inflation increases demand for BTC as a hard-money inflation hedge and store of value. |

#### Exchange

| mechanism | valid_directions | description |
|-----------|-----------------|-------------|
| `liquidity_increase` | bullish, volatile | New spot or derivatives pair listing increases market depth and trading volume for an asset. |
| `trust_erosion` | bearish | Revealed exchange insolvency, fraud, or misuse of customer funds destroys user confidence and triggers withdrawals. |
| `market_access_expansion` | bullish | Exchange opens to a new geographic region or user segment, expanding the addressable market for crypto trading. |
| `trading_suspension` | bearish, volatile | Exchange halts trading or withdrawals, signaling operational distress and trapping user funds. |
| `counterparty_risk` | bearish, volatile | Credible concern raised about exchange solvency or reserve adequacy, triggering preventive user withdrawals. |

#### Supply

| mechanism | valid_directions | description |
|-----------|-----------------|-------------|
| `sell_pressure` | bearish | Large volume of previously locked or unavailable BTC enters the market, increasing supply and downward price pressure. |
| `supply_reduction` | bullish | Programmatic or structural reduction in new crypto supply entering the market, supporting scarcity and higher prices. |
| `miner_capitulation` | bearish, volatile | Miners forced to sell accumulated BTC to cover operational costs, adding meaningful sell pressure to the market. |
| `scarcity_signal` | bullish | On-chain data showing declining exchange reserves or concentrated supply signals reduced available sell-side inventory. |
| `vesting_unlock` | bearish, volatile | Large pre-mined, VC-held, or team token unlock event increases circulating supply, creating near-term selling risk. |

#### DeFi

| mechanism | valid_directions | description |
|-----------|-----------------|-------------|
| `contagion_risk` | bearish, volatile | Failure or stress in one DeFi protocol creates cascading risk across interconnected protocols and stablecoins. |
| `yield_migration` | bullish, bearish | Shift in DeFi yield rates drives capital rotation between protocols or between DeFi and traditional finance. |
| `liquidity_drain` | bearish | Exploit, bank run, or mass withdrawal drains liquidity from DeFi protocols, reducing market depth and efficiency. |
| `protocol_confidence` | bullish | Successful upgrade, audit, or partnership increases user confidence in a DeFi protocol, attracting capital inflows. |
| `stablecoin_risk` | bearish, volatile | Stablecoin loses peg or faces existential threat, creating cascading liquidations and collateral collapses across DeFi. |

#### Sentiment

| mechanism | valid_directions | description |
|-----------|-----------------|-------------|
| `momentum_signal` | bullish, bearish | Sustained directional price movement reinforces itself through trend-following behavior and media attention loops. |
| `liquidation_risk` | bearish, volatile | Large leveraged long or short positions approach liquidation thresholds, risking cascade across the derivatives market. |
| `fear_signal` | bearish | Extreme fear metrics, capitulation selling, or panic-driven behavior signals risk-off sentiment at scale. |
| `greed_signal` | bullish | Extreme greed metrics, euphoric buying, or FOMO-driven behavior signals elevated speculative demand. |
| `narrative_shift` | bullish, bearish, volatile | The dominant market narrative (store of value, digital gold, DeFi summer) pivots direction, realigning capital flows. |

**Total: 40 entries** (5 per L1). `sell_pressure` and `trust_erosion` intentionally appear under multiple L1s with distinct `treatment_l1` keys.

---

## `CAUSAL_FEATURES_SCHEMA`

Add to `src/trading_databricks/news/schemas.py` after `NEWS_RAW_SCHEMA_HINTS`.

```python
# Gold layer — gold.news_causal_features
# 17-field ML handshake contract for the MAS causality pipeline.
# Fields story_id, published_at, is_impactful, agent_model_version, processed_at
# are always populated (NOT NULL).
# All other fields are NULL when is_impactful=False.
# market_vol_at_publish, market_trend_at_publish, dollar_bar_close are NULL in V1
# (deferred to TRD-5 dollar bars implementation).
CAUSAL_FEATURES_SCHEMA = StructType([
    StructField("story_id",                 StringType(),            nullable=False),
    StructField("published_at",             TimestampType(),         nullable=False),
    StructField("is_impactful",             BooleanType(),           nullable=False),
    StructField("treatment_class",          StringType(),            nullable=True),
    StructField("trigger_event",            StringType(),            nullable=True),
    StructField("mechanism",                StringType(),            nullable=True),
    StructField("asset",                    StringType(),            nullable=True),
    StructField("expected_direction",       StringType(),            nullable=True),
    StructField("confidence",               DoubleType(),            nullable=True),
    StructField("causal_chain_json",        StringType(),            nullable=True),
    StructField("evidence_spans",           ArrayType(StringType()), nullable=True),
    StructField("validation_status",        StringType(),            nullable=True),
    StructField("market_vol_at_publish",    DoubleType(),            nullable=True),
    StructField("market_trend_at_publish",  StringType(),            nullable=True),
    StructField("dollar_bar_close",         DecimalType(18, 8),      nullable=True),
    StructField("agent_model_version",      StringType(),            nullable=False),
    StructField("processed_at",             TimestampType(),         nullable=False),
])
```

### Field Contracts

| Field | Constraint | Notes |
|-------|-----------|-------|
| `story_id` | NOT NULL | FK → `silver.news_cleaned.story_id` |
| `treatment_class` | `"L1.l2_subcategory"` format | Must exist in `gold.treatment_taxonomy` |
| `mechanism` | Whitelist-constrained | Must exist in `gold.mechanism_whitelist` for the given L1 |
| `asset` | `"BTC"` \| `"ETH"` \| `"BOTH"` | Enum enforced by Extractor prompt |
| `expected_direction` | `"bullish"` \| `"bearish"` \| `"volatile"` | Enum enforced by Extractor prompt |
| `confidence` | `[0.0, 1.0]` | Agent self-assessed confidence |
| `causal_chain_json` | JSON string | Matches causal chain schema (Section 2 of redesign plan) |
| `evidence_spans` | 1–3 elements | Each must be an exact substring of `silver.news_cleaned.body` |
| `validation_status` | `"accepted"` \| `"rejected_mechanism"` \| `"rejected_direction"` \| `"rejected_evidence"` | Set by Mechanism Validator |
| `dollar_bar_close` | `DecimalType(18, 8)` | Matches `BAR_SCHEMA` precision; NULL in V1 |
| `agent_model_version` | NOT NULL | MLflow model version string for lineage + A/B testing |

---

## V2 Consideration: Literature-Extracted Ontology

An alternative validator architecture was evaluated: extracting causal claims from institutional crypto literature (Messari, Glassnode, Fed/NBER papers) via `ai_parse_document` + LLM extraction + embedding deduplication, then querying a Vector Search index at inference time.

**Decision: Defer to V2.** Rationale:
- Extraction pipeline is a ~2-week project that blocks MAS V1 shipping
- Semantic similarity threshold introduces tuning complexity without ground truth to calibrate against
- V1 rejection patterns (`validation_status = rejected_mechanism`) are the best empirical signal for building V2's extraction prompts

**Migration path:** V1's 40-entry whitelist becomes the seed/bootstrapping set for V2's ontology extraction. The validator interface (`validate_mechanism(l1, mechanism, direction, evidence)`) is abstracted — swapping from exact-lookup to vector search requires only changing the validator's internals, not the agent graph or output schema.

---

## Files

| File | Action |
|------|--------|
| `src/trading_databricks/agents/taxonomy.py` | **Create** — seed data + schemas |
| `src/trading_databricks/news/schemas.py` | **Modify** — add `CAUSAL_FEATURES_SCHEMA` |
| `src/trading_databricks/agents/pipeline.py` | Task 3.4 (out of scope) |
| `resources/agents_pipeline.pipeline.yml` | Task 3.4 (out of scope) |

---

## Verification Checklist

- [ ] `len(TREATMENT_TAXONOMY_SEED) == 34`
- [ ] `len(MECHANISM_WHITELIST_SEED) == 40`
- [ ] Every `(treatment_l1, mechanism)` pair is unique in `MECHANISM_WHITELIST_SEED`
- [ ] Every `treatment_l1` in whitelist exists as an `l1_category` in taxonomy
- [ ] Every `valid_directions` list is a non-empty subset of `{"bullish", "bearish", "volatile"}`
- [ ] `len(CAUSAL_FEATURES_SCHEMA.fields) == 17`
- [ ] Non-nullable fields exactly: `story_id`, `published_at`, `is_impactful`, `agent_model_version`, `processed_at`
- [ ] `dollar_bar_close` is `DecimalType(18, 8)`, not `DoubleType`

Post-deploy SQL checks (Task 3.4):
```sql
SELECT COUNT(*) FROM gold.treatment_taxonomy;   -- expect 34
SELECT COUNT(*) FROM gold.mechanism_whitelist;  -- expect 40
SELECT l1_category, COUNT(*) as n FROM gold.treatment_taxonomy GROUP BY 1 ORDER BY 1;
-- Regulation: 5, Macro: 5, others: 4
SELECT treatment_l1, COUNT(*) as n FROM gold.mechanism_whitelist GROUP BY 1 ORDER BY 1;
-- all 8 L1s: 5 each
```
