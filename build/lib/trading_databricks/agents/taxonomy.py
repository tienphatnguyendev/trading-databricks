"""Treatment taxonomy and mechanism whitelist seed data and loader functions.

TREATMENT_TAXONOMY_SEED: 34 rows, 8 L1 categories.
MECHANISM_WHITELIST_SEED: 40 rows, 5 mechanisms per L1 category.

Call seed_taxonomy() once per environment to populate the Delta reference tables.
Use load_taxonomy() and load_whitelist() at agent runtime to read them.
"""
from __future__ import annotations

from trading_databricks.agents.schemas import (
    MECHANISM_WHITELIST_SCHEMA,
    TREATMENT_TAXONOMY_SCHEMA,
)

# ---------------------------------------------------------------------------
# Canonical seed data — 34 rows, 8 L1 categories
# L2 counts: Sentiment(4), DeFi(4), Supply(4), Exchange(4),
#            Macro(5), Adoption(4), Security(4), Regulation(5) = 34 total
# ---------------------------------------------------------------------------
TREATMENT_TAXONOMY_SEED: list[dict] = [
    # Sentiment
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
    # DeFi
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
    # Supply
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
    # Exchange
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
    # Macro
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
    # Adoption
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
    # Security
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
    # Regulation
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

# ---------------------------------------------------------------------------
# Canonical seed data — 40 rows, 5 mechanisms per L1 category
# ---------------------------------------------------------------------------
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


def seed_taxonomy(spark, *, catalog: str, schema: str) -> None:
    """Write treatment taxonomy and mechanism whitelist to Delta.

    Uses overwrite + overwriteSchema=True. One-shot per environment.
    Requires Databricks Connect or cluster context.
    """
    taxonomy_df = spark.createDataFrame(TREATMENT_TAXONOMY_SEED, schema=TREATMENT_TAXONOMY_SCHEMA)
    (
        taxonomy_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"`{catalog}`.`{schema}`.treatment_taxonomy")
    )

    whitelist_df = spark.createDataFrame(MECHANISM_WHITELIST_SEED, schema=MECHANISM_WHITELIST_SCHEMA)
    (
        whitelist_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"`{catalog}`.`{schema}`.mechanism_whitelist")
    )


def load_taxonomy(spark, *, catalog: str, schema: str) -> dict[str, list[str]]:
    """Load taxonomy as {l1_category: [l2_subcategory, ...]} dict."""
    rows = spark.table(f"`{catalog}`.`{schema}`.treatment_taxonomy").collect()
    result: dict[str, list[str]] = {}
    for row in rows:
        result.setdefault(row.l1_category, []).append(row.l2_subcategory)
    return result


def load_whitelist(spark, *, catalog: str, schema: str) -> dict[tuple[str, str], list[str]]:
    """Load whitelist as {(treatment_l1, mechanism): [valid_directions]} dict."""
    rows = spark.table(f"`{catalog}`.`{schema}`.mechanism_whitelist").collect()
    return {(row.treatment_l1, row.mechanism): list(row.valid_directions) for row in rows}
