# Multi-Agent System for Financial News Causal Analysis — Final Specification

**Date:** 2026-04-04
**Version:** 2.0 (V2 RAG-Augmented)
**Status:** Draft — Pending Approval
**Supersedes:** `multi-agent-system-for-financial-news-analysis-specs.md` (4-agent sketch, 6-field schema)
**Consolidates:**
- `2026-04-03-mas-causality-layer-redesign.md` (agent topology + SCM foundation)
- `2026-04-03-mas-causality-source-of-truth-design.md` (taxonomy + whitelist + schema)
- `2026-04-04-gatekeeper-ner-classifier-plan.md` (NER/sklearn gatekeeper)
- Reliability fixes from cross-document review (6 concerns resolved)
- RAG/Vector Search architecture (dynamic few-shot retrieval)

**Depends on:** `silver.news_cleaned` (implemented), `bronze.news_agent_eval_dataset` (implemented)
**Deferred dependency:** TRD-5 (dollar bars) — market context columns schema-present but NULL until implemented

---

## 1. Business Objective

The MAS is the **treatment extraction engine** for the trading platform's structural causal model. It transforms raw financial news articles into structured, validated causal features that the downstream ML pipeline consumes to estimate the causal effect of news events on BTC/ETH price returns.

**What it does:**
1. **Filters** the ~95% irrelevant general financial wire to the ~5% crypto-impactful articles (Gatekeeper)
2. **Extracts** a discrete treatment class, mechanism, and evidence from each impactful article (Causality Extractor)
3. **Validates** the extraction against a deterministic ontology (Mechanism Validator)
4. **Writes** a 20-field structured row per article to `gold.news_causal_features`

**What it does NOT do:** predict price direction, execute trades, or compute returns. Those are downstream ML responsibilities.

---

## 2. SCM: Mathematical Foundation

### 2.1 Causal DAG

The system models: **does a news event (T) cause a price movement (Y), or is observed correlation confounded by market conditions (X)?**

```
         U (unobserved: order flow, insider info, whale positioning)
         |
         v
Z ─────> T ───M───> Y
         ^          ^
         |          |
         X ─────────┘
```

| Variable | Definition | Source | MAS Role |
|----------|-----------|--------|----------|
| Z | Raw news article (headline + body) | `silver.news_cleaned` | Input |
| T | Discrete treatment — causal event class | `treatment_class` output | **Primary output** |
| M | Mechanism — causal transmission channel | `mechanism` output | Mediator |
| X | Pre-treatment confounders — market state at publication | `gold.dollar_bars` (NULL until TRD-5) | Context |
| Y | Outcome — forward BTC/ETH return over horizon h | `gold.dollar_bars` (post-publication) | Not in MAS |
| U | Unobserved (order book, positioning, latent sentiment) | Not measured | Irreducible |

### 2.2 Structural Equations

**Equation 1 — Treatment extraction (the MAS):**
```
T_i = g(Z_i, X_i, R_i; θ_MAS) + ε_T
```
- `g()` = the full agent pipeline (gatekeeper → extractor → validator)
- `θ_MAS` = model weights, prompt parameters, retrieval index state
- `R_i` = **retrieved few-shot exemplars** from Vector Search (NEW in V2)
- `ε_T` = LLM non-determinism (same article → slightly different extractions)

The MAS IS the structural equation for T. Adding R_i (RAG retrieval) **reduces ε_T** by grounding the LLM in validated prior extractions.

**Equation 2 — Outcome (downstream causal ML, not in MAS):**
```
Y_i = f(T_i, X_i, U_i)
```
- The downstream ML model estimates `f_hat()` from observational data
- U_i introduces irreducible noise

**Equation 3 — Confounders (exogenous):**
```
X_i = h(market_history_before_published_at_i)
```
- X is determined PRIOR to the news event (no post-treatment bias)
- Measured via dollar bar features at nearest bar before `published_at`

### 2.3 Potential Outcomes Framework

Under the Rubin causal framework:
- `Y_i(t)` = price return that WOULD occur if treatment were `t`
- `Y_i(∅)` = price return under no news treatment (counterfactual baseline)

**Individual Treatment Effect:**
```
ITE_i(t) = Y_i(t) - Y_i(∅)
```

**Average Treatment Effect for treatment class t:**
```
ATE(t) = E[Y(t)] - E[Y(∅)]
```

We only observe the factual outcome `Y_i = Y_i(T_i)` — never the counterfactual.

### 2.4 Causal Identification via Backdoor Adjustment

Under the **backdoor criterion**, if X blocks all backdoor paths from T to Y:
```
E[Y | do(T=t)] = Σ_x E[Y | T=t, X=x] · P(X=x)
```

**Three identification conditions:**
1. **Backdoor criterion:** X blocks all non-causal paths T ← X → Y
2. **No post-treatment conditioning:** X measured BEFORE T (dollar bars at publication time)
3. **Positivity:** P(T=t | X=x) > 0 for all x in support

### 2.5 CATE — The Trading-Relevant Estimand

**Conditional Average Treatment Effect:**
```
τ(t, x) = E[Y | do(T=t), X=x] - E[Y | do(T=∅), X=x]
```

The **trading signal** for a new article:
```
Given: new article z_new at market state x_new
1. MAS extracts:          t_new = g(z_new, x_new, R_new; θ_MAS)
2. Causal ML predicts:    ŷ = τ̂(t_new, x_new)
3. Trading decision:      if |ŷ| > threshold ⟹ trade in direction sign(ŷ)
```

### 2.6 RAG and Causal Identification

**Key insight:** The RAG component (R_i) does NOT introduce new confounding. R_i is a function of Z_i (the article text) and θ_MAS (the index state). It is an **input to g()**, not a separate causal path to Y.

```
R_i = retrieve(embed(Z_i), VS_index)    — R is a function of Z and the index
T_i = g(Z_i, X_i, R_i; θ_MAS)            — R augments extraction quality
```

RAG improves the **quality of T estimation** (reduces ε_T) without violating the DAG structure. The causal identification conditions are unchanged.

**Formal claim:** If `g(Z, X; θ)` satisfies the backdoor criterion for causal identification, then `g(Z, X, R(Z); θ)` also satisfies it, because R is a deterministic function of Z (the instrument) and does not create new backdoor paths to Y.

### 2.7 Why Each MAS Design Decision Serves Causal Identification

| MAS Component | Causal Role | Why It Matters |
|---------------|------------|----------------|
| **Gatekeeper (NER + sklearn)** | Separates T=t from T=∅ | Creates natural control group: non-impactful articles provide counterfactual Y(∅) |
| **Discrete T (taxonomy)** | Finite treatment space | Enables stratified ATE estimation; continuous T requires kernel methods |
| **Mechanism M** | Mediator T → M → Y | Enables mediation analysis: does T affect Y THROUGH the stated mechanism? |
| **Dynamic few-shot (VS)** | ε_T reduction | Grounds LLM in validated prior extractions, reducing extraction variance |
| **Confidence** | Propensity score proxy | Weights observations by extraction quality in causal estimation |
| **Evidence spans** | Grounding verification | Preserves Z → T path in DAG (T was actually extracted FROM Z) |
| **Deterministic validator** | Quality filter | Rejected chains violate ontology — including them introduces measurement error in T |
| **X pre-treatment** | No post-treatment bias | Dollar bars at publication time satisfy backdoor criterion |
| **`rejected_mechanism_raw`** | Ontology expansion | Captures what the LLM produces for whitelist governance |

### 2.8 CausalBERT Connection (Downstream)

The downstream ML pipeline uses causally sufficient embeddings (Veitch et al., 2020):
```
Loss = L_treatment(predict T from φ(Z)) + L_outcome(predict Y from φ(Z), T)
```
- `φ(Z)` = text embedding that isolates causal semantic features
- The MAS provides the **labeled (T, Z, Y) triples** CausalBERT needs for training

### 2.9 Assumptions and Limitations

1. **No unmeasured confounding (strong):** U (order flow, positioning) confounds T→Y. Sensitivity analysis needed.
2. **SUTVA:** Each article's outcome is independent. Violated when multiple articles about the same event cluster.
3. **No interference:** Treatment of article i doesn't affect outcome j. Partially violated by correlated cascades.
4. **Consistency:** Observed Y under T=t equals potential outcome Y(t). Discrete taxonomy with validation ensures well-defined T.
5. **Text-as-confounder:** Z contains confounders not in X (tone, source). CausalBERT addresses this; MAS alone does not.

---

## 3. Agent Architecture — V2 RAG-Augmented

### 3.1 Pipeline Topology

```
silver.news_cleaned  (100% of articles, ~15min refresh)
        │
┌───────▼──────────────────────────────────────────────┐
│  GATEKEEPER (pandas UDF in SDP gold materialized view) │
│                                                        │
│  Stage 1: spaCy EntityRuler — CRYPTO_ENTITY_PATTERNS   │
│    No entity → is_impactful=False, confidence=0.0  ──► SHORT CIRCUIT │
│    Entity found + ner_only_mode=True → conf=1.0    ──► PASS         │
│                                                        │
│  Stage 2: sklearn Pipeline (ner_only_mode=False)       │
│    Features: NER flags(6) + TfidfVectorizer(500)       │
│    Model: LogisticRegression(class_weight="balanced")  │
│    confidence < threshold → ──── SHORT CIRCUIT         │
│    confidence ≥ threshold → ──── PASS                  │
└───────┬──────────────────────────────────────────────┘
        │ is_impactful=True (~5-15%)
        │
┌───────▼──────────────────────────────────────────────┐
│  LANGGRAPH AGENT PIPELINE (Model Serving endpoint)     │
│  Endpoint: mas-news-causal                             │
│                                                        │
│  ┌──────────────────────────┐                          │
│  │  Context Retriever       │  UC function tool         │
│  │  (no-op in V1)           │  → market state X         │
│  └────────────┬─────────────┘                          │
│               │                                        │
│  ┌────────────▼─────────────┐                          │
│  │  Few-Shot Retriever      │  Vector Search query      │
│  │  (V2 RAG)                │  → top-5 accepted chains  │
│  └────────────┬─────────────┘                          │
│               │                                        │
│  ┌────────────▼─────────────┐                          │
│  │  Causality Extractor     │  sonnet-class LLM         │
│  │  + dynamic few-shot      │  → causal chain JSON      │
│  └────────────┬─────────────┘                          │
│               │                                        │
│  ┌────────────▼─────────────┐                          │
│  │  Mechanism Validator     │  Deterministic code        │
│  │  (whitelist lookup)      │  → validation_status       │
│  └────────────┬─────────────┘                          │
└───────────────┼──────────────────────────────────────┘
                │
        ┌───────▼───────┐
        │ gold.news_    │
        │ causal_features │ Materialized View (20 fields)
        └───────────────┘
```

### 3.2 Cost Architecture

| Component | Model | Cost | Volume | Deterministic? |
|-----------|-------|------|--------|---------------|
| Gatekeeper Stage 1 | spaCy EntityRuler | ~$0 | 100% | ✅ Yes |
| Gatekeeper Stage 2 | sklearn LogisticRegression | ~$0 | ~15-40% (pass Stage 1) | ✅ Yes |
| Context Retriever | UC function (no-op V1) | ~$0 | ~5-15% | ✅ Yes |
| Few-Shot Retriever | Databricks Vector Search | ~$0.001/query | ~5-15% | ✅ Yes |
| Causality Extractor | sonnet-class LLM | ~$0.03/article | ~5-15% | ❌ No |
| Mechanism Validator | Python whitelist lookup | ~$0 | ~5-15% | ✅ Yes |

**Only one stochastic component:** the Causality Extractor. Everything else is deterministic. The RAG retriever provides dynamic few-shot exemplars that reduce the Extractor's output variance.

### 3.3 Mosaic AI Deployment

- **Agent definition:** `ResponsesAgent` subclass wrapping LangGraph `CompiledStateGraph`
- **Tools as UC functions:**
  - `fetch_market_context(published_at)` — reads `gold.dollar_bars` (no-op V1)
  - `fetch_mechanism_whitelist()` — reads `gold.mechanism_whitelist` as JSON
  - `retrieve_few_shot_chains(headline)` — queries Vector Search index (V2)
- **Endpoint:** Serverless Model Serving endpoint `mas-news-causal`
- **SDP invocation:** `ai_query('mas-news-causal', ...)` inside the gold materialized view

---

## 4. Vector Search — Dynamic Few-Shot Retrieval

### 4.1 Architecture

The Causality Extractor receives **dynamic few-shot examples** retrieved from a Vector Search index over previously accepted causal chains.

```
New article headline: "Mt. Gox trustee confirms July 2026 BTC distribution"
        │
        ▼  embed via databricks-gte-large-en
┌───────────────────────────────────────────────────┐
│  Vector Search Index:                              │
│  trading-signal-gen-{env}.03_gold.causal_chains_idx   │
│                                                    │
│  Source: gold.news_causal_features                 │
│  Filter: validation_status = 'accepted'            │
│  Embedding column: trigger_event                   │
│  Sync: TRIGGERED (after each MV refresh)           │
│  Endpoint: mas-few-shot (STANDARD, <100ms)         │
└───────────────────────┬───────────────────────────┘
                        │ top-5 results
                        ▼
┌──────────────────────────────────────────────┐
│  Retrieved few-shot exemplars:                │
│                                               │
│  1. trigger_event: "Mt. Gox announces..."     │
│     treatment_class: Supply.unlock_distribution│
│     mechanism: sell_pressure                   │
│     expected_direction: bearish                │
│                                               │
│  2. trigger_event: "Grayscale GBTC..."        │
│     treatment_class: Supply.vesting_unlock     │
│     mechanism: sell_pressure                   │
│     expected_direction: bearish                │
│                                               │
│  ... (3 more)                                 │
└──────────────────────────────────────────────┘
        │
        ▼  injected into Extractor system prompt
    Causality Extractor LLM produces grounded extraction
```

### 4.2 Cold Start Strategy

The bootstrapping problem: VS index needs accepted chains, but accepted chains come from the MAS.

**Phase 1 (cold start):** The Extractor uses a static few-shot prompt with 3-5 hardcoded examples covering different L1 categories. No Vector Search queries. The few-shot retriever returns empty.

**Phase 2 (V2 warm-up, ~200+ accepted chains):** Activate VS retrieval. The prompt switches from static to dynamic few-shot:
```python
static_examples = STATIC_FEW_SHOT_EXAMPLES          # 3-5 hardcoded
dynamic_examples = retrieve_few_shot_chains(headline) # 0-5 from VS

# Combine: dynamic examples take priority, fall back to static
few_shot = (dynamic_examples or []) + static_examples
few_shot = few_shot[:5]  # cap at 5 total
```

**Phase 3 (steady state, 500+ accepted chains):** Dynamic retrieval fully dominates. Static examples are kept as fallback only.

### 4.3 Databricks Implementation

**Vector Search Endpoint:**
```python
# Created once, reused
create_or_update_vs_endpoint("mas-few-shot", endpoint_type="STANDARD")
```

**Delta Sync Index:**
```python
create_or_update_vs_index(
    name="{catalog}.03_gold.causal_chains_idx",
    endpoint_name="mas-few-shot",
    primary_key="story_id",
    delta_sync_index_spec={
        "source_table": "{catalog}.03_gold.news_causal_features",
        "embedding_source_columns": [
            {
                "name": "trigger_event",
                "embedding_model_endpoint_name": "databricks-gte-large-en"
            }
        ],
        "pipeline_type": "TRIGGERED",
        "columns_to_sync": [
            "story_id", "treatment_class", "mechanism",
            "expected_direction", "confidence",
            "causal_chain_json", "validation_status"
        ]
    }
)
```

**Query at inference time:**
```python
results = query_vs_index(
    index_name="{catalog}.03_gold.causal_chains_idx",
    columns=["treatment_class", "mechanism", "expected_direction",
             "causal_chain_json"],
    query_text=current_article_headline,
    num_results=5,
    filters_json='{"validation_status": ["accepted"]}'
)
```

### 4.4 Why This RAG Architecture is Reliable

| Risk | Mitigation |
|------|-----------|
| Poison loop — bad extractions get indexed and influence future extractions | Filter `validation_status = 'accepted'` ensures only whitelist-validated chains enter the index |
| VS latency degrades pipeline | STANDARD endpoint < 100ms; Extractor LLM call is 2-5s anyway (5% overhead) |
| VS endpoint cost | Only queried for ~5% of articles (post-gatekeeper); < $1/month at current volume |
| Stale few-shot examples | TRIGGERED sync after each MV refresh keeps index within 15min of freshness |
| Retrieval returns irrelevant examples | Extractor prompt instructs: "Use retrieved examples as reference, but extract based on THIS article's actual content" |

---

## 5. Gatekeeper — NER + sklearn Classifier

### 5.1 Stage 1: Crypto Entity Lexicon

**File:** `src/trading_databricks/agents/gatekeeper.py`

Uses `spacy.blank("en")` + `EntityRuler` — no model download. Pattern matching only.

```python
CRYPTO_ENTITY_PATTERNS: list[dict] = [
    # ── CRYPTO_ASSET: Primary tracked assets ──
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
    # ── CRYPTO_EXCHANGE ──
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "binance"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "coinbase"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "kraken"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "bybit"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"TEXT": "OKX"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "ftx"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "gemini"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "bitstamp"}]},
    # ── CRYPTO_STABLECOIN ──
    {"label": "CRYPTO_STABLECOIN", "pattern": [{"TEXT": "USDC"}]},
    {"label": "CRYPTO_STABLECOIN", "pattern": [{"TEXT": "USDT"}]},
    {"label": "CRYPTO_STABLECOIN", "pattern": [{"LOWER": "tether"}]},
    {"label": "CRYPTO_STABLECOIN", "pattern": [{"LOWER": "stablecoin"}]},
    {"label": "CRYPTO_STABLECOIN", "pattern": [{"LOWER": "dai"}, {"LOWER": "stablecoin"}]},
    # ── CRYPTO_PROTOCOL ──
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "uniswap"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "aave"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "makerdao"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "chainlink"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "arbitrum"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "compound"}, {"LOWER": "finance"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"TEXT": "COMP"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"TEXT": "OP"}, {"LOWER": "token"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "lido"}, {"LOWER": "finance"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "steth"}]},
    # ── CRYPTO_CONCEPT ──
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "blockchain"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "defi"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "nft"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "halving"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "staking"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "tokenomics"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "layer"}, {"TEXT": "2"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"TEXT": "L2"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "smart"}, {"LOWER": "contract"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "proof"}, {"LOWER": "of"}, {"LOWER": "work"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "proof"}, {"LOWER": "of"}, {"LOWER": "stake"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "decentralized"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "on-chain"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "whitepaper"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "bitcoin"}, {"LOWER": "mining"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "crypto"}, {"LOWER": "mining"}]},
]
```

**50 patterns** across 5 categories. Ambiguous standalone patterns (`mining`, `compound`, `optimism`, `lido`, `SOL`, `DAI`) replaced with compound alternatives that eliminate false positive vectors.

### 5.2 Feature Extraction

```python
def build_nlp() -> spacy.Language:
    """Build blank spaCy pipeline with EntityRuler. No model download needed."""
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(CRYPTO_ENTITY_PATTERNS)
    return nlp

def extract_ner_features(doc: spacy.tokens.Doc) -> dict[str, bool | int]:
    """Boolean entity category flags + total entity count."""
    labels = {ent.label_ for ent in doc.ents}
    return {
        "has_crypto_asset":  "CRYPTO_ASSET" in labels,
        "has_exchange":      "CRYPTO_EXCHANGE" in labels,
        "has_stablecoin":    "CRYPTO_STABLECOIN" in labels,
        "has_protocol":      "CRYPTO_PROTOCOL" in labels,
        "has_concept":       "CRYPTO_CONCEPT" in labels,
        "entity_count":      len(doc.ents),
    }
```

### 5.3 Stage 2: sklearn Classifier (V1.5)

```
headline (str)  ──► ColumnTransformer ──► LogisticRegression ──► P(is_impactful)
                        ├─ TfidfVectorizer(max_features=500, ngram_range=(1,2))
NER features (6)──►     └─ passthrough (6 dense values)
```

Uses `sklearn.compose.ColumnTransformer` to handle heterogeneous inputs. `class_weight="balanced"` compensates for 95/5 class imbalance.

Model stored: `models:/news-gatekeeper/Production` in MLflow.

### 5.4 pandas UDF — Safe Serialization Pattern

`build_nlp()` and model loading INSIDE the UDF body (not in closure):

```python
@pandas_udf(return_schema)
def _gate(headlines: pd.Series, bodies: pd.Series) -> pd.DataFrame:
    # Built once per partition batch on executor — no serialization risk
    nlp = build_nlp()
    model = None if ner_only_mode else mlflow.sklearn.load_model(model_uri)
    ...
```

### 5.5 GatekeeperConfig

```python
@dataclass(frozen=True)
class GatekeeperConfig:
    model_uri: str = "models:/news-gatekeeper/Production"
    confidence_threshold: float = 0.5
    ner_only_mode: bool = True    # V1: NER only, no MLflow dependency
    text_window: int = 500        # chars of body to concat with headline
```

---

## 6. Causality Extractor — sonnet-class LLM

### 6.1 Causal Chain JSON Schema

```json
{
  "trigger_event": "Mt. Gox trustee announces $9B BTC distribution schedule",
  "mechanism": "sell_pressure",
  "asset": "BTC",
  "expected_direction": "bearish",
  "confidence": 0.85,
  "evidence_spans": [
    "Mt. Gox trustee confirmed the timeline for distributing approximately 142,000 BTC",
    "creditors will receive their Bitcoin holdings starting July 2026"
  ],
  "reasoning": "Large-scale distribution of long-dormant BTC creates sell pressure."
}
```

| Field | Type | Constraint |
|-------|------|-----------|
| `trigger_event` | string | max 200 chars, grounded in article text |
| `mechanism` | string | must exist in `gold.mechanism_whitelist` |
| `asset` | string | enum: `"BTC"`, `"ETH"`, `"BOTH"` |
| `expected_direction` | string | enum: `"bullish"`, `"bearish"`, `"volatile"` |
| `confidence` | float | [0.0, 1.0] |
| `evidence_spans` | array[string] | 1–3 spans, **normalized substrings** of article body |
| `reasoning` | string | max 500 chars, CoT trace |

### 6.2 System Prompt Template

```
You are a financial news causality extractor for BTC/ETH markets.

TASK: Extract a single-hop causal chain from the article.

TREATMENT TAXONOMY (classify the event into one of these):
{taxonomy_injection}   ← all 34 taxonomy entries

MECHANISM WHITELIST (the mechanism MUST be one of these):
{whitelist_injection}  ← all 40 whitelist entries, grouped by L1

SIMILAR PAST EXTRACTIONS (reference, do not copy blindly):
{few_shot_examples}    ← 0-5 dynamic from VS + 0-5 static fallback

RULES:
1. Copy evidence_spans VERBATIM from the article body — exact character sequences.
2. mechanism must exactly match a whitelist entry for the chosen L1 category.
3. If the article does not describe a clear causal event for BTC/ETH, respond
   with {"is_impactful": false}.
4. Use retrieved examples as reference patterns, but extract based on THIS
   article's actual content.
```

---

## 7. Mechanism Validator — Deterministic

### 7.1 Validator Logic

```
1. Parse treatment_class → extract L1  (e.g., "Supply.halving" → "Supply")
2. Lookup (L1, mechanism) in gold.mechanism_whitelist
3. NOT FOUND       → validation_status = "rejected_mechanism"
                     rejected_mechanism_raw = extracted_mechanism
4. direction ∉ valid_directions → validation_status = "rejected_direction"
5. any casefold(strip(evidence_span)) ∉ substrings(casefold(strip(article.body)))
                   → validation_status = "rejected_evidence"
6. all pass        → validation_status = "accepted"
```

**Evidence validation uses normalized matching** (casefold + strip whitespace). LLMs frequently alter casing or whitespace when extracting spans. Normalized matching prevents false rejections while catching fabricated evidence.

### 7.2 `rejected_mechanism_raw` — Governance Field

When `validation_status = "rejected_mechanism"`, the raw LLM output is captured in `rejected_mechanism_raw`. This enables:
- Monitoring what the LLM produces that isn't in the whitelist
- SQL queries to discover clusters of rejected mechanisms for whitelist expansion
- V2: VS index over rejected mechanisms for automated governance suggestions

---

## 8. Treatment Taxonomy — `gold.treatment_taxonomy`

### Schema
```python
TREATMENT_TAXONOMY_SCHEMA = StructType([
    StructField("l1_category",    StringType(), nullable=False),
    StructField("l2_subcategory", StringType(), nullable=False),
    StructField("description",    StringType(), nullable=False),
    StructField("example",        StringType(), nullable=False),
])
```

**34 rows** across 8 L1 categories. PK: `(l1_category, l2_subcategory)`.

| L1 Category | L2 Count | L2 Subcategories |
|-------------|----------|------------------|
| **Regulation** | 5 | etf_ruling, legislation, enforcement_action, cbdc_policy, tax_policy |
| **Security** | 4 | exchange_hack, protocol_exploit, rug_pull, wallet_compromise |
| **Adoption** | 4 | institutional_buy, corporate_treasury, payment_integration, country_adoption |
| **Macro** | 5 | rate_decision, inflation_data, employment_data, gdp, geopolitical |
| **Exchange** | 4 | listing, delisting, insolvency, reserve_proof |
| **Supply** | 4 | halving, unlock_distribution, mining_difficulty, burn_event |
| **DeFi** | 4 | tvl_shift, protocol_upgrade, stablecoin_depeg, yield_event |
| **Sentiment** | 4 | whale_movement, influencer_signal, fund_flow, options_expiry |

**L1 categories frozen** in V1. New L2 subcategories require: (1) insert into taxonomy, (2) add whitelist entries, (3) update Extractor prompt.

---

## 9. Mechanism Whitelist — `gold.mechanism_whitelist`

### Schema
```python
MECHANISM_WHITELIST_SCHEMA = StructType([
    StructField("treatment_l1",     StringType(),            nullable=False),
    StructField("mechanism",        StringType(),            nullable=False),
    StructField("valid_directions", ArrayType(StringType()), nullable=False),
    StructField("description",      StringType(),            nullable=False),
])
```

**40 entries** (5 per L1). PK: `(treatment_l1, mechanism)`.

| L1 | Mechanisms |
|----|-----------|
| Regulation | regulatory_clarity, regulatory_restriction, institutional_access, compliance_burden, market_legitimization |
| Security | sell_pressure, trust_erosion, market_exit, contagion_spread, user_withdrawal |
| Adoption | demand_increase, institutional_inflow, network_expansion, utility_unlock, strategic_accumulation |
| Macro | risk_on, risk_off, liquidity_expansion, liquidity_contraction, inflation_hedge |
| Exchange | liquidity_increase, trust_erosion, market_access_expansion, trading_suspension, counterparty_risk |
| Supply | sell_pressure, supply_reduction, miner_capitulation, scarcity_signal, vesting_unlock |
| DeFi | contagion_risk, yield_migration, liquidity_drain, protocol_confidence, stablecoin_risk |
| Sentiment | momentum_signal, liquidation_risk, fear_signal, greed_signal, narrative_shift |

Cross-L1 mechanisms (`sell_pressure`, `trust_erosion`) intentionally appear under multiple L1s.

---

## 10. Output Schema — `gold.news_causal_features` (20 fields)

```python
CAUSAL_FEATURES_SCHEMA = StructType([
    # ── Always populated (5 NOT NULL fields) ──
    StructField("story_id",                 StringType(),            nullable=False),  # 1  PK, FK
    StructField("published_at",             TimestampType(),         nullable=False),  # 2
    StructField("is_impactful",             BooleanType(),           nullable=False),  # 3  Gatekeeper decision
    StructField("gatekeeper_confidence",    DoubleType(),            nullable=True),   # 4  NER/sklearn confidence
    StructField("agent_model_version",      StringType(),            nullable=False),  # 5  MLflow lineage
    StructField("processed_at",             TimestampType(),         nullable=False),  # 6

    # ── NULL when is_impactful=False ──
    StructField("treatment_class",          StringType(),            nullable=True),   # 7  "L1.l2_subcategory"
    StructField("trigger_event",            StringType(),            nullable=True),   # 8
    StructField("mechanism",                StringType(),            nullable=True),   # 9
    StructField("asset",                    StringType(),            nullable=True),   # 10 BTC|ETH|BOTH
    StructField("expected_direction",       StringType(),            nullable=True),   # 11 bullish|bearish|volatile
    StructField("confidence",               DoubleType(),            nullable=True),   # 12 Agent confidence [0,1]
    StructField("causal_chain_json",        StringType(),            nullable=True),   # 13 Full JSON chain
    StructField("evidence_spans",           ArrayType(StringType()), nullable=True),   # 14 1-3 grounding spans
    StructField("validation_status",        StringType(),            nullable=True),   # 15 accepted|rejected_*
    StructField("rejected_mechanism_raw",   StringType(),            nullable=True),   # 16 LLM output before rejection
    StructField("retrieval_story_ids",      ArrayType(StringType()), nullable=True),   # 17 VS few-shot source IDs

    # ── Market context (NULL until TRD-5) ──
    StructField("market_vol_at_publish",    DoubleType(),            nullable=True),   # 18
    StructField("market_trend_at_publish",  StringType(),            nullable=True),   # 19
    StructField("dollar_bar_close",         DecimalType(18, 8),      nullable=True),   # 20
])
```

### Field Contracts

| # | Field | Type | Constraint |
|---|-------|------|-----------|
| 1 | `story_id` | STRING, NOT NULL | FK → `silver.news_cleaned.story_id` |
| 2 | `published_at` | TIMESTAMP, NOT NULL | From silver layer |
| 3 | `is_impactful` | BOOLEAN, NOT NULL | Gatekeeper decision |
| 4 | `gatekeeper_confidence` | DOUBLE, nullable | 0.0 (rejected) / 1.0 (NER mode) / [0,1] (sklearn mode) |
| 5 | `agent_model_version` | STRING, NOT NULL | MLflow model version for lineage + A/B tests |
| 6 | `processed_at` | TIMESTAMP, NOT NULL | MAS processing timestamp |
| 7 | `treatment_class` | STRING, nullable | `"L1.l2_subcategory"` from taxonomy |
| 8 | `trigger_event` | STRING, nullable | Max 200 chars, grounded in article |
| 9 | `mechanism` | STRING, nullable | Must exist in whitelist |
| 10 | `asset` | STRING, nullable | `"BTC"` \| `"ETH"` \| `"BOTH"` |
| 11 | `expected_direction` | STRING, nullable | `"bullish"` \| `"bearish"` \| `"volatile"` |
| 12 | `confidence` | DOUBLE, nullable | LLM self-assessed [0.0, 1.0] |
| 13 | `causal_chain_json` | STRING, nullable | Full extraction JSON |
| 14 | `evidence_spans` | ARRAY<STRING>, nullable | 1-3 spans, normalized substrings |
| 15 | `validation_status` | STRING, nullable | `accepted` / `rejected_mechanism` / `rejected_direction` / `rejected_evidence` |
| 16 | `rejected_mechanism_raw` | STRING, nullable | Raw LLM mechanism output (only when rejected) |
| 17 | `retrieval_story_ids` | ARRAY<STRING>, nullable | story_ids of VS few-shot results (V2 governance/lineage) |
| 18 | `market_vol_at_publish` | DOUBLE, nullable | Dollar bar volatility (NULL until TRD-5) |
| 19 | `market_trend_at_publish` | STRING, nullable | `uptrend`/`downtrend`/`sideways` (NULL until TRD-5) |
| 20 | `dollar_bar_close` | DECIMAL(18,8), nullable | BTC price at nearest bar (NULL until TRD-5) |

### Nullability Rules
- `is_impactful=False` → fields 7-17 are ALL NULL
- `is_impactful=True, validation_status=rejected_mechanism` → `rejected_mechanism_raw` populated
- `is_impactful=True, validation_status=accepted` → fields 7-14 all populated, field 16 NULL
- Fields 18-20 are ALWAYS NULL until TRD-5 is implemented

### SDP Construct
Materialized View (full recompute):
- Agent outputs non-deterministic — MV allows re-processing with updated models
- Consistent with gold layer pattern (`gold.dollar_bars`, `gold.volume_bars`)

---

## 11. Downstream ML Handshake

### Feature Engineering Path

```
gold.news_causal_features (20 fields, this spec)
        ↓
silver.features_news (feature engineering transforms)
    - treatment_class → one-hot encoding by L1
    - mechanism → categorical feature
    - confidence, gatekeeper_confidence → continuous features
    - expected_direction → ordinal encoding
    - validation_status → filter to accepted | include as quality feature
        ↓
silver.training_datasets (PIT-safe temporal join with bar/microstructure features)
        ↓
CPCV model training (AFML Ch. 12)
```

### CausalBERT Training Data

The MAS provides labeled `(T, Z, Y)` triples:
- **T** = `treatment_class` (from accepted chains)
- **Z** = article text (from `silver.news_cleaned`)
- **Y** = forward return (from `gold.dollar_bars` post-publication)

CausalBERT Loss:
```
L = L_treatment(T | φ(Z)) + L_outcome(Y | φ(Z), T)
```

---

## 12. LangGraph State and Topology

### State TypedDict

```python
class MASState(TypedDict):
    # ── Input ──
    story_id: str
    headline: str
    body: str
    published_at: str
    # ── Context Retriever output ──
    market_context: dict | None           # None in V1
    # ── Few-Shot Retriever output ──
    few_shot_examples: list[dict]          # [] when cold-start
    retrieval_story_ids: list[str]         # [] when cold-start
    # ── Causality Extractor output ──
    extraction: dict | None
    # ── Mechanism Validator output ──
    validation_status: str | None
    rejected_mechanism_raw: str | None
    # ── Final output ──
    output_row: dict
```

### Graph Definition

```python
graph = StateGraph(MASState)

graph.add_node("context_retriever", context_retriever)
graph.add_node("few_shot_retriever", few_shot_retriever)
graph.add_node("causality_extractor", causality_extractor)
graph.add_node("mechanism_validator", mechanism_validator)

graph.set_entry_point("context_retriever")
graph.add_edge("context_retriever", "few_shot_retriever")
graph.add_edge("few_shot_retriever", "causality_extractor")
graph.add_edge("causality_extractor", "mechanism_validator")
graph.add_edge("mechanism_validator", END)

compiled = graph.compile()
```

**Gatekeeper is NOT in the LangGraph.** It runs as a pandas UDF in the SDP materialized view, BEFORE `ai_query()`. Only impactful articles enter the graph.

---

## 13. Evaluation Strategy

### Dataset
- `bronze.news_agent_eval_dataset` (500 rows) for development
- Human-labeled gold set (~50-100 articles) with ground truth

### Metrics

| Metric | Target | Component |
|--------|--------|-----------|
| Gatekeeper precision | >0.80 | NER + sklearn |
| Gatekeeper recall | >0.95 | NER + sklearn (recall > precision: missing events is worse) |
| Treatment class L1 accuracy | >0.70 | Causality Extractor |
| Evidence grounding rate | >0.90 | Normalized substring match |
| Validation pass rate | >0.85 | Mechanism whitelist acceptance |
| VS retrieval relevance | >0.60 | Retrieved exemplars share L1 with correct extraction |

### Eval Pipeline
- `mlflow.genai.evaluate()` with custom scorers
- Pre-deployment gate before promoting agent model versions
- Results stored in MLflow experiments

---

## 14. Data Flow — Complete Pipeline Sequence

```
[A] news-scraper-aws → S3 NDJSON
                          ↓ (Auto Loader, every 15min)
[B] bronze.news_raw → silver.news_cleaned
                          ↓ (SDP Materialized View)
[C] apply_gatekeeper(silver.news_cleaned, config)       ← pandas UDF, 100% volume
                          ↓ is_impactful=True (~5-15%)
[D] ai_query('mas-news-causal', article_json)            ← LangGraph endpoint
    ┌─ context_retriever (no-op V1)
    ├─ few_shot_retriever (VS query V2)
    ├─ causality_extractor (sonnet LLM + dynamic few-shot)
    └─ mechanism_validator (deterministic whitelist)
                          ↓
[E] gold.news_causal_features (20-field MV)
                          ↓ (Delta Sync TRIGGERED)
[F] gold.causal_chains_idx (VS index, V2)                ← feeds back to step [D]
                          ↓
[G] silver.features_news → silver.training_datasets
                          ↓
[H] CPCV model training → MLflow Model Registry
                          ↓
[I] Inference pipeline → silver.signals → silver.positions
```

---

## 15. Infrastructure & Resources

### Databricks Resources

| Resource | Type | Catalog |
|----------|------|---------|
| `gold.treatment_taxonomy` | Delta table (SDP `@dp.table`) | `trading-signal-gen-{env}` |
| `gold.mechanism_whitelist` | Delta table (SDP `@dp.table`) | `trading-signal-gen-{env}` |
| `gold.news_causal_features` | Materialized View (SDP `@dp.materialized_view`) | `trading-signal-gen-{env}` |
| `gold.causal_chains_idx` | VS Delta Sync Index | `trading-signal-gen-{env}` |
| `mas-few-shot` | VS Endpoint (STANDARD) | Workspace-level |
| `mas-news-causal` | Model Serving Endpoint (Serverless) | Workspace-level |
| `models:/news-gatekeeper` | MLflow Model | Model Registry |

### Dependencies (pyproject.toml additions)

```toml
[project.optional-dependencies]
agents = [
    "spacy>=3.7,<4",
    "langgraph>=0.3,<1",
    "langchain-databricks>=0.1,<1",
    "mlflow>=3.0.1",
]
```

---

## 16. Task Breakdown

### Phase 1: Deterministic Foundation

| Task | File | Description |
|------|------|-------------|
| 1.1 | `agents/taxonomy.py` | Seed data + schemas (34 taxonomy + 40 whitelist) |
| 1.2 | `news/schemas.py` | Add `CAUSAL_FEATURES_SCHEMA` (20 fields) |
| 1.3 | `common/config.py` | Add `GatekeeperConfig`, `MASConfig` |
| 1.4 | `agents/gatekeeper.py` | `CRYPTO_ENTITY_PATTERNS`, `build_nlp()`, `extract_ner_features()` |
| 1.5 | `tests/agents/test_gatekeeper.py` | NER pattern unit tests |
| 1.6 | `tests/agents/test_taxonomy.py` | Seed data integrity assertions |

### Phase 2: Agent Graph

| Task | File | Description |
|------|------|-------------|
| 2.1 | `agents/graph.py` | `MASState`, `StateGraph`, compile |
| 2.2 | `agents/nodes.py` | context_retriever, few_shot_retriever, causality_extractor, mechanism_validator |
| 2.3 | `agents/prompts.py` | System prompts with taxonomy/whitelist injection + few-shot slots |
| 2.4 | `agents/tools.py` | UC function tools: market context, whitelist, VS retrieval |
| 2.5 | `agents/serve.py` | `ResponsesAgent` wrapper |
| 2.6 | `tests/agents/test_validator.py` | Validator acceptance/rejection unit tests |
| 2.7 | `tests/agents/test_graph.py` | Graph routing tests with mocked LLM |

### Phase 3: SDP + VS Integration

| Task | File | Description |
|------|------|-------------|
| 3.1 | `news/gold.py` | `apply_gatekeeper()` + `causal_features()` transforms |
| 3.2 | `news/pipeline.py` | Wire `@dp.materialized_view` for gold.news_causal_features |
| 3.3 | `agents/pipeline.py` | Wire `@dp.table` for taxonomy + whitelist |
| 3.4 | `resources/` | Pipeline YAML definitions |
| 3.5 | Deploy VS endpoint + index | `mas-few-shot` endpoint + `causal_chains_idx` |

### Phase 4: Evaluation & Deployment

| Task | File | Description |
|------|------|-------------|
| 4.1 | `tests/agents/test_eval.py` | MLflow eval pipeline with custom scorers |
| 4.2 | Human labeling | Label 50-100 articles for gold eval set |
| 4.3 | Deploy endpoint | Register agent model, create `mas-news-causal` endpoint |
| 4.4 | E2E smoke test | Full pipeline refresh on dev catalog |
| 4.5 | Cost profiling | Process 500 articles, measure cost per path |

---

## 17. Verification Checklist

### Seed Data Integrity
- [ ] `len(TREATMENT_TAXONOMY_SEED) == 34`
- [ ] `len(MECHANISM_WHITELIST_SEED) == 40`
- [ ] Every `(treatment_l1, mechanism)` pair unique in whitelist
- [ ] Every `treatment_l1` exists as `l1_category` in taxonomy
- [ ] Every `valid_directions` is non-empty subset of `{"bullish", "bearish", "volatile"}`

### Schema Contract
- [ ] `len(CAUSAL_FEATURES_SCHEMA.fields) == 20`
- [ ] Non-nullable fields exactly: `story_id`, `published_at`, `is_impactful`, `agent_model_version`, `processed_at` (5 fields)
- [ ] `gatekeeper_confidence` at index 3 (after `is_impactful`)
- [ ] `rejected_mechanism_raw` at index 15 (after `validation_status`)
- [ ] `retrieval_story_ids` at index 16 (after `rejected_mechanism_raw`)
- [ ] `dollar_bar_close` is `DecimalType(18, 8)`, not `DoubleType`

### NER Patterns
- [ ] `len(CRYPTO_ENTITY_PATTERNS) == 50`
- [ ] No standalone patterns for: `mining`, `compound`, `optimism`, `lido`, `SOL`, `DAI`
- [ ] Compound patterns present: `bitcoin mining`, `crypto mining`, `compound finance`, `lido finance`, `dai stablecoin`

### Gatekeeper
- [ ] `build_nlp()` called INSIDE UDF body (not in closure)
- [ ] spaCy model serialization test: `pickle.dumps(build_nlp())` passes
- [ ] BTC headline → `is_impactful=True`; HK biotech → `is_impactful=False`

### Validator
- [ ] Accepted chain passes all 3 checks (mechanism, direction, evidence)
- [ ] Unknown mechanism → `rejected_mechanism`, `rejected_mechanism_raw` populated
- [ ] Invalid direction → `rejected_direction`
- [ ] Fabricated evidence → `rejected_evidence`
- [ ] Evidence validation uses `casefold(strip(...))` normalized matching

### Vector Search
- [ ] VS endpoint `mas-few-shot` created (STANDARD type)
- [ ] Delta Sync index filters to `validation_status = 'accepted'`
- [ ] Cold-start fallback returns empty list (no errors)
- [ ] `retrieval_story_ids` populated when VS results available

### End-to-End
- [ ] `gold.news_causal_features` schema matches 20-field contract exactly
- [ ] Non-impactful rows have NULL for fields 7-17
- [ ] Impactful + accepted rows have all fields 7-14 populated
- [ ] Pipeline runs on dev catalog without errors
- [ ] MLflow eval metrics meet targets

### Post-Deploy SQL
```sql
SELECT COUNT(*) FROM gold.treatment_taxonomy;    -- 34
SELECT COUNT(*) FROM gold.mechanism_whitelist;   -- 40
SELECT COUNT(*) FROM gold.news_causal_features;  -- > 0
SELECT validation_status, COUNT(*) FROM gold.news_causal_features
  WHERE is_impactful = TRUE GROUP BY 1;          -- monitor acceptance rate
SELECT rejected_mechanism_raw, COUNT(*) FROM gold.news_causal_features
  WHERE validation_status = 'rejected_mechanism'
  GROUP BY 1 ORDER BY 2 DESC LIMIT 20;           -- whitelist expansion candidates
```
