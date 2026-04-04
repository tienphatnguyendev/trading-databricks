# MAS Gatekeeper — NER + sklearn Classifier Design

**Date:** 2026-04-04  
**Replaces:** LLM-based Deep Impact Assessor (haiku-class model) in `2026-04-03-mas-causality-layer-redesign.md`  
**Status:** Design only — no implementation yet  
**Blocks:** Task 2.2 (agent nodes), Task 3.2 (gold transform), Task 3.1 (config)  
**Schema impact:** `CAUSAL_FEATURES_SCHEMA` grows from 17 → 18 fields

---

## Context

The 2026-04-03 redesign plan places a haiku-class LLM as the "Deep Impact Assessor" gatekeeper node inside the LangGraph agent graph. It runs on **100% of articles** to decide if an article is relevant to BTC/ETH crypto markets.

**Problem:** ~95% of TradingView financial wire articles have zero crypto relevance (HK biotech, Italian fashion, UAE credit bureau). Paying per-token LLM cost to make a binary relevance decision on every article is architecturally wrong — this is exactly what a lightweight NLP classifier does better, faster, and at near-zero marginal cost.

**Decision:** Replace the gatekeeper LLM with a two-stage NLP pipeline running as a pandas UDF in the SDP gold materialized view:

1. **Stage 1 — NER entity matching:** spaCy EntityRuler with a curated crypto lexicon. Zero training required. Deterministic. Sub-millisecond per article. Rejects clearly irrelevant articles before any ML inference.
2. **Stage 2 — sklearn binary classifier:** TF-IDF(headline) + NER entity flags → LogisticRegression. Handles borderline cases. Requires labeled training data (V1.5).

**V1 ships with `ner_only_mode=True`** — Stage 1 only, no training data needed, no MLflow model dependency. Stage 2 added in V1.5 after labeling `bronze.news_agent_eval_dataset`.

---

## Architecture

```
silver.news_cleaned  (100% of articles)
        ↓
  apply_gatekeeper(df, config=GatekeeperConfig)   ← pandas UDF in news/gold.py
  ┌──────────────────────────────────────────────────────────┐
  │  text = headline + " " + body[:config.text_window]       │
  │                                                          │
  │  Stage 1: spaCy EntityRuler (CRYPTO_ENTITY_PATTERNS)     │
  │    No entity found → is_impactful=False, conf=0.0  EXIT  │
  │    Entity found + ner_only_mode=True → conf=1.0    EXIT  │
  │                                                          │
  │  Stage 2: sklearn Pipeline (ner_only_mode=False)         │
  │    Features: NER flags (6) + TF-IDF headline (500)       │
  │    Model: LogisticRegression(class_weight="balanced")    │
  │    confidence ≥ threshold → is_impactful=True            │
  └──────────────────────────────────────────────────────────┘
        ↓ is_impactful=True (~5-20% depending on mode)
  ai_query('mas-news-causal', ...)
  [LangGraph: Context Retriever → Causality Extractor → Mechanism Validator]
  (Deep Impact Assessor node REMOVED from LangGraph)
        ↓
  gold.news_causal_features (18 fields)
```

**LangGraph simplification:** 4 nodes → 3 nodes. Remove `impact_assessor`. Remaining: `context_retriever` (no-op V1), `causality_extractor` (sonnet-class), `mechanism_validator` (deterministic).

**Cost profile:**
| Component | Cost | Volume |
|-----------|------|--------|
| NER Stage 1 | ~$0 | 100% |
| sklearn Stage 2 | ~$0 | ~20-50% (passes Stage 1) |
| LangGraph (sonnet) | LLM cost | ~5% (passes both) |

---

## Stage 1: NER Entity Lexicon

**File:** `src/trading_databricks/agents/gatekeeper.py`

Uses `spacy.blank("en")` + `EntityRuler` — **no spaCy model download required**. The EntityRuler does pattern matching against our crypto-specific lexicon; standard NER categories (PERSON, ORG, GPE) are not used.

### Entity Categories and Patterns

```python
CRYPTO_ENTITY_PATTERNS: list[dict] = [
    # CRYPTO_ASSET — primary assets tracked
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
    {"label": "CRYPTO_ASSET", "pattern": [{"TEXT": "SOL"}]},
    {"label": "CRYPTO_ASSET", "pattern": [{"LOWER": "solana"}]},
    # CRYPTO_EXCHANGE
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "binance"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "coinbase"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "kraken"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "bybit"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"TEXT": "OKX"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "ftx"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "gemini"}]},
    {"label": "CRYPTO_EXCHANGE", "pattern": [{"LOWER": "bitstamp"}]},
    # CRYPTO_STABLECOIN
    {"label": "CRYPTO_STABLECOIN", "pattern": [{"TEXT": "USDC"}]},
    {"label": "CRYPTO_STABLECOIN", "pattern": [{"TEXT": "USDT"}]},
    {"label": "CRYPTO_STABLECOIN", "pattern": [{"LOWER": "tether"}]},
    {"label": "CRYPTO_STABLECOIN", "pattern": [{"TEXT": "DAI"}]},
    {"label": "CRYPTO_STABLECOIN", "pattern": [{"LOWER": "stablecoin"}]},
    # CRYPTO_PROTOCOL
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "uniswap"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "aave"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "compound"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "makerdao"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "chainlink"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "lido"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "arbitrum"}]},
    {"label": "CRYPTO_PROTOCOL", "pattern": [{"LOWER": "optimism"}]},
    # CRYPTO_CONCEPT
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "blockchain"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "defi"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "nft"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "halving"}]},
    {"label": "CRYPTO_CONCEPT", "pattern": [{"LOWER": "mining"}]},
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
]
```

### Feature Extraction

```python
def build_nlp() -> spacy.Language:
    """Build spaCy blank pipeline with EntityRuler. Called once per process."""
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(CRYPTO_ENTITY_PATTERNS)
    return nlp


def extract_ner_features(doc: spacy.tokens.Doc) -> dict[str, bool | int]:
    """Extract boolean entity category flags + total entity count from a doc."""
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

---

## Stage 2: sklearn Classifier

### Feature Pipeline

```
headline (str)  ──► TfidfVectorizer(max_features=500, ngram_range=(1,2))  ─┐
NER features    ──► (6 bool/int values → dense array)                     ─┤─► LogisticRegression
                                                                            │    → P(is_impactful)
```

`class_weight="balanced"` compensates for the 95/5 class imbalance.  
Model stored in MLflow as `models:/news-gatekeeper/Production`.

### Training Data (V1.5)

| Source | Label | Rows |
|--------|-------|------|
| FinRL_BTC_news_signals (HuggingFace, CDLA-2.0) | positive | 279 |
| `bronze.news_agent_eval_dataset` labeled non-crypto | negative | ~450 |
| **Total** | | **~730** |

Training script: `notebooks/train_gatekeeper.py` (notebook job, not SDP).

### V1 → V1.5 Transition

1. Label `bronze.news_agent_eval_dataset` (Task 4.3)
2. Run `notebooks/train_gatekeeper.py` → register `models:/news-gatekeeper/Production`
3. Set `GatekeeperConfig(ner_only_mode=False)` in `news/pipeline.py`
4. Redeploy news pipeline

---

## `GatekeeperConfig` Dataclass

Add to `src/trading_databricks/common/config.py` (following existing frozen dataclass pattern):

```python
@dataclass(frozen=True)
class GatekeeperConfig:
    model_uri: str = "models:/news-gatekeeper/Production"
    confidence_threshold: float = 0.5
    ner_only_mode: bool = True    # V1 default: NER only, no MLflow dependency
    text_window: int = 500        # chars of body to concat with headline
```

---

## `apply_gatekeeper()` — pandas UDF in `news/gold.py`

Pattern follows existing `silver.py` pure-transform convention. No SDP imports. Called by the `@dp.materialized_view` in `news/pipeline.py`.

```python
def apply_gatekeeper(df: DataFrame, *, config: GatekeeperConfig) -> DataFrame:
    """
    Adds is_impactful (bool) and gatekeeper_confidence (double) to df.

    Stage 1: spaCy EntityRuler on headline + body[:text_window].
    Stage 2: sklearn LogisticRegression (bypassed if ner_only_mode=True).
    """
    import mlflow
    import pandas as pd
    from pyspark.sql import functions as F
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import BooleanType, DoubleType, StructField, StructType

    from trading_databricks.agents.gatekeeper import build_nlp, extract_ner_features

    nlp = build_nlp()
    model = None if config.ner_only_mode else mlflow.sklearn.load_model(config.model_uri)
    window = config.text_window
    threshold = config.confidence_threshold

    return_schema = StructType([
        StructField("is_impactful", BooleanType()),
        StructField("gatekeeper_confidence", DoubleType()),
    ])

    @pandas_udf(return_schema)
    def _gate(headlines: pd.Series, bodies: pd.Series) -> pd.DataFrame:
        results = []
        for headline, body in zip(headlines, bodies):
            text = f"{headline or ''} {(body or '')[:window]}"
            doc = nlp(text)
            feats = extract_ner_features(doc)
            has_entity = any([
                feats["has_crypto_asset"], feats["has_exchange"],
                feats["has_stablecoin"], feats["has_protocol"], feats["has_concept"],
            ])
            if not has_entity:
                results.append({"is_impactful": False, "gatekeeper_confidence": 0.0})
            elif model is None:
                results.append({"is_impactful": True, "gatekeeper_confidence": 1.0})
            else:
                prob = float(model.predict_proba([[headline, feats]])[0][1])
                results.append({
                    "is_impactful": prob >= threshold,
                    "gatekeeper_confidence": prob,
                })
        return pd.DataFrame(results)

    gate = _gate(df["headline"], df["body"])
    return (
        df.withColumn("is_impactful", gate["is_impactful"])
          .withColumn("gatekeeper_confidence", gate["gatekeeper_confidence"])
    )
```

---

## Schema Update — `CAUSAL_FEATURES_SCHEMA`

`gatekeeper_confidence` inserted as the 4th field (after `is_impactful`). Schema: **17 → 18 fields**.

```python
StructField("gatekeeper_confidence", DoubleType(), nullable=True),
```

`nullable=True` — V1 ner_only_mode always writes 0.0 or 1.0, but future model versions may emit null on inference errors.

Files to update:
- `src/trading_databricks/news/schemas.py` — insert new StructField
- `docs/superpowers/specs/2026-04-03-mas-causality-source-of-truth-design.md` — update schema table

---

## Files Changed

| File | Action | What |
|------|--------|------|
| `src/trading_databricks/agents/gatekeeper.py` | **Create** | `CRYPTO_ENTITY_PATTERNS`, `build_nlp()`, `extract_ner_features()` |
| `src/trading_databricks/common/config.py` | **Modify** | Add `GatekeeperConfig` dataclass |
| `src/trading_databricks/news/gold.py` | **Modify** | `apply_gatekeeper()` pandas UDF; update `causal_features()` to call it first |
| `src/trading_databricks/news/schemas.py` | **Modify** | Add `gatekeeper_confidence` field → 18 fields |
| `docs/superpowers/specs/2026-04-03-mas-causality-source-of-truth-design.md` | **Modify** | Update schema table to 18 fields |
| `notebooks/train_gatekeeper.py` | **Create** | sklearn training script (V1.5 — not blocking V1) |
| `tests/agents/test_gatekeeper.py` | **Create** | Unit tests for NER patterns and feature extraction |
| `pyproject.toml` | **Modify** | Add `spacy>=3.7,<4` |

**Unchanged by this plan:** `agents/taxonomy.py`, `agents/pipeline.py`, `resources/agents_pipeline.pipeline.yml`, `news/pipeline.py` SDP wiring.

---

## Verification

### Unit Tests (pure Python, no Spark)

```python
# tests/agents/test_gatekeeper.py

def test_btc_headline_detected():
    nlp = build_nlp()
    feats = extract_ner_features(nlp("Bitcoin price drops 10% after ETF rejection"))
    assert feats["has_crypto_asset"] is True

def test_unrelated_article_rejected():
    nlp = build_nlp()
    feats = extract_ner_features(nlp("Hong Kong biotech raises $50M in Series B funding"))
    assert not any([feats["has_crypto_asset"], feats["has_exchange"],
                    feats["has_stablecoin"], feats["has_protocol"], feats["has_concept"]])

def test_exchange_detected():
    nlp = build_nlp()
    feats = extract_ner_features(nlp("Binance delists trading pairs amid regulatory pressure"))
    assert feats["has_exchange"] is True

def test_stablecoin_detected():
    nlp = build_nlp()
    feats = extract_ner_features(nlp("USDC briefly depegs to $0.87 following SVB collapse"))
    assert feats["has_stablecoin"] is True

def test_multi_entity_count():
    nlp = build_nlp()
    feats = extract_ner_features(nlp("Coinbase lists new BTC spot pair"))
    assert feats["entity_count"] >= 2
    assert feats["has_exchange"] is True
    assert feats["has_crypto_asset"] is True
```

### Schema Assertion

```python
assert len(CAUSAL_FEATURES_SCHEMA.fields) == 18
names = [f.name for f in CAUSAL_FEATURES_SCHEMA.fields]
assert "gatekeeper_confidence" in names
assert names.index("gatekeeper_confidence") == 3  # immediately after is_impactful
```

### Smoke Test (eval dataset, `uv run python`)

Run against the 5 eval articles from the data reality finding:

```python
articles = [
    ("Hong Kong biotech raises $50M Series B", ""),                    # → False
    ("Italian fashion brand Gucci launches NFT line", ""),             # → True (NFT)
    ("UAE credit bureau adopts new digital ID system", ""),            # → False
    ("FTX founder Sam Bankman-Fried faces fraud charges", ""),         # → True (FTX)
    ("Bitcoin ETF sees record $800M single-day inflow", ""),           # → True
]
# Expected: 3 True, 2 False
# Note: Gucci NFT is a correct true positive — crypto-adjacent content
```

Expected gatekeeper recall on `bronze.news_agent_eval_dataset`: **>0.95** (target from redesign plan).
