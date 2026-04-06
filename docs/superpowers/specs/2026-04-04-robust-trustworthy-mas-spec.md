# Robust & Trustworthy MAS — Causal Discovery Enhancement Specification

**Date:** 2026-04-04
**Version:** 3.0 (V3 Causal-Hardened)
**Status:** Draft — Pending Approval
**Extends:** `2026-04-04-mas-final-spec.md` (V2 RAG-Augmented, 20-field schema)
**Grounded in:**
- Chapter 13: Constraint-based (PC), Score-based (GES), Functional (ANM, LiNGAM), Gradient-based (NOTEARS, GOLEM)
- Chapter 14: DECI (Bayesian posterior + ICGNN), FCI (hidden confounding), ABCI (active experiment design)

---

## 0. Why This Extension Exists

The V2 MAS has one **critical structural weakness**: every causal claim is evaluated in **isolation**. Article _i_ produces chain _(T_i, M_i, D_i)_, validated only against a static whitelist. The system never asks:

1. **"Is the aggregate knowledge graph consistent?"** — Over 10,000 articles, do the extracted causal relationships form a valid DAG, or do they contain logical cycles?
2. **"Is the proposed direction empirically supported?"** — The LLM says _inflation → bearish via risk_off_, but historical data may show the opposite under certain market regimes.
3. **"Is this a real cause or a confounded correlation?"** — Multiple articles attribute ETH price drops to different treatments, suggesting a hidden common cause the LLM can't observe.
4. **"How confident are we in each edge?"** — A single extraction is not equally trustworthy as a pattern observed across 50 articles.
5. **"Should this article get full compute treatment?"** — Novel, high-uncertainty events need more resources; routine extractions matching prior patterns can be fast-tracked.

These are precisely the questions the causal discovery literature is designed to answer. This spec maps five mathematical frameworks to five concrete MAS enhancement modules.

---

## 1. Architecture: Six Enhancement Modules

```
V2 MAS pipeline (unchanged):
  silver.news_cleaned → gatekeeper → LangGraph → gold.news_causal_features
                                                        │
                                                        ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │  V3 ENHANCEMENT LAYER (6 post-extraction + intra-pipeline modules)    │
  │                                                                       │
  │  [PIE]  Pipeline-Integrated Enhancements (within LangGraph):          │
  │    Module A: Uncertainty-Aware Router (ABCI-inspired)                  │
  │    Module B: Evidence Semantic Grounding (ANM residual analogy)        │
  │                                                                       │
  │  [POST] Post-Extraction Batch Jobs (Databricks Jobs, periodic):       │
  │    Module C: DAG-ness Consistency Auditor (NOTEARS)                    │
  │    Module D: Empirical Direction Verifier (LiNGAM / ANM)              │
  │    Module E: Confounding Detector (FCI-inspired)                       │
  │    Module F: Dynamic Whitelist Governance (DECI PrioriKnowledge)       │
  └───────────────────────────────────────────────────────────────────────┘
```

### Module Classification

| Module | Type | Runs When | Frequency | Mathematical Basis |
|--------|------|-----------|-----------|-------------------|
| A. Uncertainty Router | Pipeline (LangGraph) | Per article, after VS retrieval | Real-time | ABCI (sequential uncertainty reduction) |
| B. Evidence Grounding | Pipeline (LangGraph) | Per article, in validator | Real-time | ANM (residual independence test analogy) |
| C. DAG-ness Auditor | Batch Job | Post-refresh | Daily | NOTEARS (acyclicity constraint `R(A) = tr(e^{A⊙A}) - d`) |
| D. Direction Verifier | Batch Job | When TRD-5 data available | Weekly | LiNGAM / ANM (HSIC on regression residuals) |
| E. Confounding Detector | Batch Job | Post-refresh | Daily | FCI (hidden common cause detection via bidirectional edges) |
| F. Whitelist Governance | Batch Job | Governance cycle | Monthly | DECI (PrioriKnowledge: required/forbidden edges from data) |

---

## 2. Module A — Uncertainty-Aware Router

### 2.1 Theoretical Basis: Active Bayesian Causal Inference (ABCI)

ABCI (Chapter 14) teaches that when discovering causal structure, we should prioritize interventions (compute expenditure) on the edges with **highest posterior uncertainty**. Translated to MAS: not all articles deserve equal compute. A routine "Fed rate decision" article that matches 50 prior accepted chains is low-uncertainty. A novel "African central bank announces BTC legal tender" with zero VS matches is high-uncertainty.

### 2.2 Design

Insert a **conditional edge** in the LangGraph between `few_shot_retriever` and `causality_extractor`:

```
few_shot_retriever
       │
       ▼
  ┌─────────────┐
  │  ROUTER     │
  │  (new node) │
  └──┬──────┬───┘
     │      │
     ▼      ▼
  FAST    CAREFUL
  PATH    PATH
     │      │
     ▼      ▼
  causality_extractor (shared, but with different prompts)
```

### 2.3 Routing Logic

```python
@dataclass(frozen=True)
class RouterConfig:
    high_similarity_threshold: float = 0.85   # VS cosine similarity
    low_similarity_threshold: float = 0.50     # below → high uncertainty
    self_consistency_k: int = 3                # number of LLM calls for careful path
    min_vs_results: int = 2                    # minimum similar past extractions

def route_article(state: MASState, *, config: RouterConfig) -> str:
    """Determine extraction path based on retrieval uncertainty.
    
    ABCI insight: focus compute budget on edges with highest
    posterior entropy. In our MAS analogy, the 'edge' is the
    (article → causal_chain) mapping and 'posterior entropy' is
    inversely proportional to VS retrieval similarity.
    """
    examples = state["few_shot_examples"]
    
    if not examples:
        return "careful"  # Cold start — maximum uncertainty
    
    top_similarity = examples[0].get("similarity_score", 0.0)
    n_results = len(examples)
    
    if top_similarity >= config.high_similarity_threshold and n_results >= config.min_vs_results:
        return "fast"     # Routine event with strong prior
    elif top_similarity < config.low_similarity_threshold:
        return "careful"  # Novel event — high uncertainty
    else:
        return "standard" # Default path
```

### 2.4 Path Behaviors

| Path | When | Prompt | LLM Calls | Latency | Cost |
|------|------|--------|-----------|---------|------|
| **Fast** | `top_similarity ≥ 0.85` & `n_results ≥ 2` | Lean prompt, top-3 few-shot only | 1 | ~1.5s | ~$0.01 |
| **Standard** | Middle range | Full prompt, top-5 few-shot | 1 | ~3s | ~$0.03 |
| **Careful** | `top_similarity < 0.50` or cold start | Full prompt + explicit CoT + self-consistency | k=3 calls, majority vote | ~9s | ~$0.09 |

**Self-Consistency (careful path):** Make k independent LLM calls with temperature > 0. Extract the treatment_class, mechanism, and direction from each. Select the majority consensus for each field. If no majority exists for any field, set `confidence = max(0, raw_conf - 0.3)` and flag `extraction_uncertainty = "high"`.

### 2.5 Impact on Schema

Add one field to `gold.news_causal_features`:

| # | Field | Type | Description |
|---|-------|------|-------------|
| 21 | `extraction_path` | STRING, nullable | `"fast"` / `"standard"` / `"careful"` — routing decision |

**Backward compatible:** nullable field, existing accepted chains have NULL.

### 2.6 Cost/Benefit

For a typical distribution (60% fast, 30% standard, 10% careful), the average cost per impactful article drops from $0.03 to ~$0.022 while the careful path **increases** extraction quality for the hardest 10% of articles.

---

## 3. Module B — Evidence Semantic Grounding

### 3.1 Theoretical Basis: ANM Residual Independence

The Additive Noise Model (ANM, Chapter 13) determines causal direction by testing whether regression residuals are **independent** of the predictor. If X→Y is the true direction, then `residuals(Y|X)` should be independent of X. If the residuals show dependence, the model is wrong.

**Analogy for MAS:** The current validator checks that `evidence_spans` are substrings of the article body. This proves the evidence is **not fabricated** but doesn't prove the evidence **actually supports the claimed causal chain**.

The ANM-inspired check: if removing the evidence spans from consideration would NOT change the extraction, then the evidence is **semantically independent** of the claim — meaning it's filler text, not genuine grounding.

### 3.2 Design

Add a scoring step inside `mechanism_validator`:

```python
def compute_evidence_relevance(
    trigger_event: str,
    mechanism: str,
    evidence_spans: list[str],
    *,
    embed_fn: Callable[[str], list[float]],
) -> float:
    """Score how semantically relevant evidence is to the causal claim.
    
    ANM analogy: if evidence_spans are truly grounding the claim
    (X causes Y), then 'residuals' after accounting for evidence
    should be near-zero. If evidence is irrelevant (independent of
    the claim), the 'residual' is large.
    
    Implementation: cosine similarity between claim embedding and
    evidence embedding. Low similarity = evidence doesn't support 
    the claim = analogous to dependent residuals in anti-causal direction.
    """
    claim_text = f"{trigger_event} causes {mechanism}"
    claim_embedding = embed_fn(claim_text)
    
    evidence_text = " ".join(evidence_spans)
    evidence_embedding = embed_fn(evidence_text)
    
    similarity = cosine_similarity(claim_embedding, evidence_embedding)
    return float(similarity)
```

### 3.3 Integration into Validator

```
Current validator pipeline:
  1. Parse L1 from treatment_class
  2. Lookup (L1, mechanism) in whitelist          → reject_mechanism
  3. Check direction ∈ valid_directions            → reject_direction
  4. Check evidence_spans ⊂ substrings(body)       → reject_evidence
  5. All pass → accepted

Enhanced validator pipeline (V3):
  1. Parse L1 from treatment_class
  2. Lookup (L1, mechanism) in whitelist          → reject_mechanism
  3. Check direction ∈ valid_directions            → reject_direction
  4. Check evidence_spans ⊂ substrings(body)       → reject_evidence
  5. ★ Compute evidence_relevance_score             → (informational, no rejection)
  6. All pass → accepted, with evidence_relevance_score populated
```

**Key decision:** Step 5 is **informational only** in V3. It does NOT reject chains. The score is written to the output schema for downstream analysis. After collecting enough data, we may add a rejection threshold in V4.

### 3.4 Impact on Schema

| # | Field | Type | Description |
|---|-------|------|-------------|
| 22 | `evidence_relevance_score` | DOUBLE, nullable | Cosine similarity between claim and evidence embeddings [0, 1]. NULL when `is_impactful=False`. Higher = better grounding. |

### 3.5 Embedding Strategy

Use the same `databricks-gte-large-en` endpoint already employed for VS index embedding. This avoids adding a new model dependency. Each validator call requires 2 embed calls (~50 tokens each) = negligible cost.

---

## 4. Module C — DAG-ness Consistency Auditor

### 4.1 Theoretical Basis: NOTEARS Acyclicity Constraint

NOTEARS (Chapter 13) encodes the acyclicity constraint as a differentiable function:

```
R(A) = tr(e^{A⊙A}) - d
```

Where:
- `A` is the adjacency matrix of the extracted causal graph
- `e` is the matrix exponential
- `⊙` is the Hadamard (element-wise) product
- `d` is the number of nodes
- `R(A) = 0` if and only if the graph represented by A is a DAG

**Why it matters for MAS:** Over thousands of articles, the LLM will inevitably extract chains that form logical cycles when composed. For example:
- Article 1: "Fed rate hike → liquidity_contraction → bearish" (Macro.rate_decision)
- Article 2: "BTC crash → regulatory_clarity → bullish" (Regulation.enforcement_action)
- Article 3: "regulations fuel institutional demand → demand_increase → bullish" (Adoption.institutional_buy)
- Article 4: "institutional demand drives up price → risk_on → bullish" (Macro)

Composing: rate_hike → crash → regulation → institutional_demand → price_up → rate_hike ⟹ **CYCLE**

Any downstream model that ingests this aggregate graph will inherit circular reasoning.

### 4.2 Design

**Build an aggregate adjacency matrix from all accepted chains:**

```python
from scipy import linalg
import numpy as np

def build_aggregate_adjacency(
    accepted_chains: list[dict],
) -> tuple[np.ndarray, list[str]]:
    """Build a weighted adjacency matrix from accepted causal chains.
    
    Nodes: unique treatment_class values
    Edge weight A[i][j]: number of accepted chains where treatment_class[i]'s
    mechanism causally implies treatment_class[j] via narrative composition.
    
    NOTEARS insight: R(A) = tr(e^{A⊙A}) - d measures 'DAG-ness'.
    If R(A) > 0, the aggregate knowledge graph contains cycles.
    """
    # Extract unique treatment_class values as nodes
    nodes = sorted({c["treatment_class"] for c in accepted_chains})
    node_idx = {n: i for i, n in enumerate(nodes)}
    d = len(nodes)
    A = np.zeros((d, d))
    
    # Build edges: for each pair of chains within a time window
    # that share mechanism/asset, create a directed edge from
    # the "cause" treatment to the "effect" treatment
    for chain in accepted_chains:
        src = chain["treatment_class"]
        # The mechanism implies a downstream effect on another node
        # (determined by co-occurrence in same event cluster)
        for target in chain.get("implied_targets", []):
            if target in node_idx:
                A[node_idx[src], node_idx[target]] += 1.0
    
    return A, nodes


def compute_dagness(A: np.ndarray) -> float:
    """NOTEARS acyclicity score: R(A) = tr(e^{A⊙A}) - d.
    
    Returns 0.0 if and only if A represents a DAG.
    Returns > 0 if cycles exist. Larger values = more cyclical.
    """
    d = A.shape[0]
    return float(np.trace(linalg.expm(A * A)) - d)
```

### 4.3 Cycle Detection and Flagging

When `R(A) > 0`, identify the edges contributing most to cyclicality:

```python
def find_cycle_contributors(
    A: np.ndarray,
    nodes: list[str],
    *,
    top_k: int = 10,
) -> list[dict]:
    """Identify edges that contribute most to cyclicality.
    
    Strategy: for each edge (i,j) where A[i][j] > 0, compute R(A)
    with and without that edge. The edge whose removal most reduces
    R(A) is the 'weakest link' in the cycle.
    
    This is analogous to GOLEM's approach of using gradient information
    to identify which edges to penalize most heavily.
    """
    dagness_full = compute_dagness(A)
    if dagness_full == 0.0:
        return []  # Already a DAG
    
    contributions = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] > 0:
                A_modified = A.copy()
                A_modified[i, j] = 0
                dagness_reduced = compute_dagness(A_modified)
                contributions.append({
                    "source": nodes[i],
                    "target": nodes[j],
                    "edge_weight": float(A[i, j]),
                    "dagness_reduction": dagness_full - dagness_reduced,
                })
    
    contributions.sort(key=lambda x: x["dagness_reduction"], reverse=True)
    return contributions[:top_k]
```

### 4.4 Databricks Implementation

**Job type:** Databricks Workflow (not SDP), runs nightly after `gold.news_causal_features` MV refresh.

**Output table:** `gold.dagness_audit_log`

```python
DAGNESS_AUDIT_SCHEMA = StructType([
    StructField("audit_date",          DateType(),              nullable=False),
    StructField("total_accepted_chains", IntegerType(),         nullable=False),
    StructField("num_nodes",           IntegerType(),           nullable=False),
    StructField("num_edges",           IntegerType(),           nullable=False),
    StructField("dagness_score",       DoubleType(),            nullable=False),  # 0.0 = DAG
    StructField("is_dag",              BooleanType(),           nullable=False),
    StructField("top_cycle_contributors", StringType(),         nullable=True),   # JSON array
    StructField("flagged_story_ids",   ArrayType(StringType()), nullable=True),   # chains to review
])
```

### 4.5 Operational Response

| dagness_score | Status | Action |
|---------------|--------|--------|
| 0.0 | ✅ Consistent | No action |
| 0.0 < R ≤ 1.0 | ⚠️ Mild cyclicality | Log and monitor; likely benign semantic overlap |
| R > 1.0 | 🔴 Significant cycles | Alert; review top cycle contributors; consider removing weakest-edge chains from VS index |

---

## 5. Module D — Empirical Direction Verifier

### 5.1 Theoretical Basis: LiNGAM + ANM Residual Tests

LiNGAM (Chapter 13) exploits **non-Gaussianity** in financial data to determine causal direction. Financial returns are notoriously heavy-tailed (non-Gaussian), making them ideal candidates.

ANM tests directionality by fitting non-linear regressions in both directions and checking residual independence via HSIC:

```
1. Fit Y = f(X) + ε_xy     (causal direction)
2. Fit X = g(Y) + ε_yx     (anti-causal direction)
3. HSIC(X, ε_xy) → independent?    If yes → X→Y is correct
4. HSIC(Y, ε_yx) → independent?    If yes → Y→X is correct
```

If the residuals for the causal model are independent (HSIC test returns 1) but the residuals for the anti-causal model are dependent (HSIC returns 0), the causal direction is confirmed.

### 5.2 Applicability Constraint

**This module requires quantitative market data.** It only activates after TRD-5 (dollar bars) delivers `gold.dollar_bars`. Additionally, it only applies to **Macro** treatment classes where the treatment has a quantifiable proxy variable:

| Treatment Class | Proxy Variable | Source |
|----------------|---------------|--------|
| Macro.rate_decision | Federal funds rate change | External API |
| Macro.inflation_data | CPI/PPI surprise vs. consensus | External API |
| Macro.employment_data | NFP surprise | External API |
| Macro.gdp | GDP surprise | External API |

For non-quantifiable treatments (Security.exchange_hack, Adoption.corporate_treasury), this module is not applicable — the LLM's directional claim cannot be empirically falsified at the individual article level.

### 5.3 Design

```python
from scipy.stats import spearmanr
import numpy as np

def verify_direction_empirically(
    treatment_proxy: np.ndarray,
    outcome_returns: np.ndarray,
    *,
    claimed_direction: str,
    alpha: float = 0.05,
) -> dict:
    """Verify LLM-claimed direction using ANM-inspired residual test.
    
    ANM principle (Chapter 13): Fit regressions in both directions.
    If residuals in the causal direction are independent of the
    predictor (HSIC test), the direction is supported.
    
    Simplification for MAS: We use Spearman rank correlation for
    direction verification (not full HSIC/GAM) because:
    1. We're testing monotonic direction, not complex nonlinear structure
    2. Spearman is robust to heavy tails in financial data
    3. We only need to confirm/contradict the sign, not the full DAG
    
    Returns a directional confidence assessment.
    """
    rho, p_value = spearmanr(treatment_proxy, outcome_returns)
    
    empirical_direction = "bearish" if rho < 0 else "bullish"
    direction_agrees = (empirical_direction == claimed_direction)
    
    return {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "empirical_direction": empirical_direction,
        "claimed_direction": claimed_direction,
        "direction_agrees": direction_agrees,
        "is_significant": p_value < alpha,
        "verdict": _compute_verdict(direction_agrees, p_value, alpha),
    }


def _compute_verdict(agrees: bool, p: float, alpha: float) -> str:
    if p >= alpha:
        return "inconclusive"      # Not enough statistical power
    elif agrees:
        return "empirically_confirmed"
    else:
        return "empirically_contradicted"
```

### 5.4 HSIC Extension (V4 — When TRD-5 Matures)

Once sufficient data accumulates (>500 accepted Macro chains), upgrade to the full ANM residual test:

```python
# V4 upgrade path — requires gCastle or custom HSIC implementation
from castle.common.independence_tests import hsic_test

def verify_direction_anm(
    treatment_proxy: np.ndarray,
    outcome_returns: np.ndarray,
    *,
    n_splines: int = 50,
    alpha: float = 0.05,
) -> dict:
    """Full ANM direction test using HSIC on spline regression residuals.
    
    As described in Chapter 13:
    1. Fit Y = f(X) + ε via spline regression (causal)  
    2. Fit X = g(Y) + ε via spline regression (anti-causal)
    3. HSIC test on residuals determines direction
    
    Independent residuals (HSIC=1) → correct causal direction.
    """
    from pygam import GAM
    
    # Causal direction: treatment → returns
    model_xy = GAM(n_splines=n_splines).fit(
        treatment_proxy.reshape(-1, 1), outcome_returns
    )
    residuals_xy = outcome_returns - model_xy.predict(
        treatment_proxy.reshape(-1, 1)
    )
    
    # Anti-causal direction: returns → treatment
    model_yx = GAM(n_splines=n_splines).fit(
        outcome_returns.reshape(-1, 1), treatment_proxy
    )
    residuals_yx = treatment_proxy - model_yx.predict(
        outcome_returns.reshape(-1, 1)
    )
    
    # HSIC independence test
    is_indep_causal = hsic_test(
        x=treatment_proxy.reshape(-1, 1),
        y=residuals_xy.reshape(-1, 1),
        alpha=alpha,
    )  # 1 = independent, 0 = dependent
    
    is_indep_anticausal = hsic_test(
        x=outcome_returns.reshape(-1, 1),
        y=residuals_yx.reshape(-1, 1),
        alpha=alpha,
    )
    
    # Decision logic from Chapter 13:
    # Independent causal + dependent anti-causal → X causes Y
    # Dependent causal + independent anti-causal → Y causes X
    if is_indep_causal == 1 and is_indep_anticausal == 0:
        return {"verdict": "treatment_causes_outcome", "confidence": "strong"}
    elif is_indep_causal == 0 and is_indep_anticausal == 1:
        return {"verdict": "reverse_causation", "confidence": "strong"}
    else:
        return {"verdict": "inconclusive", "confidence": "weak"}
```

### 5.5 Output Table

`gold.direction_verification_log`

```python
DIRECTION_VERIFICATION_SCHEMA = StructType([
    StructField("verification_date",  DateType(),    nullable=False),
    StructField("treatment_class",    StringType(),  nullable=False),
    StructField("mechanism",          StringType(),  nullable=False),
    StructField("claimed_direction",  StringType(),  nullable=False),
    StructField("empirical_direction", StringType(), nullable=False),
    StructField("spearman_rho",       DoubleType(),  nullable=False),
    StructField("p_value",            DoubleType(),  nullable=False),
    StructField("verdict",            StringType(),  nullable=False),
    StructField("sample_size",        IntegerType(), nullable=False),
    StructField("story_ids",          ArrayType(StringType()), nullable=True),
])
```

---

## 6. Module E — Confounding Detector

### 6.1 Theoretical Basis: FCI Hidden Common Cause Detection

FCI (Chapter 14) extends the PC algorithm to handle **hidden confounders**. Its key output is an edge type vocabulary richer than traditional DAGs:

| FCI Edge | Meaning | MAS Analogy |
|----------|---------|-------------|
| `i → j` (G[i,j]=-1, G[j,i]=1) | i causes j | Article correctly identifies T→Y |
| `i ↔ j` (G[i,j]=1, G[j,i]=1) | Hidden common cause of i and j | Two treatments that APPEAR related but are actually both caused by an unobserved event |
| `i o→ j` (G[i,j]=2, G[j,i]=1) | j is not an ancestor of i | Partial direction information |
| `i o─o j` (G[i,j]=2, G[j,i]=2) | No set d-separates i and j | Maximum uncertainty |

**Why it matters:** Financial news is rife with confounded reporting. When three articles within 2 hours attribute an ETH price drop to three different treatments (exchange_hack, macro risk_off, whale_movement), the most parsimonious explanation is often a **hidden common cause** (e.g., a large leveraged unwind) that simultaneously triggered the hack reporting, risk sentiment shift, and whale selling.

### 6.2 Design: Temporal Confounding Heuristic

We cannot run the full FCI algorithm on extracted chains (it requires multivariate observational data, not LLM text outputs). Instead, we apply FCI's **conceptual insight** as a heuristic:

**Rule:** When multiple accepted chains within a **confounding window** target the same asset + same direction but cite **different treatment classes**, flag as potentially confounded.

```python
from pyspark.sql import functions as F, Window

def detect_confounded_clusters(
    causal_features_df,  # gold.news_causal_features
    *,
    window_hours: int = 4,
    min_distinct_treatments: int = 3,
    min_articles: int = 3,
):
    """FCI-inspired confounding detection via temporal clustering.
    
    FCI's key insight (Chapter 14): when two variables appear
    bidirectionally related (i ↔ j), it signals a hidden common
    cause. In news extraction, the analog is:
    
    Multiple distinct treatments → same outcome + same time window
    ═══════  signals hidden confounder U  ═══════
    
    This is the FCI 'bidirectional edge' pattern projected onto
    the temporal structure of news coverage.
    """
    window_spec = Window.partitionBy("asset", "expected_direction").orderBy("published_at")
    
    return (
        causal_features_df
        .filter(F.col("validation_status") == "accepted")
        .filter(F.col("is_impactful") == True)
        .withColumn("cluster_id",
            F.window(F.col("published_at"), f"{window_hours} hours"))
        .groupBy("cluster_id", "asset", "expected_direction")
        .agg(
            F.countDistinct("treatment_class").alias("distinct_treatments"),
            F.count("*").alias("article_count"),
            F.collect_set("treatment_class").alias("treatment_classes"),
            F.collect_list("story_id").alias("story_ids"),
        )
        .filter(F.col("distinct_treatments") >= min_distinct_treatments)
        .filter(F.col("article_count") >= min_articles)
        .withColumn("confounding_flag", F.lit("potential_hidden_confounder"))
        .withColumn("fci_interpretation",
            F.lit(
                "Multiple distinct treatments pointing to same outcome "
                "within time window suggests hidden common cause (U). "
                "Analogous to FCI bidirectional edge (i ↔ j)."
            )
        )
    )
```

### 6.3 Downstream Impact

Confounded clusters should be **weighted down** in downstream ML training. The ML pipeline can use the confounding flag to:

1. Apply a discount factor to `confidence` for affected chains
2. Exclude confounded clusters from CausalBERT training batches
3. Include a `confounded_cluster_id` feature in the training data to let the model learn to discount correlated events

### 6.4 Output Table

`gold.confounding_audit_log`

```python
CONFOUNDING_AUDIT_SCHEMA = StructType([
    StructField("audit_date",           DateType(),              nullable=False),
    StructField("cluster_start",        TimestampType(),         nullable=False),
    StructField("cluster_end",          TimestampType(),         nullable=False),
    StructField("asset",                StringType(),            nullable=False),
    StructField("expected_direction",   StringType(),            nullable=False),
    StructField("distinct_treatments",  IntegerType(),           nullable=False),
    StructField("article_count",        IntegerType(),           nullable=False),
    StructField("treatment_classes",    ArrayType(StringType()), nullable=False),
    StructField("story_ids",            ArrayType(StringType()), nullable=False),
    StructField("confounding_flag",     StringType(),            nullable=False),
])
```

### 6.5 Impact on Schema

Add one field to `gold.news_causal_features`:

| # | Field | Type | Description |
|---|-------|------|-------------|
| 23 | `confounding_cluster_id` | STRING, nullable | UUID of confounding cluster if detected. NULL when not part of a confounded cluster. |

---

## 7. Module F — Dynamic Whitelist Governance

### 7.1 Theoretical Basis: DECI PrioriKnowledge + Bayesian Posterior

DECI (Chapter 14) combines data-driven discovery with expert knowledge injection:

```python
# From Chapter 14 — PrioriKnowledge pattern
prior = prior_class(...)
prior.set(row=3, col=0, value=0.9)  # high belief in edge (3→0)
```

The key insight: DECI doesn't treat the graph as fixed. It maintains a **Bayesian posterior over possible graph structures**, updated by data. The `PrioriKnowledge` framework allows injecting human expertise as prior beliefs that:
- **Required edges** = known causal relationships (existing whitelist entries)
- **Forbidden edges** = known impossibilities (e.g., "stablecoin_risk" cannot be a mechanism for Macro treatments)
- **Uncertain edges** = candidate mechanisms awaiting enough evidence

**Translated to MAS:** The 40-entry whitelist is a set of required edges. The `rejected_mechanism_raw` data reveals candidate edges that the LLM believes exist but humans haven't validated. The monthly governance job applies DECI's principle: use data to update beliefs about which candidates should be promoted.

### 7.2 Design: Evidence-Driven Whitelist Expansion

```python
def analyze_rejected_mechanisms(
    causal_features_df,
    *,
    min_frequency: int = 10,
    min_distinct_articles: int = 5,
    clustering_threshold: float = 0.80,
) -> list[dict]:
    """DECI PrioriKnowledge-inspired whitelist governance.
    
    Existing whitelist = PrioriKnowledge required_edges (high prior belief)
    Rejected mechanisms = uncertain edges (prior ≈ 0.5)
    
    This function computes a posterior-like score for each rejected
    mechanism based on:
    1. Frequency (how often the LLM produces this mechanism)
    2. Diversity (how many distinct articles trigger it)
    3. Semantic clustering (do different phrasings cluster together?)
    4. Cross-validation with downstream (do articles with this mechanism
       actually predict price movements?)
    
    High-scoring candidates are promoted for human review.
    """
    rejected = (
        causal_features_df
        .filter(F.col("validation_status") == "rejected_mechanism")
        .filter(F.col("rejected_mechanism_raw").isNotNull())
        .groupBy("rejected_mechanism_raw")
        .agg(
            F.count("*").alias("frequency"),
            F.countDistinct("story_id").alias("distinct_articles"),
            F.collect_set(F.substring("treatment_class", 1, 
                F.instr("treatment_class", F.lit(".")) - 1
            )).alias("source_l1_categories"),
        )
        .filter(F.col("frequency") >= min_frequency)
        .filter(F.col("distinct_articles") >= min_distinct_articles)
        .orderBy(F.desc("frequency"))
    )
    
    # Cluster semantically similar rejected mechanisms
    # e.g., "regulatory_pressure", "regulatory_crackdown", "regulation_risk"
    # should cluster together as one candidate
    # (Use embed + agglomerative clustering)
    
    return rejected
```

### 7.3 Governance Workflow

```
Monthly governance cycle:
                                                          
  gold.news_causal_features (rejected_mechanism_raw)
           │
           ▼
  ┌─────────────────────────┐
  │  analyze_rejected_mechs │  Databricks Job (monthly)
  │  + embed + cluster      │
  └──────────┬──────────────┘
             │ candidate_mechanisms with posterior scores
             ▼
  ┌─────────────────────────┐
  │  gold.mechanism_         │  New Delta table (governance log)
  │  governance_candidates   │
  └──────────┬──────────────┘
             │ Human reviews dashboard
             ▼
  ┌─────────────────────────┐
  │  Approved? → INSERT     │  ←── DECI "required_edge" promotion
  │  into gold.mechanism_   │
  │  whitelist               │
  │                         │
  │  Rejected? → INSERT     │  ←── DECI "forbidden_edge" establishment
  │  into gold.mechanism_   │
  │  blocklist               │
  └─────────────────────────┘
```

### 7.4 Mechanism Blocklist (New)

A **blocklist** prevents the LLM from re-proposing mechanisms that have been explicitly rejected by human governance. This is the DECI `forbidden_edge` analog.

```python
MECHANISM_BLOCKLIST_SCHEMA = StructType([
    StructField("mechanism_raw",    StringType(), nullable=False),
    StructField("rejection_reason", StringType(), nullable=False),
    StructField("rejected_date",    DateType(),   nullable=False),
    StructField("reviewer",         StringType(), nullable=False),
])
```

When a blocklisted mechanism is extracted, the validator assigns `validation_status = "rejected_mechanism_blocked"` (new status value), providing a stronger signal than generic rejection.

### 7.5 Output Table

`gold.mechanism_governance_candidates`

```python
GOVERNANCE_CANDIDATE_SCHEMA = StructType([
    StructField("candidate_mechanism",   StringType(),  nullable=False),
    StructField("cluster_label",         StringType(),  nullable=True),
    StructField("frequency",             IntegerType(), nullable=False),
    StructField("distinct_articles",     IntegerType(), nullable=False),
    StructField("source_l1_categories",  ArrayType(StringType()), nullable=False),
    StructField("posterior_score",        DoubleType(),  nullable=False),
    StructField("governance_status",     StringType(),  nullable=False),  # pending|approved|rejected
    StructField("reviewed_date",         DateType(),    nullable=True),
    StructField("reviewer",              StringType(),  nullable=True),
])
```

---

## 8. Extended Schema Summary

### V3 Additions to `gold.news_causal_features`

The V2 schema (20 fields) is extended with 3 new nullable fields:

| # | Field | Type | Source Module | Description |
|---|-------|------|---------------|-------------|
| 21 | `extraction_path` | STRING | Module A (Router) | `"fast"` / `"standard"` / `"careful"` |
| 22 | `evidence_relevance_score` | DOUBLE | Module B (Evidence Grounding) | Cosine similarity [0,1], NULL for non-impactful |
| 23 | `confounding_cluster_id` | STRING | Module E (Confounding) | Cluster UUID, NULL if not confounded |

**Total: 23 fields** (was 20). All new fields are nullable. **Backward compatible.**

### New Gold Tables

| Table | Purpose | Module | Write Mode |
|-------|---------|--------|-----------|
| `gold.dagness_audit_log` | DAG-ness consistency tracking | C | Append |
| `gold.direction_verification_log` | Empirical direction checks | D | Append |
| `gold.confounding_audit_log` | Hidden confounding clusters | E | Append |
| `gold.mechanism_governance_candidates` | Whitelist expansion pipeline | F | Append |
| `gold.mechanism_blocklist` | Forbidden mechanism registry | F | SCD1 |

---

## 9. Mathematical Mapping Summary

| MAS Weakness | Causal Discovery Concept | Chapter | Key Formula / Technique | MAS Module |
|-------------|------------------------|---------|------------------------|------------|
| No global consistency check | NOTEARS acyclicity constraint | 13 | `R(A) = tr(e^{A⊙A}) - d = 0 ⟹ DAG` | C |
| Direction based on LLM opinion only | LiNGAM (non-Gaussianity) + ANM (residual test) | 13 | `HSIC(X, ε_xy) → independent ⟹ X→Y correct` | D |
| Evidence is verified but not semantically relevant | ANM residual independence analogy | 13 | Embedding cosine sim as "semantic HSIC" | B |
| Hidden confounders in news | FCI (bidirectional edge = hidden cause) | 14 | `G[i,j]=1, G[j,i]=1 ⟹ hidden common cause` | E |
| Static whitelist, manual expansion | DECI PrioriKnowledge + Bayesian posterior | 14 | `prior.set(row, col, value)` + frequency scoring | F |
| All articles get equal compute | ABCI (target highest posterior entropy) | 14 | Entropy-guided experiment selection | A |

---

## 10. Implementation Roadmap

### Phase 0: Schema Extension (1 day)
- Add 3 nullable fields to `CAUSAL_FEATURES_SCHEMA`
- Validate `len(schema.fields) == 23`
- No pipeline changes — new fields are NULL by default

### Phase 1: Pipeline-Integrated Modules — A + B (1 week)
- **Module A:** Add `RouterConfig`, `route_article()` node, conditional edges in LangGraph
- **Module B:** Add `compute_evidence_relevance()` in validator
- Deploy updated LangGraph endpoint
- Tests: routing unit tests, evidence scoring assertions

### Phase 2: DAG-ness Auditor — C (3 days)
- Implement `build_aggregate_adjacency()`, `compute_dagness()`, `find_cycle_contributors()`
- Create `gold.dagness_audit_log` table
- Schedule nightly Databricks Job
- Tests: synthetic cycle detection, dagness=0 for clean graphs

### Phase 3: Confounding Detector — E (3 days)
- Implement `detect_confounded_clusters()` Spark transform
- Create `gold.confounding_audit_log` table
- Backfill `confounding_cluster_id` into causal_features
- Tests: synthetic confounded cluster matching

### Phase 4: Whitelist Governance — F (1 week)
- Implement `analyze_rejected_mechanisms()` with embedding clustering
- Create `gold.mechanism_governance_candidates` and `gold.mechanism_blocklist`
- Build governance dashboard (SQL queries + optional Streamlit)
- Integrate blocklist into validator
- Tests: governance scoring assertions

### Phase 5: Direction Verifier — D (deferred to post-TRD-5)
- Requires `gold.dollar_bars` data
- Implement `verify_direction_empirically()` + HSIC upgrade path
- Create `gold.direction_verification_log`
- Tests: synthetic direction agreement/disagreement

---

## 11. Verification Checklist

### Schema
- [ ] `len(CAUSAL_FEATURES_SCHEMA.fields) == 23`
- [ ] Fields 21-23 are ALL nullable
- [ ] `extraction_path` at index 20 (0-indexed)
- [ ] `evidence_relevance_score` at index 21
- [ ] `confounding_cluster_id` at index 22

### Module A — Router
- [ ] Cold start (no VS results) → `"careful"` path
- [ ] High similarity (>0.85) + ≥2 results → `"fast"` path
- [ ] Low similarity (<0.50) → `"careful"` path
- [ ] Self-consistency (careful) with k=3 returns majority vote
- [ ] `extraction_path` populated in output row

### Module B — Evidence Grounding
- [ ] Relevant evidence (spans matching trigger_event) → score > 0.5
- [ ] Irrelevant evidence (random article sentences) → score < 0.3
- [ ] Same `databricks-gte-large-en` endpoint as VS index
- [ ] `evidence_relevance_score` populated in output row
- [ ] Informational only — NO rejections based on this score

### Module C — DAG-ness
- [ ] `compute_dagness(identity_matrix)` returns 0.0 (identity = no edges = trivially DAG)
- [ ] `compute_dagness(cyclic_matrix)` returns > 0.0
- [ ] `find_cycle_contributors()` identifies correct weakest edge
- [ ] Nightly job writes to `gold.dagness_audit_log`

### Module D — Direction Verifier
- [ ] Deferred until TRD-5 delivers dollar bars
- [ ] Spearman test correctly identifies positive/negative correlation
- [ ] p-value threshold respected (default α=0.05)

### Module E — Confounding
- [ ] 3+ distinct treatments in 4-hour window for same asset → flagged
- [ ] Single-treatment clusters → NOT flagged
- [ ] `confounding_cluster_id` populated in causal_features
- [ ] Audit log written to `gold.confounding_audit_log`

### Module F — Governance
- [ ] Rejected mechanisms with frequency ≥ 10 appear in candidates
- [ ] Semantic clustering groups variations (e.g., "regulatory_pressure" ≈ "regulatory_crackdown")
- [ ] Approved candidate → inserted into `gold.mechanism_whitelist`
- [ ] Rejected candidate → inserted into `gold.mechanism_blocklist`
- [ ] Blocklisted mechanism → `validation_status = "rejected_mechanism_blocked"`

---

## 12. Assumptions and Limitations

### What This Extension Does NOT Do

1. **Does not run full causal discovery algorithms on extracted text.** FCI, DECI, and NOTEARS require multivariate observational numerical data. The extracted causal chains are structured text, not iid samples from a joint distribution.

2. **Does not replace human governance.** Module F surfaces candidates; humans approve/reject. The "Bayesian posterior" is a scoring heuristic, not a literal posterior distribution.

3. **Does not guarantee cycle-free extraction.** Module C detects cycles post-hoc. The LLM can still produce individual chains that would form cycles when composed. The goal is detection and monitoring, not prevention.

4. **Does not handle interventional data.** ABCI (Module A) designs sequential experiments; our system only observes articles as they arrive. The uncertainty routing is ABCI-inspired but not interventionist.

### What It DOES Accomplish

1. **Makes hidden logical contradictions visible** (Module C, NOTEARS)
2. **Grounds directional claims in empirical data** (Module D, LiNGAM/ANM)
3. **Flags spurious correlations from hidden confounders** (Module E, FCI)
4. **Automates whitelist evolution with human-in-the-loop** (Module F, DECI PrioriKnowledge)
5. **Allocates compute budget proportional to extraction uncertainty** (Module A, ABCI)
6. **Quantifies evidence quality beyond substring matching** (Module B, ANM residual analogy)

---

## 13. Dependencies on Existing Spec

| This Spec Section | Depends On | From `2026-04-04-mas-final-spec.md` |
|-------------------|-----------|-------------------------------------|
| Module A (Router) | VS retrieval similarity scores | §4 Vector Search |
| Module B (Evidence) | `evidence_spans`, `trigger_event` | §6.1 Causal Chain JSON |
| Module C (DAG-ness) | `treatment_class`, `validation_status` | §10 Output Schema |
| Module D (Direction) | `gold.dollar_bars` | §2.3, TRD-5 dependency |
| Module E (Confounding) | `published_at`, `asset`, `expected_direction` | §10 Output Schema |
| Module F (Governance) | `rejected_mechanism_raw` | §7.2 Governance Field |
| Schema extension | 20-field schema | §10 (extended to 23) |

**No changes to the V2 pipeline are required.** All modules extend or wrap existing functionality. The LangGraph topology adds nodes (Module A, B) but does not remove or modify existing nodes.
