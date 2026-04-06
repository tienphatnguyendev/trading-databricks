# Causal DAG Audit — MAS Final Spec §2

**Reviewer posture:** Senior Data Scientist, Causality Reasoning
**Document under review:** [mas-final-spec.md §2.1–2.9](file:///Users/aaronng/repos/trading-databricks/docs/superpowers/specs/2026-04-04-mas-final-spec.md#L33-L173)
**Verdict:** The DAG has **fundamental structural defects** that undermine the causal identification claims. The framework is a useful *first approximation* but cannot be called robust or trustworthy in its current form.

---

## The Current DAG

```
         U (unobserved: order flow, insider info, whale positioning)
         |
         v
Z ─────> T ───M───> Y
         ^          ^
         |          |
         X ─────────┘
```

---

## Defect 1 — FATAL: Ontological vs. Epistemological Conflation

> [!CAUTION]
> The DAG conflates two fundamentally different causal structures into one diagram. This is the most serious error.

The spec uses a **single DAG** to represent two different things:

### Layer 1 — The Real World (ontological)
In reality, events happen in the world and THEN journalists write about them:

```
Real-world event E  ──────────>  Price movement Y
       |                              ^
       |                              |
       v                              |
  Article Z (report of E)       Market state X
```

The causal arrow is **E → Z** (the event causes the article), NOT **Z → T** (the article causes the treatment). A Bitcoin halving happens, and THEN Reuters reports on it. The article didn't cause the halving.

### Layer 2 — The Measurement Process (epistemological)
The MAS reads articles and extracts treatment labels:

```
Article Z  ──[MAS pipeline]──>  T̂ (estimate of treatment class)
```

This is a **measurement process**, not a causal mechanism. The MAS produces T̂ (an estimate of the real-world event E), not T itself.

### Why This Matters

The current DAG writes `Z → T` as if Z *causes* T. But:
- Z is a **report** of the real event E
- T is a **label** the MAS assigns to describe E
- The real causal arrow is `E → Z` and `E → T`(the event causes both the article and the treatment)

By writing `Z → T`, the spec accidentally treats Z as if it were an **instrument** — an exogenous variable that affects T but not Y. But Z is not exogenous. Z is a downstream consequence of the real event E, contaminated by journalistic selection bias, editorial framing, and publication timing.

**Severity: FATAL** — This error propagates through §2.2 (structural equations), §2.4 (backdoor criterion), and §2.6 (RAG/instrument claim).

---

## Defect 2 — SERIOUS: Z Is Not An Instrument

> [!WARNING]
> Line 141 makes a formal claim that Z is an instrument. This claim does not hold under scrutiny.

The spec states:

> *"R is a deterministic function of Z (the instrument)"*

For Z to be a valid instrument, three conditions must hold:

| Condition | Requirement | Holds? | Why |
|-----------|------------|--------|-----|
| **Relevance** | Z affects T | ✅ Yes | T is literally extracted from Z |
| **Exclusion restriction** | Z affects Y **only** through T | ❌ No | Publication of article Z itself moves markets (information revelation) |
| **Independence** | Z is independent of U | ❌ No | Insider info (U) leaks into articles; journalist access to sources is confounded |

### Exclusion Violation: Z → Y Direct Effect

The act of publishing a news article about Bitcoin causes price movement **regardless of the article's content**. This is the [information revelation effect](https://en.wikipedia.org/wiki/Information_asymmetry). Examples:

- Bloomberg publishes "SEC considering Bitcoin ETF" → traders react to the publication itself, not just the classified treatment. The *framing*, *source authority*, and *timing* of Z all directly affect Y.
- Two articles with identical treatment_class (`Regulation.etf_ruling`) but different sources (Reuters vs. unknown blog) have different effects on Y. The causal pathway is Z → Y directly, not only through T.

This means the exclusion restriction is violated: Z has a **direct effect on Y** that doesn't pass through the discrete treatment T.

### Independence Violation: U → Z

Order flow (U) affects what news gets written. Market makers and institutional traders have information that leaks into news coverage through:
- Unnamed sources ("according to a person familiar with the matter")
- Trading pattern reporting ("unusual options activity suggests...")
- Self-fulfilling prophecy cycles where traders' positions influence journalist narratives

**Severity: SERIOUS** — If Z is not a valid instrument, the "formal claim" in §2.6 is incorrect, and the RAG justification needs re-grounding.

---

## Defect 3 — SERIOUS: Incomplete Backdoor Adjustment

> [!WARNING]
> The backdoor criterion claim in §2.4 does **not** hold for the U-path.

The spec claims (line 105–111):

> *"Under the backdoor criterion, if X blocks all backdoor paths from T to Y"*

Let's enumerate ALL backdoor paths from T to Y (non-causal paths that create spurious association):

| # | Path | Blocked by X? |
|---|------|:------------:|
| 1 | T ← **X** → Y | ✅ Yes — X is conditioned on |
| 2 | T ← **U** → Y | ❌ **No** — U is unobserved, X cannot block it |

The DAG explicitly shows U → T and U → Y (line 40–42). This creates a **backdoor path** T ← U → Y that **cannot be blocked** by conditioning on X, because U is unobserved.

The spec acknowledges this in §2.9 line 168: *"U (order flow, positioning) confounds T→Y."* But this acknowledgment **contradicts** the backdoor adjustment formula in §2.4 which claims identification IS achieved via X.

### The Honesty Problem

You cannot simultaneously:
1. Draw U → T and U → Y in your DAG (admitting confounding)
2. Claim the backdoor criterion is satisfied by conditioning on X (claiming identification)

These are mutually exclusive. Either:
- **Remove U from the DAG** and acknowledge you're assuming no unmeasured confounding (ignorability), OR
- **Keep U in the DAG** and admit the backdoor criterion fails, requiring alternative methods (sensitivity analysis, bounds, IV, or front-door)

**Severity: SERIOUS** — The core identification strategy is internally inconsistent.

---

## Defect 4 — SERIOUS: Missing Edges

The DAG omits at least 4 plausible causal edges:

### 4a. X → Z (Market conditions influence news coverage)

Journalists write more articles about crypto during bull markets. Market volatility (X) directly affects the **volume, tone, and selection** of articles (Z). This means Z is NOT independent of X — it's a consequence of X.

```
Missing edge:  X ──────> Z
```

This is a **selection bias** problem. The articles you observe are a non-random sample of the information space, selected by market conditions.

### 4b. Y → Z (Reflexivity / Post-hoc reporting)

Price movements cause news coverage. A 20% BTC crash at 9am causes articles to be published at 10am attributing the crash to various causes. These articles are **post-hoc rationalizations**, not genuine treatment reports.

```
Missing edge:  Y ──────> Z  (reverse causality)
```

This is the [Soros reflexivity](https://en.wikipedia.org/wiki/Reflexivity_(social_theory)) problem. Markets and news are bidirectionally coupled. The current DAG is purely unidirectional (Z → T → Y) and cannot represent this.

### 4c. T_past → Z_future (Temporal autocorrelation)

Previous treatments influence future articles. If a "Regulation.enforcement_action" treatment occurred yesterday, today's articles are more likely to discuss regulatory topics. The treatment history shapes the news landscape.

```
Missing edge:  T_{t-1} ──────> Z_t
```

This violates the implicit iid assumption in the structural equations.

### 4d. Z → Y Direct (Information revelation)

As discussed in Defect 2: publication itself moves markets.

```
Missing edge:  Z ──────> Y  (direct, bypassing T)
```

### Corrected edge set

```
Current:  Z→T, X→T, X→Y, U→T, U→Y, T→M→Y
Missing:  X→Z, Y→Z, T_{t-1}→Z_t, Z→Y(direct)
```

**Severity: SERIOUS** — Missing edges mean the DAG does not represent the true data-generating process. Any causal identification relying on this DAG inherits the false independence assumptions.

---

## Defect 5 — MODERATE: M Is Not A Standard Mediator

> [!IMPORTANT]
> The `mechanism` field is a label extracted from text, not an independently measured mediator variable.

In a standard mediation analysis (Baron & Kenny 1986, Pearl 2001), the mediator M is:
1. **Caused by T** — the treatment causes the mediator to activate
2. **Causes Y** — the mediator independently affects the outcome
3. **Measured independently** — M is observed from a separate data source, not extracted from the same text as T

In the MAS:
- M is extracted from the **same article** as T, by the **same LLM call**
- M is a **label** ("sell_pressure", "regulatory_clarity") chosen from a whitelist
- M is NOT independently measured from market microstructure data

This means M is not a mediator in the causal DAG sense. It's a **classification label** that the LLM assigns to describe the hypothesized causal pathway. The arrow T → M → Y is not a causal claim backed by data — it's a **linguistic annotation** of what the article *says* happens.

### What This Means For Mediation Analysis

§2.7 line 149 claims:

> *"Mechanism M: Mediator T → M → Y — Enables mediation analysis"*

True mediation analysis requires estimating:
```
NIE(t) = E[Y(t, M(t))] - E[Y(t, M(0))]    (Natural Indirect Effect)
NDE(t) = E[Y(t, M(0))] - E[Y(0, M(0))]    (Natural Direct Effect)
```

This requires **interventional data on M** — setting M to different values while holding T fixed. But since M is extracted from the same text as T, you cannot intervene on M independently. Two articles with the same T but different M are different because they describe different events, not because M was experimentally varied.

**Severity: MODERATE** — The mediation language is aspirational, not currently actionable. M is useful as a descriptive feature, not as a causal mediator.

---

## Defect 6 — MODERATE: SUTVA Violation Is Structural, Not Edge-Case

§2.9 line 169 acknowledges:

> *"SUTVA: Each article's outcome is independent. Violated when multiple articles about the same event cluster."*

This understates the severity. SUTVA (Stable Unit Treatment Value Assumption) requires:

1. **No interference:** Article i's treatment doesn't affect article j's outcome
2. **No hidden versions of treatment:** Treatment t is well-defined (no variation within t)

Both are **structurally violated** in financial news:

### Interference

When 50 articles about the same ETF ruling are published within 2 hours, they collectively move the market. Each article's marginal contribution to Y is negligible; the **aggregate treatment** matters. The current DAG treats each article as an independent observation, which is false.

### Hidden Versions

`treatment_class = "Regulation.etf_ruling"` encompasses:
- Full ETF approval
- Preliminary ETF filing acknowledgment
- ETF delay announcement
- ETF rejection

These are all classified as the same treatment but have drastically different causal effects. The 34-treatment taxonomy is too coarse to satisfy SUTVA's "no hidden versions" requirement.

**Severity: MODERATE** — The current 34-category taxonomy creates within-treatment heterogeneity that violates SUTVA. Consider: is L2 granular enough, or do you need L3?

---

## Defect 7 — MODERATE: The RAG "Formal Claim" Is Circular

§2.6 line 141:

> *"If g(Z, X; θ) satisfies the backdoor criterion, then g(Z, X, R(Z); θ) also satisfies it, because R is a deterministic function of Z"*

This claim has two problems:

### Problem A: The premise is false
As shown in Defect 3, `g(Z, X; θ)` does **not** satisfy the backdoor criterion (the U-path is unblocked). So the conditional "if g satisfies..." has a false antecedent.

### Problem B: R is not purely a function of Z
R depends on the Vector Search index state, which contains **all past accepted extractions**. These past extractions were influenced by past market conditions, past articles, and the LLM's past non-deterministic outputs. So:

```
R_i = retrieve(embed(Z_i), VS_index(T_1, T_2, ..., T_{i-1}))
```

R is a function of Z_i AND the entire extraction history. If past extractions were influenced by U (through U → Z → T), then R carries information about U into the current extraction. This creates a subtle **temporal confounding** path:

```
U_{past} → Z_{past} → T_{past} → VS_index → R_i → T_i
```

The VS index acts as a **memory** that can propagate past confounding into current extractions.

**Severity: MODERATE** — The formal claim needs re-grounding. R is not confounding-free; it inherits the quality of past extractions.

---

## Verdict Summary

| # | Defect | Severity | Impact |
|---|--------|----------|--------|
| 1 | Ontological/epistemological conflation | 🔴 **Fatal** | The DAG does not represent the real-world causal structure |
| 2 | Z is not a valid instrument | 🔴 **Serious** | Exclusion restriction and independence both violated |
| 3 | Incomplete backdoor adjustment | 🔴 **Serious** | U-path T←U→Y is not blocked by X |
| 4 | Missing edges (X→Z, Y→Z, Z→Y direct) | 🔴 **Serious** | DAG encodes false independence assumptions |
| 5 | M is not a standard mediator | 🟡 **Moderate** | Mediation analysis claims are not actionable |
| 6 | SUTVA violation is structural | 🟡 **Moderate** | 34-category taxonomy too coarse; event clustering |
| 7 | RAG formal claim is circular | 🟡 **Moderate** | VS index can propagate temporal confounding |

---

## Proposed Corrected DAG

### Two-Layer Architecture

Separate the **real-world causal structure** from the **measurement process**:

```
═══════════════════════════════════════════════════════════
  LAYER 1: REAL-WORLD CAUSAL STRUCTURE (the data-generating process)
═══════════════════════════════════════════════════════════

         U (unobserved confounders)
        ╱ ╲
       v   v
  X → E → Y        ← this is what we want to estimate
       |   ^
       |   |
       v   |
       Z ──┘         ← Z has a DIRECT effect on Y (information revelation)
       ^
       |
       X              ← market state influences news coverage (selection bias)

  Variables:
    E = real-world event (the actual halving, hack, ETF decision)
    Z = news article (report of E, influenced by X and U)
    T = E's class in our taxonomy (the TRUE treatment, partially observed via Z)
    X = pre-treatment market state
    Y = price outcome
    U = unobservable confounders

═══════════════════════════════════════════════════════════
  LAYER 2: MEASUREMENT PROCESS (the MAS extraction pipeline)
═══════════════════════════════════════════════════════════

  Z ──[MAS: g(Z, X, R; θ)]──> T̂ + ε_T

    T̂ = the MAS's ESTIMATE of E's true treatment class
    ε_T = measurement error (LLM non-determinism + classification error)
    
  The MAS does NOT produce T. It produces T̂.
  Downstream causal inference must account for measurement error in T̂.
```

### What Changes With The Corrected DAG

| Current Claim | Corrected Understanding |
|---------------|------------------------|
| "Z causes T" | Z is a report of E; the MAS *measures* T̂ from Z |
| "X blocks all backdoor paths" | X blocks T←X→Y but NOT T←U→Y |
| "Z is an instrument" | Z violates exclusion (Z→Y direct) and independence (U→Z) |
| "M is a mediator" | M is a descriptive label, not an independently measured mediator |
| "RAG preserves identification" | RAG inherits confounding from past extractions via VS index |
| "Backdoor adjustment identifies ATE" | Only identifies ATE under **additional** assumption of no unmeasured confounding |

### Recommended Actions

1. **Rewrite §2.1 with a two-layer DAG** that honestly separates the real-world structure from measurement
2. **Replace the instrument claim** in §2.6 with an honest statement: *"Z is the primary data source for T̂ estimation. Z is not a formal instrument because it has direct effects on Y and shares confounders with Y through U."*
3. **Rewrite §2.4 to state:** *"Backdoor adjustment with X identifies ATE(t) ONLY under the additional assumption of no unmeasured confounding (ignorability). Since U is present, we require sensitivity analysis (e.g., Rosenbaum bounds, E-value) to assess the robustness of causal estimates to violations of this assumption."*
4. **Rename M from "mediator" to "mechanism label"** and clarify that mediation analysis requires independent measurement of M from market microstructure data, which is a V4+ capability
5. **Add the missing edges** (X→Z, Z→Y_direct, Y→Z_reflexivity) and discuss their implications
6. **Introduce measurement error notation:** Use T̂ instead of T when referring to MAS output, and T for the true (partially latent) treatment
7. **Address SUTVA structurally:** Add L3 subcategories or event-level deduplication to reduce within-treatment heterogeneity

---

## What IS Robust About The Current Design

Despite the defects above, several architectural choices are genuinely strong:

| Feature | Why It's Good |
|---------|--------------|
| **Discrete T (taxonomy)** | Finite treatment space enables stratified estimation; far better than continuous sentiment scores |
| **Deterministic validator** | Prevents garbage extractions from entering the causal feature set — genuine quality control |
| **`rejected_mechanism_raw`** | Honest capture of what the model doesn't know — essential for governance |
| **Pre-treatment X measurement** | Correct temporal ordering prevents post-treatment bias |
| **Gatekeeper as T=∅ assignment** | Creates a natural comparison group — good causal design intuition |
| **VS filter: accepted only** | Prevents poison loops in the RAG index |
| **CausalBERT connection** | Correct dual-loss formulation for causally sufficient embeddings |

The *engineering* is strong. The *causal formalism* needs correction to match the engineering's actual capabilities.
