# LLM-Powered Validation Judge — Design Specification

**Date:** 2026-04-14
**Status:** Design — Awaiting Review
**Depends on:** `2026-04-07-mechanism-validator-design.md` (deterministic validator)
**Supersedes (partially):** Module B cosine-similarity relevance scoring becomes one input of many
**Related tickets:** TRD-30 (deterministic validator), TRD-57 (to be created)

---

## 1. Context

### 1.1 Problem

The current `mechanism_validator` is deterministic: whitelist lookup for `(treatment_l1, mechanism)`, whitelist check for `expected_direction`, substring match for `evidence_spans`, and an informational cosine-similarity score for evidence grounding. This catches two classes of extractor error well:

- **Whitelist violations** — LLM invents a non-taxonomy mechanism.
- **Fabricated evidence** — LLM quotes text not in the article.

It does **not** catch the class of error that matters most for downstream quant ML: **LLM-plausible-but-wrong extractions**. Concretely:

- LLM picks a valid whitelist mechanism that is not the most likely pathway for this specific article (e.g., labels a supply-side unlock event as `institutional_inflow`).
- LLM picks a valid direction that is semantically at odds with the article's tone (e.g., labels a lawsuit announcement `bullish` because it triggers short-term volatility).
- LLM extracts evidence spans that exist verbatim in the article but do not actually support the claim (they're adjacent text the LLM anchored on).

These errors pass the deterministic validator with `validation_status="accepted"`, then feed the triple-barrier labeler (TRD-44) as poisoned labels. In low-SNR regimes like crypto news → price, label noise compounds nonlinearly — one mislabeled high-confidence event can materially shift a Sharpe estimate.

### 1.2 Goal

Add an LLM-powered judge layer **after** the deterministic validator that catches this error class, optimizing for **precision** (minimizing false accepts). Multi-dimensional judge scores are written to the gold table as features for downstream ML consumption, enabling de Prado-style meta-labeling patterns.

### 1.3 Non-goals

- **Improving recall.** False rejects from strict substring matching are addressed separately by adding unicode normalization + fuzzy containment to the deterministic evidence check. No LLM work there.
- **Replacing the deterministic validator.** The deterministic validator remains the cheap first-pass gate.
- **Training a custom judge model.** V1 uses prompted Databricks-hosted Llama or Claude via Model Serving. Fine-tuning is out of scope.
- **Real-time reflection/repair loops.** Judge verdicts are terminal; rejected rows are not retried. Rejected extractions become training signal for offline prompt iteration.

---

## 2. Design Principles

1. **Precision-first with explicit recall floor.** Threshold selection maximizes precision subject to `recall ≥ 0.7` on calibration data. Downstream ML punishes false accepts asymmetrically more than false rejects.

2. **Strict-AND aggregation.** An extraction is `accepted` only if **all three** judges (Direction, Mechanism, Grounding) independently pass their thresholds. Any single rejection is terminal.

3. **Decomposed rubrics, not holistic judging.** Three narrow specialist prompts outperform one generalist prompt on rubric tasks. Each judge answers one question, emits one score + one verdict + one short rationale.

4. **Coverage-as-features, not coverage-as-filters.** Each judge's numeric score is written to the gold table as an ML feature, regardless of verdict. Downstream models can learn nonlinear combinations without being gated on them.

5. **Attributable rejections.** Rejection reasons are categorical (`rejected_judge_direction`, `rejected_judge_mechanism`, `rejected_judge_grounding`), enabling per-axis prompt iteration and monitoring.

6. **Fail conservative on LLM errors.** Timeout / JSON parse failure after retries → `failed_judge`, treated as rejection. Precision-first: drop the row rather than risk a false accept.

7. **Separate node, not mutation.** The judge lives in its own LangGraph node, not inside `mechanism_validator`. Keeps the deterministic gate auditable and lets us toggle the judge behind a feature flag without touching deterministic logic.

---

## 3. Architecture

### 3.1 Graph topology change

```
Before (current):
  context_retriever → few_shot_retriever → uncertainty_router
    → causality_extractor → mechanism_validator → END

After:
  context_retriever → few_shot_retriever → uncertainty_router
    → causality_extractor → mechanism_validator
    → [conditional: _route_after_deterministic]
    → llm_judge (if deterministic accepted)
    → END
```

`_route_after_deterministic` returns `"llm_judge"` if `validation_status == "accepted"`, else `"__end__"` (LangGraph END sentinel). This skips the judge for rows already rejected deterministically — no cost waste on certain-rejects.

### 3.2 Module layout

```
src/trading_databricks/agents/nodes/
├── mechanism_validator.py          # unchanged (deterministic gate)
├── llm_judge.py                    # NEW
│   ├── JudgeInput (TypedDict)
│   ├── JudgeScore (TypedDict)        # {score, verdict, reasoning}
│   ├── JudgeResult (TypedDict)       # aggregated output
│   ├── direction_judge()             # single-responsibility judge
│   ├── mechanism_judge()             # single-responsibility judge
│   ├── grounding_judge()             # single-responsibility judge
│   ├── _aggregate_judges()           # strict-AND + score merge
│   ├── _call_judge_with_retry()      # 2x exponential backoff
│   └── llm_judge_node()              # LangGraph bridge
└── ...

src/trading_databricks/agents/prompts/
├── __init__.py                     # re-exports for backward import paths
├── causality_extractor.py          # RELOCATED from agents/prompts.py
├── direction_judge.py              # NEW — prompt template
├── mechanism_judge.py              # NEW — prompt template
└── grounding_judge.py              # NEW — prompt template
```

`agents/prompts.py` is converted into a package (`agents/prompts/`). The existing extractor prompts move to `prompts/causality_extractor.py` in the same commit; `prompts/__init__.py` re-exports `build_system_prompt` etc. so existing imports keep working. This colocates all prompt assets and lets judge prompts evolve independently.

**Prompt template framework:** Migrate to `ChatPromptTemplate` in a follow-on refactor.

Each prompt module exports `SYSTEM_PROMPT: ChatPromptTemplate` (or `str` for the legacy extractor) and `PROMPT_HASH: str` (SHA-256 of the serialized template, for audit and for downstream ML to filter by prompt version).

### 3.3 Judge call pattern

Each judge is a single LLM call with structured JSON output. No chain-of-thought — the rubric is the prompt, and the LLM emits `{score: float, verdict: "accept"|"reject", reasoning: str}` directly. Chain-of-thought is deliberately avoided because it trends toward post-hoc rationalization in judge tasks.

The three judges run **sequentially** in V1 (simpler error handling, deterministic ordering in traces). If latency becomes a production bottleneck, parallelize via `asyncio.gather` in V2 — the calls have no data dependencies between each other.

---

## 4. The Three Judges

### 4.1 Direction Judge

**Question asked:** *"Given the full article, would a markets-literate reader classify this event's crypto-price impact as bullish / bearish / volatile? Does that match the extracted direction `{expected_direction}`?"*

**Inputs:** `headline`, `body`, `trigger_event`, `expected_direction`.

**Output JSON:**
```json
{
  "judged_direction": "bullish|bearish|volatile",
  "score": 0.87,
  "verdict": "accept|reject",
  "reasoning": "Single sentence, max 200 chars."
}
```

**Verdict rule:** `accept` if `judged_direction == expected_direction` AND `score ≥ direction_threshold`. Threshold is calibrated against CryptoLin (§6.1).

**Failure mode caught:** Extractor labels an event `bullish` when a markets-literate reader would call it `bearish` or `volatile`. Calibration shows this is the dominant error in FAST-path extractions where VS similarity anchored the LLM on an analogous-but-differently-signed exemplar.

**Score semantics:** `score` is the judge's calibrated confidence in its own `judged_direction`, **not** in the agreement with the extraction. This matters: a confident disagreement (`score=0.95`, `judged_direction != expected_direction`) is stronger signal than an uncertain agreement.

### 4.2 Mechanism Judge

**Question asked:** *"The extractor proposed `mechanism = {mechanism}` for treatment class `{treatment_class}`. Rank this mechanism against these 2 alternatives from the whitelist: {alt_1, alt_2}. Here are past rejected mechanism selections for similar articles — avoid analogous errors: {past_mistakes}. Which is the most plausible causal pathway given the article?"*

**Inputs:** `headline`, `body`, `treatment_class`, `mechanism`, `trigger_event`, 2 alternative mechanisms sampled from the same `treatment_l1` in the whitelist, **1–2 retrieved past-mistake exemplars** (see §4.2.1).

**Output JSON:**
```json
{
  "ranking": ["mechanism_a", "mechanism_b", "mechanism_c"],
  "score": 0.72,
  "verdict": "accept|reject",
  "reasoning": "Single sentence, max 200 chars."
}
```

**Verdict rule:** `accept` iff `ranking[0] == mechanism` AND `score ≥ mechanism_threshold`. The ranking format forces the judge to *compare*, not just rubber-stamp — contrastive ranking empirically resists rationalization better than Likert scoring (this echoes Approach 2 from the brainstorm, folded in as the Mechanism Judge's internal mechanism).

**Alternative sampling:** 2 alternatives are deterministically sampled per `(story_id, treatment_l1)` using a seeded hash to keep traces reproducible. If a whitelist `treatment_l1` has fewer than 3 mechanisms total, use all others; if fewer than 2, skip the Mechanism Judge for this row and mark `mechanism_score = None, mechanism_verdict = "n/a"` (not a rejection — absent signal). Downstream ML treats `None` as missing feature.

**Failure mode caught:** Extractor picks a valid whitelist mechanism that's plausible in the abstract but not the most likely pathway for this article. Contrastive ranking forces the judge to explicitly consider alternatives.

#### 4.2.1 Past-mistake retrieval (negative exemplars)

The Mechanism Judge prompt includes 1–2 retrieved *past-rejected* extractions that serve as adversarial counter-exemplars — telling the judge explicitly what kinds of errors have been made for similar articles in this L1 category, so it recognizes analogous errors in the current extraction.

**Retrieval source:** `gold.news_causal_features` rows where `validation_status IN ('rejected_mechanism', 'rejected_judge_mechanism')` AND `treatment_class` starts with the same L1 as the current extraction.

**Query:** semantic similarity (VS cosine) between current article's `headline` and the past mistake's `headline`, `num_results=2`.

**Prompt format** (per retrieved mistake):

```text
PAST MISTAKE — L1 "{l1}", headline: "{past_headline}"
  Wrong mechanism: {past_mechanism}
  Reason it was wrong: {past_rejection_reason or "not in whitelist"}
```

**Cold-start:** if `gold.news_causal_features` has < 20 rejected rows for this L1, skip past-mistake retrieval for that L1. Document this as the judge's "warm-up phase."

**Cost delta:** ~100 tokens added to Mechanism Judge prompt, no extra LLM call. Vector Search query is cheap (~10ms). Self-reinforcing: as the judge rejects extractions, the mistake bank grows, and the judge sharpens.

**What this is NOT:** This is retrieval-augmented guardrailing with negative exemplars — a well-supported prompting pattern. It is not front-door adjustment or any other causal-inference formalism (see §13.6 for the CAPITAL framework consideration and why its full apparatus was rejected).

### 4.3 Grounding Judge

**Question asked:** *"Do the quoted evidence spans explicitly support the claim `{trigger_event} → {mechanism} → {expected_direction}`, or only loosely relate to it?"*

**Inputs:** `headline`, `body`, `trigger_event`, `mechanism`, `expected_direction`, `evidence_spans` (list).

**Output JSON:**
```json
{
  "support_level": "explicit|implicit|unrelated",
  "score": 0.80,
  "verdict": "accept|reject",
  "reasoning": "Single sentence, max 200 chars."
}
```

**Verdict rule:** `accept` iff `support_level == "explicit"` AND `score ≥ grounding_threshold`. `implicit` always rejects — if evidence is only implicit, the extraction does not meet our precision bar for ML labeling.

**Failure mode caught:** Evidence spans pass the deterministic substring check (they exist in the article) but the chosen quotes do not actually ground the causal claim — they are adjacent text the LLM anchored on.

**Interaction with existing `evidence_relevance_score`:** The existing cosine-similarity score (Module B from the prior spec) remains computed and written to the gold table. The Grounding Judge is a richer, categorical signal; Module B is a dense continuous score. Keeping both gives downstream ML two features with different failure modes.

---

## 5. Aggregation and Decision Logic

### 5.1 Strict-AND aggregation

```python
def _aggregate_judges(
    direction: JudgeScore,
    mechanism: JudgeScore,
    grounding: JudgeScore,
) -> tuple[str, str | None]:
    """Return (validation_status, judge_rejection_reason)."""
    # Order matters: report the most actionable rejection reason.
    if direction["verdict"] == "reject":
        return "rejected_judge_direction", direction["reasoning"]
    if grounding["verdict"] == "reject":
        return "rejected_judge_grounding", grounding["reasoning"]
    if mechanism["verdict"] == "reject":
        return "rejected_judge_mechanism", mechanism["reasoning"]
    return "accepted", None
```

Ordering of the three checks is deliberate: Direction rejection is the most information-dense failure (label itself is wrong, so no point arguing about mechanism or evidence). Grounding before Mechanism because grounding failures are fact-based (spans don't support claim) while mechanism failures are judgment-based (wrong-but-plausible pathway).

### 5.2 Extended `validation_status` vocabulary

| Status | Source | Meaning |
|---|---|---|
| `accepted` | deterministic + judge | Passes all 4 checks |
| `rejected_mechanism` | deterministic | Mechanism not in whitelist for L1 |
| `rejected_direction` | deterministic | Direction not allowed for (L1, mechanism) |
| `rejected_evidence` | deterministic | Evidence spans not in article body |
| `rejected_judge_direction` | LLM judge | Direction Judge verdict=reject |
| `rejected_judge_mechanism` | LLM judge | Mechanism Judge verdict=reject |
| `rejected_judge_grounding` | LLM judge | Grounding Judge verdict=reject |
| `failed_extraction` | extractor | LLM returned no valid JSON after n retries (NEW — fixes silent-failure bug) |
| `failed_judge` | LLM judge | All retries exhausted on at least one judge |

Downstream ML (TRD-44 labeling) filters `validation_status = 'accepted'`. Telemetry dashboards break down rejections by reason.

### 5.3 Threshold parameters

Three float thresholds, all in `[0.0, 1.0]`:

```python
@dataclass(frozen=True)
class LLMJudgeConfig:
    direction_threshold: float = 0.70       # calibrated against CryptoLin
    mechanism_threshold: float = 0.65       # calibrated against Option-B bootstrap
    grounding_threshold: float = 0.75       # calibrated against FinCausal
    judge_llm_endpoint: str = "databricks-claude-sonnet-4"  # stronger than extractor's llama-3-1-8b
    max_retries: int = 2
    retry_base_delay_s: float = 1.0
    enabled: bool = True                    # feature flag for rollout
    judge_model_version: str = "v1.0"
    prompt_version_hashes: dict[str, str] = field(default_factory=dict)  # populated at boot
```

Initial threshold values are placeholders. §6 specifies the calibration methodology that determines them.

---

## 6. Calibration Data Plan

**Terminology:** *Option-B bootstrap* refers to using a stronger LLM (Claude Opus 4.6) as a weak-supervision label generator, with ≥ 20% human spot-check verification before the labels are trusted for threshold calibration. This trades labeling labor for calibration volume at the cost of accepting the stronger LLM's biases — made explicit in the spec so downstream readers understand why any calibrated threshold is "v1, to be refined with human labels."

### 6.1 Direction Judge — CryptoLin + FinRL

**Primary source:** CryptoLin (`github.com/manoelgadi/CryptoLin`), 2,683 human-annotated crypto news titles, 42-month span (Jul 2018 – Jan 2022), 3-annotator blind annotation with minority-to-neutral tie-breaking. Inter-rater agreement (Fleiss κ, Krippendorff α, Gwet AC1) reported acceptable in the source paper.

**Mapping to our 3-class direction:**
```
CryptoLin "positive"  → direction = "bullish"
CryptoLin "negative"  → direction = "bearish"
CryptoLin "neutral"   → merge with FinRL risk score:
                          if FinRL.risk_score ≥ 4 for the same/similar article → "volatile"
                          else discard (ambiguous)
```

**Content scope: full body.** Calibration runs against full article bodies, not headlines alone, to match production inputs. If CryptoLin distributes only headlines, re-scrape bodies from the original URLs into a local Delta staging table (`smart-trading-dev.others.cryptolin_bodies`) with fetch-date and licensing notes. Re-scrape must respect robots.txt and publisher ToS; any URL that fails to resolve or falls under a paywall is dropped from the calibration split and logged. This removes the headlines-only calibration-vs-production domain gap called out in earlier drafts.

**Supplementary:** FinRL_BTC_news_signals (279 rows) used solely to operationalize the `volatile` class via `risk_score ≥ 4`. Cross-dataset consistency spot-checks on the Apr 2021 – Jan 2022 overlap window (~100 rows).

**Split:** 70/30 train/test. Test split is held back for final threshold selection; train split is used for prompt iteration only.

**Threshold selection procedure:**

1. Run Direction Judge against train split (~1,900 rows after mapping + discards).
2. For each candidate threshold `t ∈ {0.5, 0.55, 0.6, ..., 0.95}`:
   - Compute precision and recall on the "accept" class.
   - Record `(t, precision, recall)`.
3. Select `t* = argmin_t {t : precision(t) ≥ 0.90, recall(t) ≥ 0.70}`.
4. Validate on test split: report precision, recall, F1.
5. If test-split precision < 0.88, flag and raise threshold or re-iterate prompt.

**Sensitivity analysis:** Repeat steps 1–4 under two alternative sentiment → direction mappings (e.g., treating CryptoLin "neutral" always as volatile vs. always discarding) to surface mapping-induced threshold brittleness. Report three threshold values in the commit; use the most conservative.

### 6.2 Grounding Judge — FinCausal + CNC + Option-B refinement

**Primary source:** FinCausal 2022 / 2025 cause-effect span annotations. Each labeled sample is `(text, cause_span, effect_span)`.

**Calibration task framing:** Given a FinCausal sample, construct a synthetic `(claim = cause_span + " causes " + effect_span, evidence_spans = [cause_span])` pair and ask the Grounding Judge to verify. Expected verdict: `explicit` support. Measure true positive rate.

Construct negative pairs by pairing `cause_span` with the `effect_span` from a **different random sample** in the same document. Expected verdict: `unrelated`. Measure true negative rate.

**Threshold:** select `t*` minimizing false-positive rate subject to true-positive rate ≥ 0.70 on held-out split.

**Supplementary:** Causal News Corpus (3,767 news sentences with ARG0/ARG1/SIG0 spans) for cross-genre robustness check. Run the Grounding Judge against CNC positive/negative pairs constructed as above; if F1 degrades > 0.15 vs FinCausal, document the genre gap.

**Domain-specific refinement (Option-B bootstrap):**
- Sample 100 articles from `silver.news_cleaned` stratified across L1 categories.
- Run a strong-LLM labeler (Claude Opus 4.6) to produce `(claim, evidence_span, supports: explicit|implicit|unrelated)` labels.
- Human spot-check 20% (20 articles) to verify strong-LLM label quality; acceptable if spot-check agreement ≥ 0.85.
- Use this seed as the crypto-domain refinement set; final Grounding Judge threshold is re-validated on it.

### 6.3 Mechanism Judge — Option-B bootstrap only (weak-calibration v1)

No public dataset uses our mechanism taxonomy (`institutional_access`, `liquidity_contraction`, etc.). Pure Option-B.

**Procedure:**
1. Sample 200 articles from `silver.news_cleaned` stratified across the 8 L1 categories (25 per category).
2. Run strong-LLM labeler (Claude Opus 4.6) to produce `(treatment_class, mechanism)` labels — with access to the full whitelist.
3. Human spot-check 20% (40 articles) for label quality; acceptable if agreement ≥ 0.80.
4. Split 70/30; calibrate Mechanism Judge threshold per §6.1 procedure.
5. **Cross-check against whitelist gaps:** any Option-B label where the strong LLM selected a mechanism outside the whitelist is logged as a *taxonomy gap candidate* — not a label error. These feed a weekly whitelist review.

**Acceptance criterion for v1 ship:** Mechanism Judge precision ≥ 0.80 on its bootstrap test split. Lower bar than Direction (0.90) and Grounding (0.85) because the calibration source is LLM-bootstrapped, not human-labeled. Explicitly noted as a v1 concession.

**v2 plan:** human-refinement of the 200-row seed (2–4 days labeling), then re-tune threshold targeting precision ≥ 0.90.

### 6.4 Pre-calibration requirements (spec blockers)

| Item | Owner | Blocker? |
|---|---|---|
| Verify CryptoLin license permits internal calibration use | Aaron | **Yes** — email author if unclear |
| Verify CryptoLin body vs headline-only content | Aaron | **Yes** — affects prompt design |
| Verify FinCausal 2022 / 2025 license (public dataset, likely permissive) | Aaron | Low — likely fine |
| Dedupe CryptoLin against `silver.news_cleaned` by URL + title hash | Aaron | **Yes** — prevents train/test leakage |

---

## 7. Data Flow

### 7.1 Input state fields

All consumed from existing `MASState`:

- `headline: str`
- `body: str`
- `extraction: dict | None` (from causality_extractor)
- `validation_status: str` (from mechanism_validator — must be `"accepted"` to reach this node)
- Whitelist loaded via config (same mechanism as mechanism_validator caches)

### 7.2 Output state fields (additions to `MASState`)

```python
class MASState(TypedDict, total=False):
    # ... existing fields ...

    # --- LLM judge (llm_judge) ---
    direction_judge_score: float | None
    direction_judge_verdict: str | None            # "accept" | "reject" | "error"
    mechanism_judge_score: float | None
    mechanism_judge_verdict: str | None            # "accept" | "reject" | "n/a" | "error"
    grounding_judge_score: float | None
    grounding_judge_verdict: str | None            # "accept" | "reject" | "error"
    judge_rejection_reason: str | None             # short rationale when rejected
    judge_model_version: str | None                # from LLMJudgeConfig
```

The LLM judge node may OVERWRITE `validation_status` from `"accepted"` to `"rejected_judge_*"` or `"failed_judge"`. It never changes any deterministic rejection status.

### 7.3 Gold table schema (additive changes)

New columns appended to `CAUSAL_FEATURES_SCHEMA`:

```python
StructField("direction_judge_score",    DoubleType(), nullable=True),
StructField("direction_judge_verdict",  StringType(), nullable=True),   # accept|reject|error
StructField("mechanism_judge_score",    DoubleType(), nullable=True),
StructField("mechanism_judge_verdict",  StringType(), nullable=True),   # accept|reject|n/a|error
StructField("grounding_judge_score",    DoubleType(), nullable=True),
StructField("grounding_judge_verdict",  StringType(), nullable=True),   # accept|reject|error
StructField("judge_rejection_reason",   StringType(), nullable=True),
StructField("judge_model_version",      StringType(), nullable=True),
StructField("prompt_version_hash",      StringType(), nullable=True),   # concat hash of all 3 judge prompts
```

All nullable. Existing rows get `NULL` for these columns. Backward-compatible.

**Verdict columns stored (not just scores):** A threshold change in a future config revision would silently reinterpret historical scores if verdicts aren't persisted. Storing verdicts pins the decision to the threshold that was active at write time, preserving label-to-verdict traceability for backtests.

**CLAUDE.md requirement:** Gold schema changes need explicit user approval. **This is flagged for explicit sign-off before implementation begins.**

### 7.4 Extractor silent-failure fix (bundled)

Prior audit surfaced a bug: when `causality_extractor` fails JSON parsing across all n samples, `extraction=None` flows to `mechanism_validator`, which returns `validation_status="rejected_evidence"` — mislabeling an extractor failure as an evidence failure.

This spec bundles the fix because it intersects the validation status vocabulary:
- `causality_extractor` sets `validation_status="failed_extraction"` directly when `extraction=None`.
- Graph routing: `_route_after_deterministic` returns END for any `validation_status.startswith("failed_")` or `validation_status != "accepted"`.

Minimal change, unblocks correct error attribution. No scope creep beyond this.

---

## 8. Error Handling

### 8.1 Retry policy

Each judge call wrapped in `_call_judge_with_retry`:
- 2 retries after first failure, exponential backoff with base 1.0s (so: 0s, 1s, 2s delays).
- Retry on: timeout, 5xx from Model Serving, JSON parse failure.
- No retry on: 4xx (likely malformed input or quota — log and fail fast).

After retries exhausted:
- Individual judge score set to `None`, verdict set to `"error"`.
- If ANY judge errors, final `validation_status = "failed_judge"`.
- All successful judge scores still written to gold (informational).

### 8.2 Conservative-fail semantics

`failed_judge` is a rejection, not an acceptance. Precision-first: dropping a row costs opportunity; accepting a row with an un-judged LLM extraction costs real P&L.

### 8.3 Timeout budget

Per-judge timeout: 10 seconds. Total per-article judge budget with 3 judges + 2 retries: worst case 90 seconds. Production target p95: 6 seconds (3 × 2s judge calls).

If p95 exceeds 10s in production, escalate parallelization to V2.

### 8.4 Graceful degradation feature flag

`LLMJudgeConfig.enabled = False` short-circuits the node to pass-through: `validation_status` remains whatever `mechanism_validator` produced, all judge fields set to `None`. Gives us a fast rollback lever without code deploy.

---

## 9. Cost and Latency Budget

### 9.1 Expected call volume

- News volume assumption: ~5,000 articles/day reach the MAS post-gatekeeper.
- Deterministic acceptance rate assumption: ~50% (needs empirical validation post-deploy).
- LLM judge volume: ~2,500 articles/day × 3 judges = ~7,500 LLM calls/day.

### 9.2 Model selection

- **V1 judge model:** Gemini 3.1 Pro, accessed via a Databricks external-model serving endpoint (`ai_query('gemini-3-1-pro-judge', ...)`). Strong reasoning + long context, chosen for precision on nuanced crypto-finance judgment where the extractor (`llama-3-1-8b-instruct`) is known to err.
- **Rationale:** a weaker judge than the extractor inverts the precision math — cheap judges miss the subtle errors that expensive extractors create. A frontier-tier judge against a small-model extractor is the configuration Zheng 2023 / G-Eval both validate.
- **Wiring dependency:** an external-model serving endpoint must be registered in the `smart-trading-v2` workspace pointing at Gemini 3.1 Pro with the project's API key in Databricks secrets. Tracked as a prerequisite task in the implementation plan.
- Challenger: `databricks-claude-sonnet-4` as a native Databricks-hosted challenger for cost/latency comparison in a shadow A/B (§10.2). If Gemini endpoint latency or quota becomes a production blocker, Claude Sonnet 4 is the pre-approved fallback with no spec change needed.

### 9.3 Cost ceiling

Daily LLM cost for judges estimated at ~$15–$40/day on Databricks-hosted Claude Sonnet 4 (7,500 calls × ~1k tokens). If observed cost exceeds $60/day for 3 consecutive days, trigger one of:
- Sample the judge (run on 50% of deterministic-accepted rows).
- Drop to cheaper challenger.
- Parallelize to reduce TCO (no cost reduction, just latency).

### 9.4 Latency SLO

- Per-article judge latency p95 ≤ 8s in V1.
- Per-article judge latency p99 ≤ 15s in V1.
- End-to-end MAS latency (deterministic + judge) p95 ≤ 18s.

Exceeding these triggers V2 parallelization work.

---

## 10. Observability

### 10.1 MLflow tracing

Every judge wrapped with `@mlflow.trace`. Spans include:
- Judge name (`direction_judge` / `mechanism_judge` / `grounding_judge`).
- Input hash (story_id + prompt_hash) — enables trace lookup.
- Output score, verdict, reasoning.
- Latency, retries used, token counts.

Aggregator span `llm_judge_node` wraps the three and records final aggregated verdict. This addresses a gap surfaced in the prior audit (no MLflow tracing on MAS nodes today).

### 10.2 Metrics dashboard (Lakeview)

Daily-partitioned metrics in a new Delta table `smart-trading-dev.03_gold.mas_judge_metrics`:

- Per-judge acceptance rate (rolling 7d).
- Per-judge score distribution (histogram buckets).
- Confusion matrix: `(validation_status_deterministic, validation_status_final)`.
- Failure rates: `failed_extraction`, `failed_judge`.
- Cost per 1k articles (tokens × price).
- p50 / p95 / p99 latency per judge.

### 10.3 Drift alarms

Weekly job compares judge score distributions to a 30-day baseline. Alert on:
- KS-test p-value < 0.01 on any judge's score distribution.
- Per-judge acceptance rate moves > 15 percentage points w/w.
- Cost/article moves > 30% w/w.

Drift usually means prompt-breaking model update upstream or taxonomy/data shift — both warrant investigation.

---

## 11. Rollout Plan

### 11.1 Phased deployment

**Phase 1 — Shadow (week 1):** Judge runs on 100% of deterministic-accepted rows, but its verdict is **not** applied. `validation_status` retains deterministic value. All judge fields written for analysis. Purpose: build empirical baseline for acceptance rates, calibrate thresholds against real production distribution (not just CryptoLin/FinCausal).

**Phase 2 — Enforced on 10% (week 2):** Judge verdict applied to a hashed 10% sample. Compare downstream ML feature quality (TRD-44 labeling stability) between sampled and unsampled rows.

**Phase 3 — Enforced on 100% (week 3+):** Full cutover. Monitor drift alarms for 2 weeks before declaring V1 stable.

### 11.2 Rollback

`LLMJudgeConfig.enabled = False` via config change, no redeploy needed. Gold table rows written during judge-enabled period retain their `validation_status` — no retroactive rewriting. Downstream ML may need to filter by processed_at if rollback happens.

### 11.3 Success criteria for V1

- Direction Judge precision ≥ 0.88 on test split.
- Mechanism Judge precision ≥ 0.80 on bootstrap test split.
- Grounding Judge precision ≥ 0.85 on FinCausal + crypto-refinement test splits.
- End-to-end acceptance rate drop vs pre-judge: 5–20 percentage points (too little = judge too lenient; too much = judge too strict).
- Downstream TRD-44 label stability (re-running on same input produces same label): ≥ 0.95 within a prompt_version_hash.

---

## 12. Testing Strategy

### 12.1 Unit tests (`tests/agents/test_llm_judge.py`)

- Each judge function mocked against a fake LLM client that returns canned JSON; asserts correct output shape and threshold application.
- `_aggregate_judges` exhaustively tested across 2^3 = 8 verdict combinations.
- `_call_judge_with_retry` tested with retry, timeout, and JSON parse failure scenarios.
- Feature flag pass-through: when `enabled=False`, node is a no-op and state is unchanged except for `None` judge fields.

### 12.2 Integration tests

- `test_graph.py` extended with end-to-end test: fake LLM returns structured JSON for each judge, full graph runs and asserts final `validation_status` and judge scores in state.
- Route test: deterministic-rejected row skips the judge (asserted via mock call count == 0).

### 12.3 Calibration regression tests

A golden-set test running weekly in CI:
- 50 hand-picked articles with known expected judge verdicts.
- Any judge prompt change must pass this regression before merge.
- Stored in `tests/agents/fixtures/judge_golden_set.jsonl`.

### 12.4 Prompt-change guardrails

Each prompt file exports `PROMPT_HASH = sha256(SYSTEM_PROMPT).hexdigest()[:12]`. CI enforces: if a prompt changes, the golden-set regression must be re-run AND the dev acknowledges threshold recalibration in the PR description.

---

## 13. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| CryptoLin license unclear | High | Email author; treat as research-only calibration; fall back to FinRL-only if blocked |
| Strong-LLM bootstrap labels biased (circularity) | Medium | Human spot-check 20% of bootstrap set before using; accept only if agreement ≥ 0.80 |
| Judge prompts rationalize incorrect extractions | Medium | Contrastive ranking in Mechanism Judge; structured-output rather than CoT; golden-set regression |
| Cost explosion under volume spike | Medium | Cost ceiling triggers with sampling fallback (§9.3) |
| Latency breaches SLO | Medium | Parallelize to V2 (async); cheaper challenger model swap |
| Silent prompt drift across model updates | Medium | KS-test drift alarm on score distributions; prompt_version_hash stamped on every row |
| Mechanism whitelist incomplete surfaces as false rejections | Low | Whitelist gap candidate log feeds weekly taxonomy review |
| V1 threshold miscalibration (too strict) | Low | Shadow phase catches this before enforcement; precision-over-recall by design |

---

### 13.6 Considered and rejected: CAPITAL (front-door adjustment) framework

A causal-inference-based validator framework (CAPITAL) was evaluated as an alternative. It proposes using Pearl's front-door adjustment over LLM chain-of-thought: generate A=20 diverse reasoning paths at high temperature, cluster into K=8 themes via K-means over embeddings, retrieve similar past-mistake CoTs per centroid, run N=5 classification calls per centroid, aggregate via P(Y|do(X)) = Σ_k P(t_k|do(X)) · P(Y|t_k).

**Rejected wholesale for the following reasons:**

1. **Scope mismatch.** CAPITAL is an extractor/generator. Our spec is a validator. Adopting it would replace `causality_extractor`, not augment `mechanism_validator`.
2. **Theoretical assumption is contested.** Front-door adjustment requires the mediator (CoT) to faithfully mediate X → Y. Turpin 2023 and Lanham 2023 show CoT is frequently post-hoc narrative, not a faithful causal intermediate. The math is rigorous but the underlying assumption is empirically shaky.
3. **K-means over CoT embeddings is heuristic, not causal.** Cluster centroids are not guaranteed to correspond to causally meaningful reasoning themes; clustering may pick up style variance.
4. **Cost prohibitive.** ~60 LLM calls per article (20 generation + 40 classification) vs our 3-judge committee's 3 calls. Exceeds §9 budget by 20×. At 5k articles/day this is 300k LLM calls/day.
5. **Non-determinism breaks quant ML reproducibility.** CAPITAL mandates high-temperature sampling. Re-running the same article produces different labels. TRD-44 triple-barrier labeling requires stable labels across runs for walk-forward backtesting integrity.
6. **Structured-extraction task mismatch.** CAPITAL outputs a scalar sentiment label. Our extraction is structured `(treatment_class, mechanism, direction, evidence_spans)` with whitelist constraints — front-door formalism does not cleanly handle these.

**Partial adoption:** CAPITAL's Phase 1 "memory bank of past mistakes" concept is genuinely useful and orthogonal to the front-door machinery. It has been folded into the Mechanism Judge as retrieval-augmented negative exemplars (§4.2.1) — a well-supported contrastive-prompting pattern, not a causal-inference claim.

## 14. Open Questions

1. **CryptoLin content scope:** Headlines-only or full body? If headlines-only, does the Direction Judge get calibrated on headline-only inputs (reality gap) or do we re-scrape bodies from URLs (extra effort + licensing concern on re-scrape)?
2. **Judge model choice (Databricks Claude Sonnet 4 vs alternatives):** Confirm availability and price-per-1k-tokens on `smart-trading-v2` workspace before committing.
3. **Prompt file location:** `src/trading_databricks/agents/prompts/` (new subdir) vs extending existing `prompts.py`? Subdir scales better but adds module count — lean subdir, confirm with team.
4. **Sampling policy in §9.3 fallback:** hash on `story_id` (deterministic, reproducible) or random (truer A/B distribution but harder to audit)? Lean deterministic hash.
5. **Feedback loop to prompt iteration:** Should rejected-by-judge rows feed an offline review queue for prompt tuning? Probably yes — but scope for v2 (need UI or notebook surface).

---

## 15. Out of Scope (V2 and Beyond)

- Asynchronous judge parallelization (latency optimization).
- Fine-tuned judge model (custom LoRA on crypto causality data).
- Active-learning loop: ML model disagreements trigger judge re-evaluation.
- Human-in-the-loop review of low-confidence judge decisions (e.g., score ∈ [threshold - 0.05, threshold + 0.05]).
- Meta-judge / debate architecture (Approach 4 from brainstorm) — deferred unless V1 precision ceiling proves too low.
- Cross-lingual support (CryptoLin is English-only; global news coverage would need multilingual judges).

---

## 16. References

- `2026-04-07-mechanism-validator-design.md` — deterministic validator, parent spec
- `2026-04-04-mas-final-spec.md` §7.1–7.2 — 5-step validator logic
- `2026-04-04-robust-trustworthy-mas-spec.md` §3.1–3.5 — Module B evidence grounding
- Nascimento et al., *A sentiment corpus for the cryptocurrency financial domain: the CryptoLin corpus*, LRE 2024
- Mariko et al., *The Financial Document Causality Detection Shared Task (FinCausal 2020 / 2022 / 2025)*, FNP Workshop
- Tan et al., *The Causal News Corpus: Annotating Causal Relations in Event Sentences from News*, LREC 2022
- de Prado, M.L., *Advances in Financial Machine Learning* — meta-labeling, triple-barrier, purged cross-validation

---
