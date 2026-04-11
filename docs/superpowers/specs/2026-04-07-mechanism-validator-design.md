# Mechanism Validator Node — Design Specification

**Date:** 2026-04-07  
**Wave:** 2 (TRD-30)  
**Status:** Design Approved  
**Implements:**
- `2026-04-04-mas-final-spec.md` §7.1-7.2 (5-step validator logic)
- `2026-04-04-robust-trustworthy-mas-spec.md` §3.1-3.5 (Module B evidence grounding)

---

## Context

The mechanism validator is the deterministic quality gate of the MAS pipeline. It validates each LLM extraction against the ontology (treatment taxonomy + mechanism whitelist) and captures governance data for whitelist expansion. It also implements Module B (Evidence Semantic Grounding), the ANM-inspired check that scores whether evidence spans are semantically relevant to the claimed causal chain.

**Why this exists:**

The upstream Causality Extractor (LangGraph node) produces unvalidated JSON with treatment_class, mechanism, direction, and evidence_spans. The validator ensures:

1. The mechanism is whitelisted for the treatment's L1 category
2. The direction is valid for that mechanism
3. The evidence spans actually exist in the article body (not fabricated)
4. The evidence is semantically relevant to the claim (Module B informational score)

**Key design decision:** Module B is informational only in V3. It computes a relevance score but does NOT reject chains. The score is written to the schema for downstream analysis. After collecting sufficient data, V4 may add a rejection threshold.

---

## 1. Architecture

### Module Organization

```
src/trading_databricks/agents/nodes/
├── __init__.py                    # exports: validate_extraction
└── mechanism_validator.py         # ~150 lines total
    ├── ExtractionInput (TypedDict)        # ~8 fields
    ├── ValidationResult (TypedDict)       # ~3 fields
    ├── validate_extraction()              # main orchestrator (~40 lines)
    ├── _parse_l1_category()               # L1 extraction (~5 lines)
    ├── _validate_evidence_spans()         # substring checks (~15 lines)
    ├── _compute_evidence_relevance()      # Module B with error handling (~25 lines)
    └── _cosine_similarity()               # NumPy helper (~10 lines)
```

### Design Principles

1. **No early returns**: Steps 1-4 determine validation status via if/elif/else chain, then step 5 runs unconditionally
2. **Module B resilience**: Embedding failures are caught, logged, and return `None` — validation continues
3. **Best-effort parsing**: Malformed `treatment_class` is parsed optimistically, fails naturally at whitelist lookup
4. **Degenerate case rejection**: Empty evidence, whitespace-only spans, or empty article body → `rejected_evidence`
5. **Validation-only return**: Function returns only validation fields; caller merges with extraction
6. **TypedDict contracts**: Static type checking for LangGraph state slices, zero runtime overhead

### Control Flow

```
validate_extraction(extraction, article_body, whitelist, config, embed_fn)
    │
    ├─ Step 1: Parse L1 from treatment_class
    │   └─→ l1 = _parse_l1_category(extraction["treatment_class"])
    │
    ├─ Steps 2-4: Determine validation_status (if/elif/else, NO returns)
    │   ├─ Step 2: (l1, mechanism) not in whitelist → rejected_mechanism
    │   ├─ Step 3: direction ∉ valid_directions → rejected_direction
    │   └─ Step 4: _validate_evidence_spans() → False → rejected_evidence
    │
    ├─ Step 5: ALWAYS runs (Module B — informational, even for rejected chains)
    │   └─→ score = _compute_evidence_relevance(...)
    │
    └─ Return: {validation_status, rejected_mechanism_raw, evidence_relevance_score}
```

**Critical invariant:** Step 5 runs for ALL validation statuses (accepted, rejected_mechanism, rejected_direction, rejected_evidence). This ensures `evidence_relevance_score` is "always populated for impactful articles (even for rejected chains)" per acceptance criteria.

---

## 2. Data Contracts

### Input: `ExtractionInput` (TypedDict)

```python
from typing import TypedDict

class ExtractionInput(TypedDict):
    """LLM extraction output consumed by the validator.
    
    Matches the subset of CAUSAL_FEATURES_SCHEMA fields that the
    extractor produces. The validator needs these 5 fields to run
    the 5-step validation pipeline.
    """
    treatment_class: str        # "L1.l2_subcategory"
    trigger_event: str          # max 200 chars, for Module B claim embedding
    mechanism: str              # lowercase snake_case
    expected_direction: str     # "bullish" | "bearish" | "volatile"
    evidence_spans: list[str]   # 1-3 quoted text fragments from article
```

**Design rationale:**
- Minimal contract: only fields needed for validation
- Does NOT include `confidence`, `asset`, `causal_chain_json` — validator doesn't use them
- TypedDict enables static type checking without runtime overhead
- Matches LangGraph state slice pattern

---

### Output: `ValidationResult` (TypedDict)

```python
class ValidationResult(TypedDict):
    """Validation-only fields returned by validate_extraction().
    
    Caller merges these into the full extraction dict before writing
    to gold.news_causal_features.
    """
    validation_status: str                    # "accepted" | "rejected_mechanism" | "rejected_direction" | "rejected_evidence"
    rejected_mechanism_raw: str | None        # populated only when validation_status = "rejected_mechanism"
    evidence_relevance_score: float | None    # [0.0, 1.0] cosine similarity
                                              # None only on: embed failure (caught exception) or empty evidence
```

**Design rationale:**
- Only 3 fields: status + governance field + Module B score
- `rejected_mechanism_raw` is `None` for all statuses except `rejected_mechanism`
- `evidence_relevance_score` is `None` when:
  - Embedding call fails (caught exception, logged, graceful degradation)
  - Evidence is empty/whitespace (cannot embed nothing)
- Caller pattern: `{**extraction, **validate_extraction(extraction, ...)}`

**Schema alignment:**
- `validation_status` → `CAUSAL_FEATURES_SCHEMA.validation_status` (STRING)
- `rejected_mechanism_raw` → `CAUSAL_FEATURES_SCHEMA.rejected_mechanism_raw` (STRING, nullable)
- `evidence_relevance_score` → `CAUSAL_FEATURES_SCHEMA.evidence_relevance_score` (DOUBLE, nullable)

---

### Whitelist Contract (Established in Wave 1)

```python
# From load_whitelist(), merged in TRD-23:
whitelist: dict[tuple[str, str], list[str]]

# Example:
{
    ("Regulation", "regulatory_clarity"): ["bullish"],
    ("Regulation", "regulatory_restriction"): ["bearish"],
    ("Security", "sell_pressure"): ["bearish"],
    ...
}
```

**Validator usage:**
- Lookup: `(l1, mechanism) in whitelist` — O(1) membership test
- Direction check: `direction in whitelist[(l1, mechanism)]` — O(k) where k ≤ 3

---

## 3. Validation Steps (1-4) — Deterministic Logic

### Step 1: Parse L1 Category

```python
def _parse_l1_category(treatment_class: str) -> str:
    """Extract L1 from treatment_class using best-effort parsing.
    
    Examples:
        "Regulation.etf_ruling"     → "Regulation"
        "Regulation"                → "Regulation"  (no dot, return as-is)
        "Regulation."               → "Regulation"  (trailing dot)
        "Regulation.etf.extra"      → "Regulation"  (split on first dot)
        ".etf_ruling"               → ""            (malformed, empty L1)
        ""                          → ""            (empty input)
    
    Malformed cases naturally fail at step 2 whitelist lookup.
    No exceptions raised — let downstream validation catch it.
    """
    if "." not in treatment_class:
        return treatment_class
    return treatment_class.split(".", maxsplit=1)[0]
```

**Design rationale:**
- `maxsplit=1`: Only split on first dot (handles "Regulation.etf.extra")
- Returns empty string for `.etf_ruling` → whitelist lookup fails naturally
- No validation here — separation of concerns
- Malformed treatment_class is a subset of "unknown mechanism" — rejected_mechanism captures it

---

### Step 2: Whitelist Lookup

**In main function:**
```python
l1 = _parse_l1_category(extraction["treatment_class"])
mechanism = extraction["mechanism"]

# Steps 2-4: Determine validation status (NO early returns)
if (l1, mechanism) not in whitelist:
    status = "rejected_mechanism"
    rejected_raw = mechanism  # governance field
elif extraction["expected_direction"] not in whitelist[(l1, mechanism)]:
    status = "rejected_direction"
    rejected_raw = None
elif not _validate_evidence_spans(extraction["evidence_spans"], article_body):
    status = "rejected_evidence"
    rejected_raw = None
else:
    status = "accepted"
    rejected_raw = None
```

**Captures:**
- Unknown mechanism (not in whitelist)
- Malformed L1 (empty string from step 1)
- Typos in either field

**Governance field:** `rejected_mechanism_raw` preserves the exact string the LLM produced. This enables SQL queries to discover clusters of rejected mechanisms for whitelist expansion.

---

### Step 3: Direction Validation

At this point in the if/elif chain, `(l1, mechanism)` is guaranteed to exist in whitelist (step 2 passed). The check `extraction["expected_direction"] not in whitelist[(l1, mechanism)]` validates that the LLM's direction is in the pre-approved list.

**Valid directions per mechanism:**
- Most mechanisms: single direction (e.g., `regulatory_clarity` → `["bullish"]`)
- Some mechanisms: bidirectional (e.g., `liquidity_expansion` → `["bullish", "volatile"]`)
- The validator enforces the whitelist contract — it does not interpret economic logic

---

### Step 4: Evidence Validation

```python
def _validate_evidence_spans(
    evidence_spans: list[str],
    article_body: str,
) -> bool:
    """Check that all evidence spans exist in article body (normalized).
    
    Normalization: casefold() + strip() — prevents false rejections from
    LLM whitespace/casing variations while catching fabricated evidence.
    
    Returns False (rejection) when:
    - evidence_spans is empty []
    - article_body is empty ""
    - Any span is empty or whitespace-only after normalization
    - Any span not found as substring of normalized article body
    """
    normed_body = article_body.casefold().strip()
    normed_spans = [s.casefold().strip() for s in evidence_spans]
    
    # Degenerate cases: empty evidence, empty body, or whitespace-only spans
    if not normed_spans or not normed_body or any(not s for s in normed_spans):
        return False
    
    # All spans must exist in body
    return all(span in normed_body for span in normed_spans)
```

**Design rationale:**
- Normalization with `casefold()` + `strip()` per spec §7.1: "normalized matching prevents false rejections from LLM whitespace/casing changes"
- Guard clause handles all degenerate cases in one check
- `all()` is more idiomatic than manual loop (per python-programing-rules.md)
- Substring match (not word boundaries) — allows partial quotes

**Edge cases handled:**

| Case | Input | Result | Why |
|------|-------|--------|-----|
| Empty spans | `[]` | `False` | No evidence = no grounding |
| Empty body | `""` | `False` | Can't validate against nothing |
| Whitespace span | `["   "]` | `False` | Vacuous truth is not evidence |
| Span normalizes to empty | `["  \n  "]` | `False` | Same as whitespace span |
| Case/whitespace variation | `["  Bitcoin PRICE  "]` in `"bitcoin price"` | `True` | Normalized match succeeds |

---

## 4. Module B — Evidence Semantic Grounding

### Theoretical Basis

**ANM (Additive Noise Model) analogy:**

In causal discovery, ANM determines direction by testing whether regression residuals are independent of the predictor. If X→Y is true, then `residuals(Y|X) ⊥ X`. If residuals depend on X, the model is wrong.

**MAS analogy:**

If evidence truly grounds the claim, embeddings should be semantically similar. Low similarity suggests the evidence is filler text or the LLM is hallucinating a connection.

---

### Implementation

```python
import logging
import numpy as np

logger = logging.getLogger(__name__)

def _compute_evidence_relevance(
    trigger_event: str,
    mechanism: str,
    evidence_spans: list[str],
    *,
    embed_fn: Callable[[str], list[float]],
) -> float | None:
    """Score semantic relevance between causal claim and evidence.
    
    ANM-inspired check: if evidence truly grounds the claim, embeddings
    should be semantically similar. Low similarity suggests evidence is
    filler text or the LLM is hallucinating.
    
    Implementation: Cosine similarity between:
    - Claim: "{trigger_event} causes {mechanism}"
    - Evidence: " ".join(evidence_spans)
    
    Returns:
        float in [0.0, 1.0] - cosine similarity score
        None - on edge cases (empty evidence) or embedding failures
    
    Error handling:
        Module B is informational only in V3. Embedding failures are
        caught, logged, and return None. Validation proceeds.
    """
    # Edge case: empty or whitespace-only evidence
    if not evidence_spans or all(not s.strip() for s in evidence_spans):
        return None  # Cannot score — no evidence to embed
    
    try:
        # Construct claim and evidence text
        claim_text = f"{trigger_event} causes {mechanism}"
        evidence_text = " ".join(evidence_spans)
        
        # Call embedding endpoint (2 calls, ~50 tokens each)
        claim_embedding = embed_fn(claim_text)
        evidence_embedding = embed_fn(evidence_text)
        
        # Compute cosine similarity
        similarity = _cosine_similarity(claim_embedding, evidence_embedding)
        
        return float(similarity)
    
    except Exception as e:
        # Graceful degradation: log and continue without score
        logger.warning(
            "Module B embedding failed, continuing without evidence_relevance_score",
            exc_info=True,
            extra={
                "trigger_event": trigger_event[:50],  # truncate for log size
                "mechanism": mechanism,
                "error": str(e),
            },
        )
        return None
```

---

### Cosine Similarity Helper

```python
def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors.
    
    Formula: cos(θ) = (A · B) / (||A|| × ||B||)
    
    Returns:
        float in [0.0, 1.0] - clamped to match schema contract
        
    Clamping rationale:
        Raw cosine similarity is [-1, 1], but negative values have no
        meaningful interpretation for "evidence relevance." Evidence
        semantically opposed to the claim has zero relevance, not
        negative relevance.
    
    Edge cases:
        - Zero-length vectors → 0.0 (no similarity)
        - Negative raw similarity → 0.0 (clamped)
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    raw_similarity = np.dot(a, b) / (norm_a * norm_b)
    return float(max(0.0, raw_similarity))  # Clamp to [0.0, 1.0]
```

**Design rationale:**
- Clamping to [0.0, 1.0] enforces schema contract (DOUBLE field is defined as "[0, 1]")
- Negative similarity has no semantic meaning for "evidence relevance"
- Zero-length vector handling prevents division by zero
- Returns `float` not `np.float64` for JSON serialization

---

### Error Handling Strategy

**Three failure modes:**

1. **Empty evidence** (edge case):
   - Guard at function start: `if not evidence_spans or all(not s.strip() ...): return None`
   - No exception, clean `None` return
   - Happens naturally for `rejected_evidence` with empty spans

2. **Embedding API failure** (network, timeout, rate limit):
   - Caught by outer `try/except`
   - Logged with `exc_info=True` for stack trace
   - Returns `None` — validation continues without score

3. **Cosine computation error** (NaN, dimension mismatch):
   - Caught by same outer `try/except`
   - Rare (requires malformed endpoint response)
   - Same graceful degradation

**Logging fields:**
- `trigger_event[:50]` — truncated to prevent log spam
- `mechanism` — full text (always short)
- `error` — exception string
- `exc_info=True` — full stack trace for debugging

**Why graceful degradation is correct:**

Module B is informational only in V3. It scores evidence quality but does NOT reject chains. If embedding fails, the validation status (accepted/rejected_*) is still valid — we just lack the Module B signal. The alternative (raising exceptions) would cause the entire validation to fail for transient network issues, which violates the "informational only" contract.

---

### Module B Cost

- 2 embedding calls per validation (claim + evidence)
- ~50 tokens each = 100 tokens total
- `databricks-gte-large-en` cost: ~$0.001 per call
- Total Module B cost: ~$0.002 per article

**Negligible compared to extractor LLM cost (~$0.03).**

---

## 5. Main Function Signature

```python
def validate_extraction(
    extraction: ExtractionInput,
    article_body: str,
    whitelist: dict[tuple[str, str], list[str]],
    *,
    config: MechanismValidatorConfig,
    embed_fn: Callable[[str], list[float]],
) -> ValidationResult:
    """Run 5-step validation. Steps 1-4 deterministic. Step 5 (Module B) informational.
    
    Args:
        extraction: LLM output with treatment_class, mechanism, direction, evidence
        article_body: Full article text for evidence substring matching
        whitelist: {(treatment_l1, mechanism): [valid_directions]} from load_whitelist()
        config: MechanismValidatorConfig (contains evidence_grounding_reject_threshold, unused in V3)
        embed_fn: Embedding function, signature (str) -> list[float], injected by caller
    
    Returns:
        ValidationResult with 3 fields:
            - validation_status: "accepted" | "rejected_mechanism" | "rejected_direction" | "rejected_evidence"
            - rejected_mechanism_raw: str | None (only for rejected_mechanism)
            - evidence_relevance_score: float [0, 1] | None (None on embed failure or empty evidence)
    
    Error handling:
        Module B embedding failures are caught and logged. The function always
        returns a ValidationResult — no uncaught exceptions propagate to caller.
    """
```

**Full implementation pseudocode:**

```python
def validate_extraction(...) -> ValidationResult:
    # Step 1: Parse L1 from treatment_class
    l1 = _parse_l1_category(extraction["treatment_class"])
    mechanism = extraction["mechanism"]
    
    # Steps 2-4: Determine validation status (NO early returns)
    if (l1, mechanism) not in whitelist:
        status = "rejected_mechanism"
        rejected_raw = mechanism
    elif extraction["expected_direction"] not in whitelist[(l1, mechanism)]:
        status = "rejected_direction"
        rejected_raw = None
    elif not _validate_evidence_spans(extraction["evidence_spans"], article_body):
        status = "rejected_evidence"
        rejected_raw = None
    else:
        status = "accepted"
        rejected_raw = None
    
    # Step 5: ALWAYS runs (Module B — informational, even for rejected chains)
    score = _compute_evidence_relevance(
        extraction["trigger_event"],
        mechanism,
        extraction["evidence_spans"],
        embed_fn=embed_fn,
    )
    
    return {
        "validation_status": status,
        "rejected_mechanism_raw": rejected_raw,
        "evidence_relevance_score": score,
    }
```

**Critical design point:** No early returns after steps 2-4. The if/elif/else chain determines status, then step 5 runs unconditionally. This ensures `evidence_relevance_score` is always computed (or explicitly set to `None` on errors), fulfilling the acceptance criteria "always populated for impactful articles (even for rejected chains)."

---

## 6. Integration Points

### LangGraph Node Wrapper

```python
# In src/trading_databricks/agents/graph.py (future implementation)
from trading_databricks.agents.nodes import validate_extraction
from trading_databricks.agents.taxonomy import load_whitelist

def mechanism_validator_node(state: MASState) -> MASState:
    """LangGraph node wrapper for validator."""
    
    # Load whitelist from Unity Catalog (cached at graph init)
    whitelist = state["whitelist"]  # Pre-loaded, not per-article
    
    # Validate extraction
    validation = validate_extraction(
        extraction=state["extraction"],
        article_body=state["article_body"],
        whitelist=whitelist,
        config=state["config"].validator,
        embed_fn=state["embed_fn"],  # Injected by graph builder
    )
    
    # Merge validation results into extraction
    return {
        **state,
        "extraction": {**state["extraction"], **validation},
    }
```

---

### Embedding Function Provider

```python
# In agents/tools.py (or graph builder)
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

def create_embedding_fn(endpoint_name: str) -> Callable[[str], list[float]]:
    """Factory for Databricks Model Serving embedding function."""
    w = WorkspaceClient()
    
    def embed(text: str) -> list[float]:
        response = w.serving_endpoints.query(
            name=endpoint_name,
            inputs=[text],
        )
        return response.data[0].embeddings  # type: ignore
    
    return embed

# Usage in graph builder:
embed_fn = create_embedding_fn("databricks-gte-large-en")
```

---

### Whitelist Loading Pattern

```python
# Caller (graph builder or SDP pipeline)
from trading_databricks.agents.taxonomy import load_whitelist

# One-time load at graph initialization (not per article)
whitelist = load_whitelist(
    spark,
    catalog="smart-trading-dev",
    schema="gold",
)  # Returns: dict[tuple[str, str], list[str]]

# Pass to validator via LangGraph state or function parameter
```

---

## 7. Testing Strategy

### Test Structure

```
tests/agents/
└── test_mechanism_validator.py        # ~200 lines
    ├── Unit tests for helpers (~80 lines)
    │   ├── test_parse_l1_category_*
    │   ├── test_validate_evidence_spans_*
    │   └── test_cosine_similarity_*
    │
    ├── Integration tests for validate_extraction (~100 lines)
    │   ├── test_accepted_chain
    │   ├── test_rejected_mechanism
    │   ├── test_rejected_direction
    │   ├── test_rejected_evidence
    │   └── test_module_b_edge_cases
    │
    └── Fixtures (~20 lines)
        ├── mock_whitelist
        ├── mock_embed_fn
        └── sample_extraction
```

---

### Key Test Cases

**1. Helper Functions (Pure Logic)**

```python
import pytest
from trading_databricks.agents.nodes.mechanism_validator import (
    _parse_l1_category,
    _validate_evidence_spans,
)

@pytest.mark.parametrize("treatment_class,expected", [
    ("Regulation.etf_ruling", "Regulation"),
    ("Regulation", "Regulation"),
    ("Regulation.", "Regulation"),
    ("Regulation.etf.extra", "Regulation"),
    (".etf_ruling", ""),
    ("", ""),
])
def test_parse_l1_category(treatment_class, expected):
    assert _parse_l1_category(treatment_class) == expected


def test_validate_evidence_spans_degenerate_cases():
    # Empty spans
    assert not _validate_evidence_spans([], "article body")
    
    # Empty body
    assert not _validate_evidence_spans(["span"], "")
    
    # Whitespace-only span
    assert not _validate_evidence_spans(["   "], "article body")
    
    # Span normalizes to empty
    assert not _validate_evidence_spans(["  \n  "], "article")


def test_validate_evidence_spans_normalized_matching():
    # Casefold + strip allows variations
    assert _validate_evidence_spans(
        ["  Bitcoin PRICE  "],
        "The bitcoin price is rising"
    )
    
    # Substring match (not word boundaries)
    assert _validate_evidence_spans(
        ["bitcoin"],
        "The bitcoin price"
    )
    
    # Fabricated evidence rejected
    assert not _validate_evidence_spans(
        ["ethereum price"],
        "The bitcoin price is rising"
    )
```

---

**2. Module B Error Handling**

```python
def test_compute_evidence_relevance_embed_failure():
    """When embed_fn raises, return None and log warning."""
    from trading_databricks.agents.nodes.mechanism_validator import (
        _compute_evidence_relevance,
    )
    
    def failing_embed_fn(text: str) -> list[float]:
        raise TimeoutError("Model endpoint unavailable")
    
    result = _compute_evidence_relevance(
        trigger_event="SEC approves ETF",
        mechanism="regulatory_clarity",
        evidence_spans=["The SEC approved..."],
        embed_fn=failing_embed_fn,
    )
    
    assert result is None  # Graceful degradation


def test_compute_evidence_relevance_empty_evidence():
    """Empty evidence returns None immediately, no embed call."""
    call_count = 0
    
    def counting_embed_fn(text: str) -> list[float]:
        nonlocal call_count
        call_count += 1
        return [0.0] * 384
    
    result = _compute_evidence_relevance(
        trigger_event="event",
        mechanism="mechanism",
        evidence_spans=[],  # Empty
        embed_fn=counting_embed_fn,
    )
    
    assert result is None
    assert call_count == 0  # Never called
```

---

**3. End-to-End Integration**

```python
def test_validate_extraction_accepted():
    """Valid extraction passes all checks."""
    from trading_databricks.agents.nodes import validate_extraction
    from trading_databricks.agents.config import MechanismValidatorConfig
    
    whitelist = {("Regulation", "regulatory_clarity"): ["bullish"]}
    
    extraction = {
        "treatment_class": "Regulation.etf_ruling",
        "trigger_event": "SEC approves ETF",
        "mechanism": "regulatory_clarity",
        "expected_direction": "bullish",
        "evidence_spans": ["The SEC approved the first spot ETF"],
    }
    
    result = validate_extraction(
        extraction,
        article_body="The SEC approved the first spot ETF for Bitcoin",
        whitelist=whitelist,
        config=MechanismValidatorConfig(),
        embed_fn=mock_embed_fn,
    )
    
    assert result["validation_status"] == "accepted"
    assert result["rejected_mechanism_raw"] is None
    assert result["evidence_relevance_score"] is not None
    assert 0.0 <= result["evidence_relevance_score"] <= 1.0


def test_validate_extraction_rejected_mechanism_still_scores():
    """Module B runs even for rejected chains."""
    whitelist = {("Regulation", "regulatory_clarity"): ["bullish"]}
    
    extraction = {
        "treatment_class": "Regulation.etf_ruling",
        "trigger_event": "SEC approves",
        "mechanism": "unknown_mechanism",  # Not in whitelist
        "expected_direction": "bullish",
        "evidence_spans": ["The SEC approved the first spot ETF"],
    }
    
    result = validate_extraction(
        extraction,
        article_body="The SEC approved the first spot ETF for Bitcoin",
        whitelist=whitelist,
        config=MechanismValidatorConfig(),
        embed_fn=mock_embed_fn,
    )
    
    assert result["validation_status"] == "rejected_mechanism"
    assert result["rejected_mechanism_raw"] == "unknown_mechanism"
    assert result["evidence_relevance_score"] is not None  # Module B still ran!
    assert 0.0 <= result["evidence_relevance_score"] <= 1.0


def test_validate_extraction_rejected_direction():
    """Invalid direction fails step 3."""
    whitelist = {("Regulation", "regulatory_clarity"): ["bullish"]}
    
    extraction = {
        "treatment_class": "Regulation.etf_ruling",
        "trigger_event": "SEC approves",
        "mechanism": "regulatory_clarity",
        "expected_direction": "bearish",  # Not in valid_directions
        "evidence_spans": ["The SEC approved"],
    }
    
    result = validate_extraction(
        extraction,
        article_body="The SEC approved the ETF",
        whitelist=whitelist,
        config=MechanismValidatorConfig(),
        embed_fn=mock_embed_fn,
    )
    
    assert result["validation_status"] == "rejected_direction"
    assert result["rejected_mechanism_raw"] is None
    assert result["evidence_relevance_score"] is not None  # Module B runs


def test_validate_extraction_rejected_evidence():
    """Evidence not in article body fails step 4."""
    whitelist = {("Regulation", "regulatory_clarity"): ["bullish"]}
    
    extraction = {
        "treatment_class": "Regulation.etf_ruling",
        "trigger_event": "SEC approves",
        "mechanism": "regulatory_clarity",
        "expected_direction": "bullish",
        "evidence_spans": ["ethereum price drops"],  # Not in article
    }
    
    result = validate_extraction(
        extraction,
        article_body="The SEC approved the Bitcoin ETF",
        whitelist=whitelist,
        config=MechanismValidatorConfig(),
        embed_fn=mock_embed_fn,
    )
    
    assert result["validation_status"] == "rejected_evidence"
    assert result["rejected_mechanism_raw"] is None
    # Module B tries to score, likely succeeds (spans exist as strings)
    # or returns None if embed_fn rejects the text
```

---

### Mock Fixtures

```python
import numpy as np

@pytest.fixture
def mock_embed_fn():
    """Deterministic embedding function for testing."""
    def _embed(text: str) -> list[float]:
        # Simple hash-based mock: same text → same vector
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(384).tolist()  # GTE-large dimension
    return _embed


@pytest.fixture
def mock_whitelist():
    """Sample whitelist matching Wave 1 taxonomy."""
    return {
        ("Regulation", "regulatory_clarity"): ["bullish"],
        ("Regulation", "regulatory_restriction"): ["bearish"],
        ("Security", "sell_pressure"): ["bearish"],
        ("Adoption", "demand_increase"): ["bullish"],
        ("Macro", "risk_on"): ["bullish"],
        ("Macro", "risk_off"): ["bearish"],
    }
```

---

### Acceptance Criteria Coverage

Each acceptance criterion from TRD-30 maps to specific tests:

| Criterion | Test |
|-----------|------|
| Valid extraction (mechanism in whitelist, direction valid, evidence found) → `accepted` | `test_validate_extraction_accepted` |
| Unknown mechanism → `rejected_mechanism`, raw LLM output in `rejected_mechanism_raw` | `test_rejected_mechanism_captures_raw` |
| Direction not in `valid_directions` → `rejected_direction` | `test_validate_extraction_rejected_direction` |
| Evidence span not in article body → `rejected_evidence` | `test_validate_extraction_rejected_evidence` |
| `evidence_relevance_score` always populated for impactful articles (even for rejected chains) | `test_rejected_mechanism_still_scores` |
| With `threshold=None`: no rejection based on evidence_relevance_score | No test needed (V3 has no threshold logic) |
| `uv run pytest tests/agents/test_mechanism_validator.py` passes | All tests above |

---

## 8. Performance Considerations

### Latency Budget

- Steps 1-4 (deterministic): <5ms
- Step 5 (Module B): ~200-500ms (2 embedding calls)
- **Total: ~200-500ms per validation**

### Optimization Opportunities (Future)

- **Batch embedding calls**: Process N articles, then call endpoint once with batch of 2N texts (claim + evidence for each)
- **Cache whitelist in memory**: Avoid Spark `collect()` per article (already done by loading once at graph init)
- **Pre-normalize article_body**: Normalize once in gatekeeper, reuse in validator

**Not needed for V3** — current throughput target is ~500 articles/day, latency is non-critical.

---

### Error Boundaries

**What can fail and how it's handled:**

| Failure Mode | Where | Handling |
|--------------|-------|----------|
| Malformed treatment_class | Step 1 | Best-effort parse → fails at step 2 |
| Unknown mechanism | Step 2 | rejected_mechanism + raw captured |
| Invalid direction | Step 3 | rejected_direction |
| Missing evidence | Step 4 | rejected_evidence |
| Embedding timeout | Step 5 | Caught, logged, score=None, validation continues |
| Embedding dimension mismatch | Step 5 | Caught, logged, score=None |
| Zero-length embedding vectors | Step 5 | Returns 0.0 similarity |

**Critical invariant:** Validation ALWAYS returns a `ValidationResult`. No uncaught exceptions propagate to caller.

---

## 9. Dependencies

### Runtime

- `numpy` — cosine similarity computation (already in project deps)
- `typing.TypedDict` — data contracts (stdlib, Python 3.8+)
- `logging` — Module B error handling (stdlib)
- `MechanismValidatorConfig` — from `agents.config` (Wave 1, TRD-24)

### Test

- `pytest` — test framework (already configured)
- `pytest.mark.parametrize` — parameterized tests for edge cases

**No new dependencies required.**

---

## 10. Verification Plan

After implementation, verify correctness via:

1. **Unit tests pass**: `uv run pytest tests/agents/test_mechanism_validator.py -v`
2. **Type checking**: `uv run mypy src/trading_databricks/agents/nodes/mechanism_validator.py`
3. **Manual integration test on cluster**:
   ```python
   # On cluster 0331-043814-ghylcr28
   from trading_databricks.agents.nodes import validate_extraction
   from trading_databricks.agents.taxonomy import load_whitelist
   from trading_databricks.agents.config import MechanismValidatorConfig
   
   whitelist = load_whitelist(spark, catalog="smart-trading-dev", schema="gold")
   
   extraction = {
       "treatment_class": "Regulation.etf_ruling",
       "trigger_event": "SEC approves spot Bitcoin ETF",
       "mechanism": "regulatory_clarity",
       "expected_direction": "bullish",
       "evidence_spans": ["The SEC approved"],
   }
   
   result = validate_extraction(
       extraction,
       article_body="The SEC approved the first spot Bitcoin ETF",
       whitelist=whitelist,
       config=MechanismValidatorConfig(),
       embed_fn=lambda text: [0.0] * 384,  # Mock for quick test
   )
   
   print(result)
   # Expected: {"validation_status": "accepted", "rejected_mechanism_raw": None, "evidence_relevance_score": <float>}
   ```

4. **Integration with LangGraph** (TRD-32, Wave 4): Validator node wired into graph, processes eval dataset rows

---

## 11. Future Enhancements (Out of Scope for V3)

### V4: Add Rejection Threshold for Module B

```python
# In MechanismValidatorConfig:
evidence_grounding_reject_threshold: float | None = 0.3  # Enable rejection

# In validate_extraction, after computing score:
if (
    config.evidence_grounding_reject_threshold is not None
    and score is not None
    and score < config.evidence_grounding_reject_threshold
):
    status = "rejected_low_grounding"
```

**Requires:** Collect V3 data to determine optimal threshold via ROC analysis.

---

### V5: Batch Embedding for Throughput

```python
# Process N articles in parallel
claims = [f"{e['trigger_event']} causes {e['mechanism']}" for e in extractions]
evidences = [" ".join(e["evidence_spans"]) for e in extractions]

# Single batch call (N claims + N evidences = 2N texts)
all_embeddings = embed_fn_batch(claims + evidences)

claim_embeddings = all_embeddings[:N]
evidence_embeddings = all_embeddings[N:]
```

**Requires:** Databricks Model Serving batch API support.

---

## 12. Summary

This design implements a modular, testable mechanism validator with:

- **5-step validation pipeline**: L1 parsing → whitelist lookup → direction check → evidence substring match → Module B grounding score
- **No early returns**: Steps 1-4 determine status, step 5 always runs to ensure `evidence_relevance_score` is populated even for rejected chains
- **Graceful error handling**: Module B embedding failures are caught, logged, and return `None` — validation continues
- **TypedDict contracts**: Static type checking for LangGraph integration, zero runtime overhead
- **Comprehensive testing**: Unit tests for helpers, integration tests for end-to-end validation, parameterized tests for edge cases
- **Wave 1 alignment**: Uses `load_whitelist()` from TRD-23, `MechanismValidatorConfig` from TRD-24

**Next step:** Write implementation plan using `superpowers:writing-plans` skill.
