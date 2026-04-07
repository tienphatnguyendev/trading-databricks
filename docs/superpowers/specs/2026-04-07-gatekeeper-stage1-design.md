# Gatekeeper Stage 1 — spaCy EntityRuler Design

**Date:** 2026-04-07
**Version:** 1.0
**Status:** Approved
**Ticket:** TRD-25
**Parent:** TRD-6 (Build Gen-AI Agent Layer)

**Extends:**
- `2026-04-04-mas-final-spec.md §5.1` — CRYPTO_ENTITY_PATTERNS (50 patterns)
- `2026-04-04-mas-final-spec.md §5.2` — build_nlp(), extract_ner_features()
- `2026-04-04-mas-final-spec.md §3.1` — gatekeeper position in pipeline

---

## 1. Context

TRD-25 is Wave 2 of the MAS pipeline, the first code ticket after Wave 1 (TRD-23/24) merged.
Implements the Stage 1 gatekeeper: a rule-based NER filter that runs on 100% of articles at
near-zero cost. Short-circuit: if no crypto entity is found, is_impactful=False immediately
without calling Stage 2 (TRD-26) or any LLM.

Wave 1 exports consumed: `GatekeeperConfig` from `agents.config`.

---

## 2. Module Structure

**File:** `src/trading_databricks/agents/gatekeeper.py`

Four public items:

```
CRYPTO_ENTITY_PATTERNS   # list[dict] — 50 patterns verbatim from spec §5.1
build_nlp()              # @functools.cache → spacy.Language
extract_ner_features()   # spacy.tokens.Doc → dict[str, bool | int]
is_crypto_relevant()     # (headline, body, *, config) → tuple[bool, float]
```

No class needed. `extract_ner_features` accepts a Doc so it can be tested independently
of text processing. `is_crypto_relevant` is the only entry point that touches GatekeeperConfig.

---

## 3. Key Design Decision: @functools.cache on build_nlp()

`build_nlp()` is decorated with `@functools.cache`. The spacy.Language instance is built
once per Python process (cache key = empty tuple — function takes no args).

- Safe: cache is per-process, never serialized. UDF callers (TRD-26) call build_nlp()
  inside the UDF body per §5.4 — the per-executor-process cache is free on reuse.
- Fast: tests and repeated calls in interactive use don't rebuild (~10ms each).
- No import-time side effect: unlike a module-level `_NLP = build_nlp()`, this is lazy.

---

## 4. Text Construction in is_crypto_relevant

```python
text = headline + " " + body[:config.text_window]
```

Headline first (always fully included), body truncated to text_window (default: 500 chars).

---

## 5. Pattern Count

| Category | Count |
|----------|-------|
| CRYPTO_ASSET | 11 |
| CRYPTO_EXCHANGE | 8 |
| CRYPTO_STABLECOIN | 5 |
| CRYPTO_PROTOCOL | 10 |
| CRYPTO_CONCEPT | 16 |
| **Total** | **50** |

Ambiguous standalone tokens (compound, lido, OP, mining) are compound patterns to avoid
false positives (e.g., `compound finance` not `compound`).

---

## 6. Acceptance Criteria

| Input | Expected |
|-------|---------|
| "Bitcoin ETF approved" | (True, 1.0) — CRYPTO_ASSET: bitcoin |
| "UK inflation data surprises upward" | (False, 0.0) — no crypto entity |
| "Fed rate decision affects DeFi yields" | (True, 1.0) — CRYPTO_CONCEPT: defi |

---

## 7. Exports (Public API for TRD-26)

| Name | Used by |
|------|---------|
| `CRYPTO_ENTITY_PATTERNS` | TRD-26 (UDF builds same ruler) |
| `build_nlp` | TRD-26 (called inside pandas UDF body) |
| `extract_ner_features` | TRD-26 (NER features fed to sklearn) |
| `is_crypto_relevant` | TRD-26, TRD-32 (pipeline integration) |

---

## 8. Verification

```bash
uv run pytest tests/agents/test_gatekeeper.py -v
uv run pytest tests/agents/test_gatekeeper.py::test_ner_patterns -v
uv run pytest tests/agents/ -v
uv run ruff check src/trading_databricks/agents/gatekeeper.py
uv run ruff check tests/agents/test_gatekeeper.py
```
