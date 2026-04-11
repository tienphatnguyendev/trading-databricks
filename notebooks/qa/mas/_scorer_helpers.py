"""Pure-Python scorer predicates for the MAS E2E deep-dive research notebook.

Extracted from the notebook so they can be unit-tested without importing
MLflow, Databricks Connect, or any heavy runtime dependency. All functions
accept only stdlib-compatible input types.
"""
from __future__ import annotations

from typing import Any

_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {"treatment_class", "mechanism", "expected_direction", "confidence", "trigger_event", "evidence_spans"}
)
_VALID_DIRECTIONS: frozenset[str] = frozenset({"bullish", "bearish", "volatile", "neutral"})


def check_schema_conformance(extraction: dict[str, Any] | None) -> float:
    """Return a per-field pass rate in [0.0, 1.0] for the extraction.

    All 6 required fields must be present (0.5 weight) and the two
    constrained fields — ``expected_direction`` and ``confidence`` —
    must satisfy value constraints (0.5 weight total). Returns 0.0 for
    None input.
    """
    if extraction is None:
        return 0.0

    total = len(_REQUIRED_FIELDS)
    present = sum(1 for f in _REQUIRED_FIELDS if extraction.get(f) is not None)
    field_ratio = present / total

    pass_constraints = False
    if (
        extraction.get("expected_direction") in _VALID_DIRECTIONS
        and isinstance(extraction.get("confidence"), int | float)
        and 0.0 <= float(extraction["confidence"]) <= 1.0
    ):
        pass_constraints = True

    return 0.5 * field_ratio + (0.5 if pass_constraints else 0.0)


def check_taxonomy_whitelist_compliance(
    extraction: dict[str, Any] | None,
    whitelist: dict[tuple[str, str], list[str]],
) -> bool:
    """Return True iff the (l1, mechanism, direction) tuple is whitelisted.

    ``whitelist`` maps ``(l1_category, mechanism)`` → ``[valid_directions]``.
    Returns False for None extraction or unknown (l1, mechanism) pair.
    """
    if extraction is None:
        return False

    treatment_class: str | None = extraction.get("treatment_class")
    mechanism: str | None = extraction.get("mechanism")
    direction: str | None = extraction.get("expected_direction")

    if not treatment_class or not mechanism or not direction:
        return False

    l1 = treatment_class.split(".")[0] if "." in treatment_class else treatment_class
    key = (l1, mechanism)
    if key not in whitelist:
        return False

    return direction in whitelist[key]


def check_routing_legality(
    extraction_path: str,
    *,
    cold_start_mode: bool,
    top_similarity: float,
    n_examples: int,
) -> bool:
    """Return True iff the routing decision was legal given index state.

    Rules (mirrors UncertaintyRouterConfig defaults):
    - cold_start: only "careful" is legal
    - warm, sim >= 0.85 and n_examples >= 2: "fast" legal
    - warm, sim < 0.50 or n_examples < 2: "careful" legal
    - warm, 0.50 <= sim < 0.85: "standard" legal
    - any path from the category above is always legal
    """
    if cold_start_mode:
        return extraction_path == "careful"

    HIGH = 0.85
    LOW = 0.50

    if extraction_path == "fast":
        return top_similarity >= HIGH and n_examples >= 2
    if extraction_path == "careful":
        return top_similarity < LOW or n_examples < 2
    if extraction_path == "standard":
        return LOW <= top_similarity < HIGH
    return False


def evidence_grounding_ratio(
    evidence_spans: list[str],
    body: str,
) -> float:
    """Return fraction of evidence_spans found (case-folded) in body.

    Returns 0.0 for empty spans list. Substring match is used; span must
    appear verbatim (case-insensitive) in the article body.
    """
    if not evidence_spans:
        return 0.0
    body_lower = body.casefold()
    grounded = sum(
        1 for span in evidence_spans if span.casefold() in body_lower
    )
    return grounded / len(evidence_spans)
