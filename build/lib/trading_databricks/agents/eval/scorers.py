"""Scorer catalog for the MAS evaluation harness.

Three tiers:
    1. Deterministic reference-free (schema, taxonomy, whitelist, verbatim, validator, latency)
    2. LLM judges (direction, mechanism, grounding) — registered at import
    3. Reference-based (direction_match, mechanism_match, grounding_alignment)

Reference-based scorers are auto-skipped by runner.py when the dataset has
empty expectations.
"""
from __future__ import annotations

import os
from typing import Any

from mlflow.genai.scorers import scorer

from trading_databricks.agents.schemas import CausalChain


@scorer
def schema_valid(*, outputs: dict[str, Any]) -> float:
    extraction = outputs.get("extraction")
    if not extraction:
        return 0.0
    try:
        CausalChain.model_validate(extraction)
        return 1.0
    except Exception:
        return 0.0


@scorer
def treatment_in_taxonomy(
    *,
    outputs: dict[str, Any],
    taxonomy: dict[str, list[str]],
) -> float:
    extraction = outputs.get("extraction") or {}
    treatment = extraction.get("treatment_class", "")
    if "." not in treatment:
        return 0.0
    l1, l2 = treatment.split(".", 1)
    return 1.0 if l2 in taxonomy.get(l1, []) else 0.0


@scorer
def mechanism_in_whitelist(
    *,
    outputs: dict[str, Any],
    whitelist: dict[tuple[str, str], list[str]],
) -> float:
    extraction = outputs.get("extraction") or {}
    treatment = extraction.get("treatment_class", "")
    mechanism = extraction.get("mechanism", "")
    if "." not in treatment:
        return 0.0
    l1 = treatment.split(".", 1)[0]
    return 1.0 if (l1, mechanism) in whitelist else 0.0


@scorer
def evidence_spans_verbatim(
    *,
    outputs: dict[str, Any],
    inputs: dict[str, Any],
) -> float:
    extraction = outputs.get("extraction") or {}
    body = inputs.get("body", "") or ""
    spans = extraction.get("evidence_spans", []) or []
    if not spans:
        return 0.0
    return 1.0 if all(s in body for s in spans) else 0.0


@scorer
def validator_accepted(*, outputs: dict[str, Any]) -> float:
    return 1.0 if outputs.get("validation_status") == "accepted" else 0.0


DETERMINISTIC_SCORERS = [
    schema_valid,
    treatment_in_taxonomy,
    mechanism_in_whitelist,
    evidence_spans_verbatim,
    validator_accepted,
]

def _register_if_absent(scorer_obj, *, experiment_id: str | None):
    """Register scorer with experiment unless a same-name scorer already exists."""
    if experiment_id is None:
        return scorer_obj
    try:
        import mlflow.genai.scorers as _gs

        existing = {s.name for s in _gs.get_all_scorers(experiment_id=experiment_id)}
    except Exception:
        existing = set()
    if scorer_obj.name not in existing:
        try:
            scorer_obj.register(experiment_id=experiment_id)
        except Exception:
            pass
    return scorer_obj


def _make_judges(experiment_id: str | None):
    from mlflow.genai.judges import make_judge

    from trading_databricks.agents.prompts import load_prompt

    def _instructions(name: str) -> str:
        messages = load_prompt(name, version="production")
        return messages[0]["content"]

    model = os.environ.get(
        "MAS_JUDGE_MODEL",
        "databricks:/gemini-3-1-pro",
    )

    direction = make_judge(
        name="judge_direction",
        instructions=_instructions("judge_direction"),
        model=model,
        feedback_value_type="categorical",
    )
    mechanism = make_judge(
        name="judge_mechanism",
        instructions=_instructions("judge_mechanism"),
        model=model,
        feedback_value_type="categorical",
    )
    grounding = make_judge(
        name="judge_grounding",
        instructions=_instructions("judge_grounding"),
        model=model,
        feedback_value_type="float",
    )

    exp_id = experiment_id or os.environ.get("MLFLOW_EXPERIMENT_ID")
    return [
        _register_if_absent(direction, experiment_id=exp_id),
        _register_if_absent(mechanism, experiment_id=exp_id),
        _register_if_absent(grounding, experiment_id=exp_id),
    ]

@scorer
def direction_match(
    *,
    outputs: dict[str, Any],
    expectations: dict[str, Any],
) -> float | None:
    if "direction_label" not in expectations:
        return None
    extraction = outputs.get("extraction") or {}
    return 1.0 if extraction.get("direction") == expectations["direction_label"] else 0.0


@scorer
def mechanism_match(
    *,
    outputs: dict[str, Any],
    expectations: dict[str, Any],
) -> float | None:
    if "mechanism_label" not in expectations:
        return None
    extraction = outputs.get("extraction") or {}
    return 1.0 if extraction.get("mechanism") == expectations["mechanism_label"] else 0.0


@scorer
def grounding_alignment(
    *,
    outputs: dict[str, Any],
    expectations: dict[str, Any],
) -> float | None:
    if "causal_span_present" not in expectations:
        return None
    has_spans = bool((outputs.get("extraction") or {}).get("evidence_spans"))
    return 1.0 if has_spans == expectations["causal_span_present"] else 0.0


REFERENCE_SCORERS = [direction_match, mechanism_match, grounding_alignment]
JUDGE_SCORERS: list = []
ALL_SCORERS = DETERMINISTIC_SCORERS + JUDGE_SCORERS + REFERENCE_SCORERS

def ensure_judges_registered(experiment_id: str | None = None) -> list:
    """Lazy builder so imports don't touch MLflow unless asked."""
    global JUDGE_SCORERS, ALL_SCORERS
    if not JUDGE_SCORERS:
        JUDGE_SCORERS = _make_judges(experiment_id)
        ALL_SCORERS = DETERMINISTIC_SCORERS + JUDGE_SCORERS + REFERENCE_SCORERS
    return JUDGE_SCORERS
