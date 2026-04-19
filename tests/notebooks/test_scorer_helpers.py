"""Unit tests for _scorer_helpers — pure Python, no Spark/MLflow/network."""
from __future__ import annotations

from notebooks.qa.mas._scorer_helpers import (
    check_evidence_quality,
    check_extraction_ok,
    check_judge_all_pass,
    check_reflection_triggered,
    # existing (regression guard)
    check_schema_conformance,
    evidence_grounding_ratio,
)


# --- check_evidence_quality ---

def test_evidence_quality_ok():
    assert check_evidence_quality("ok") is True


def test_evidence_quality_empty():
    assert check_evidence_quality("empty") is False


def test_evidence_quality_failed():
    assert check_evidence_quality("failed") is False


def test_evidence_quality_none():
    assert check_evidence_quality(None) is False


# --- check_judge_all_pass ---

def test_judge_all_pass_none_means_pass():
    assert check_judge_all_pass(None) is True


def test_judge_all_pass_empty_dict_means_fail():
    # an empty dict still means "judge_failures was populated" — treat as fail
    assert check_judge_all_pass({}) is False


def test_judge_all_pass_with_failures():
    assert check_judge_all_pass({"direction": "wrong direction"}) is False


# --- check_reflection_triggered ---

def test_reflection_triggered_zero():
    assert check_reflection_triggered(0) is False


def test_reflection_triggered_none():
    assert check_reflection_triggered(None) is False


def test_reflection_triggered_one():
    assert check_reflection_triggered(1) is True


def test_reflection_triggered_three():
    assert check_reflection_triggered(3) is True


# --- check_extraction_ok ---

def test_extraction_ok_ok():
    assert check_extraction_ok("ok") is True


def test_extraction_ok_after_reflection():
    assert check_extraction_ok("ok_after_reflection") is True


def test_extraction_ok_judge_failed_accepted():
    assert check_extraction_ok("judge_failed_accepted") is False


def test_extraction_ok_unjudged():
    assert check_extraction_ok("unjudged") is False


def test_extraction_ok_error():
    assert check_extraction_ok("error") is False


def test_extraction_ok_none():
    assert check_extraction_ok(None) is False


# --- regression: existing predicates still work ---

def test_schema_conformance_full():
    extraction = {
        "treatment_class": "Macro.rate_decision",
        "mechanism": "liquidity_contraction",
        "expected_direction": "bearish",
        "confidence": 0.8,
        "trigger_event": "Fed raises rates",
        "evidence_spans": ["rates by 75bps"],
    }
    assert check_schema_conformance(extraction) == 1.0


def test_evidence_grounding_ratio_hit():
    assert evidence_grounding_ratio(["rates by 75bps"], "Fed raises rates by 75bps.") == 1.0


def test_evidence_grounding_ratio_miss():
    assert evidence_grounding_ratio(["no match here"], "completely different body") == 0.0
