"""Tests for reflector node — pure routing logic."""
from __future__ import annotations

import pytest

from trading_databricks.agents.config import MASConfig


def test_reflector_passed_routes_to_validator():
    from trading_databricks.agents.nodes.reflector import reflector, REFLECTOR_ROUTE_VALIDATOR

    state = {
        "judge_results": {"direction": {"passed": True}, "mechanism": {"passed": True}, "grounding": {"passed": True}},
        "judge_failures": None,
        "reflection_count": 0,
        "extraction_status": "ok",
    }
    out = reflector(state, config=MASConfig())
    assert out["_route"] == REFLECTOR_ROUTE_VALIDATOR
    assert out["reflection_count"] == 0
    assert "extraction_status" not in out  # don't overwrite ok


def test_reflector_failed_within_budget_routes_back():
    from trading_databricks.agents.nodes.reflector import reflector, REFLECTOR_ROUTE_RETRY

    state = {
        "judge_results": {"direction": {"passed": False}},
        "judge_failures": {"direction": "mismatch"},
        "reflection_count": 0,
        "extraction_status": "ok",
    }
    out = reflector(state, config=MASConfig())
    assert out["_route"] == REFLECTOR_ROUTE_RETRY
    assert out["reflection_count"] == 1


def test_reflector_failed_budget_exhausted_routes_to_validator():
    from trading_databricks.agents.nodes.reflector import reflector, REFLECTOR_ROUTE_VALIDATOR

    state = {
        "judge_results": {"direction": {"passed": False}},
        "judge_failures": {"direction": "mismatch"},
        "reflection_count": 1,  # already at max_reflections=1
        "extraction_status": "ok",
    }
    out = reflector(state, config=MASConfig())
    assert out["_route"] == REFLECTOR_ROUTE_VALIDATOR
    assert out["extraction_status"] == "judge_failed_accepted"


def test_reflector_unjudged_routes_to_validator():
    from trading_databricks.agents.nodes.reflector import reflector, REFLECTOR_ROUTE_VALIDATOR

    state = {
        "judge_results": {},
        "judge_failures": None,
        "reflection_count": 0,
        "extraction_status": "unjudged",
    }
    out = reflector(state, config=MASConfig())
    assert out["_route"] == REFLECTOR_ROUTE_VALIDATOR
    # Status preserved
    assert "extraction_status" not in out or out["extraction_status"] == "unjudged"
