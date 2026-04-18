"""Reflector node — pure routing logic for the bounded reflection loop.

Reads judge_results + reflection_count, decides whether to retry
hypothesis_generator or proceed to mechanism_validator. Writes a
`_route` key in the returned dict that the graph's conditional edge
function reads.
"""
from __future__ import annotations

import logging
from typing import Any

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.state import MASState

logger = logging.getLogger(__name__)

REFLECTOR_ROUTE_VALIDATOR = "mechanism_validator"
REFLECTOR_ROUTE_RETRY = "hypothesis_generator"


def _judges_passed(judge_results: dict | None) -> bool:
    """All judges passed iff the dict is non-empty and every entry has passed=True."""
    if not judge_results:
        return False
    return all(r.get("passed") for r in judge_results.values())


def reflector(state: MASState, config: MASConfig) -> dict[str, Any]:
    """Decide whether to reflect (retry) or move on.

    Returns a dict with:
        _route: str — REFLECTOR_ROUTE_VALIDATOR or REFLECTOR_ROUTE_RETRY
        reflection_count: int (incremented when retrying)
        extraction_status: str (set to "judge_failed_accepted" when budget exhausted with failures)

    Note: graph.py wires `_route` through `add_conditional_edges`.
    """
    judge_results = state.get("judge_results")
    judge_failures = state.get("judge_failures")
    reflection_count = state.get("reflection_count", 0)
    current_status = state.get("extraction_status", "ok")

    # Unjudged path: judges all unavailable, can't reflect — proceed to validator
    if current_status == "unjudged":
        return {"_route": REFLECTOR_ROUTE_VALIDATOR}

    # Happy path: all judges passed
    if _judges_passed(judge_results):
        return {"_route": REFLECTOR_ROUTE_VALIDATOR, "reflection_count": reflection_count}

    # Failed but within budget: retry
    if reflection_count < config.reflection.max_reflections:
        return {
            "_route": REFLECTOR_ROUTE_RETRY,
            "reflection_count": reflection_count + 1,
        }

    # Failed and budget exhausted: accept anyway, flag the status
    logger.info(
        "reflection budget exhausted; accepting flagged extraction. failures=%s",
        judge_failures,
    )
    return {
        "_route": REFLECTOR_ROUTE_VALIDATOR,
        "extraction_status": "judge_failed_accepted",
    }
