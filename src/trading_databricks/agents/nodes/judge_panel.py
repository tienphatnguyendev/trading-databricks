"""Judge panel node — fan-out to 3 LLM judges, strict-AND aggregation.

The same judge prompts (judge_direction.yaml, judge_mechanism.yaml,
judge_grounding.yaml) used in the offline eval scorers are reused here for the
online reflection signal. Direction and mechanism judges return categorical
verdicts (parsed via simple equality / "match" check). Grounding judge returns
a float in [0, 1] thresholded at config.judge_panel.grounding_threshold.

The three judges are invoked sequentially in the implementation below for
simplicity; LangGraph's Send API would parallelise them across separate node
instances. Sequential is acceptable for v1 because each call has its own
timeout and the panel's wall-clock cost is dominated by the slowest judge
either way (the LLM endpoint multiplexes requests). Parallelisation can be
added in a follow-up if profiling shows it matters.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.nodes._client import _get_client
from trading_databricks.agents.prompts import load_prompt, render_messages
from trading_databricks.agents.state import MASState

logger = logging.getLogger(__name__)


JUDGE_NAMES = ("judge_direction", "judge_mechanism", "judge_grounding")
JUDGE_KEYS = ("direction", "mechanism", "grounding")


def _invoke_judge(
    judge_name: str,
    *,
    headline: str,
    body: str,
    extraction: dict,
    endpoint: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Invoke a single judge LLM. Returns {passed, score, reason}.

    Real implementation calls the LLM endpoint with the rendered judge prompt.
    Tests monkeypatch this function.
    """
    messages = load_prompt(judge_name, version="production")
    rendered = render_messages(
        messages,
        {
            "headline": headline,
            "body": body,
            "extraction": json.dumps(extraction),
        },
    )
    client = _get_client()
    response = client.predict(
        endpoint=endpoint,
        inputs={"messages": rendered, "temperature": 0.0, "max_tokens": 100},
    )
    text = (response.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    return _interpret_judge_response(judge_name, text)


def _interpret_judge_response(judge_name: str, text: str) -> dict[str, Any]:
    """Convert raw judge output to {passed, score, reason}.

    direction → expects "up" | "down" | "neutral"; passes if matches extraction direction.
    mechanism → expects "match" | "mismatch"; passes on "match".
    grounding → expects float in [0, 1]; passes if >= threshold (caller applies threshold).
    """
    text = text.strip().lower()
    if judge_name == "judge_direction":
        # Caller compares against extraction.expected_direction; we pass through the verdict.
        return {"passed": text in ("up", "down", "neutral"), "score": text, "reason": "" if text in ("up", "down", "neutral") else "unparseable verdict"}
    if judge_name == "judge_mechanism":
        passed = text.startswith("match")
        return {"passed": passed, "score": text, "reason": "" if passed else text}
    if judge_name == "judge_grounding":
        try:
            score = float(text.split()[0])
        except (ValueError, IndexError):
            return {"passed": False, "score": None, "reason": f"unparseable score: {text[:50]}"}
        return {"passed": None, "score": score, "reason": ""}
    return {"passed": False, "score": None, "reason": f"unknown judge: {judge_name}"}


def _direction_pass(extraction: dict, verdict: dict) -> tuple[bool, str]:
    """Direction judge passes only if its verdict matches the extraction's expected_direction.

    extraction.expected_direction in {"bullish", "bearish", "volatile"}
    verdict.score in {"up", "down", "neutral"}
    Mapping: bullish→up, bearish→down, volatile→neutral.
    """
    mapping = {"bullish": "up", "bearish": "down", "volatile": "neutral"}
    expected = mapping.get(extraction.get("expected_direction", ""), "")
    actual = verdict.get("score", "")
    if expected and actual == expected:
        return True, ""
    return False, f"direction mismatch: extraction={extraction.get('expected_direction')} judge={actual}"


def judge_panel(state: MASState, config: MASConfig) -> dict[str, Any]:
    """Run all three judges and aggregate via strict-AND.

    Returns:
        judge_results: dict[str, dict] keyed by ("direction", "mechanism", "grounding")
        judge_failures: dict[str, str] | None — failure reasons keyed by judge name
        extraction_status (only on all-failed): "unjudged"
    """
    extraction = state.get("extraction") or {}
    headline = state.get("headline", "")
    body = state.get("body", "")

    results: dict[str, dict] = {}
    failures: dict[str, str] = {}
    unavailable_count = 0

    for judge_name, key in zip(JUDGE_NAMES, JUDGE_KEYS):
        try:
            verdict = _invoke_judge(
                judge_name,
                headline=headline,
                body=body,
                extraction=extraction,
                endpoint=config.judge_panel.judge_endpoint,
                timeout_s=config.judge_panel.timeout_s,
            )
        except Exception as exc:
            logger.warning("%s unavailable: %s", judge_name, exc)
            verdict = {"passed": False, "score": None, "reason": "judge_unavailable"}
            unavailable_count += 1

        # Apply per-judge final pass logic
        if key == "direction" and verdict.get("score") is not None and verdict.get("reason") != "judge_unavailable":
            passed, reason = _direction_pass(extraction, verdict)
            verdict = {"passed": passed, "score": verdict["score"], "reason": reason}
        elif key == "grounding" and isinstance(verdict.get("score"), (int, float)):
            passed = verdict["score"] >= config.judge_panel.grounding_threshold
            verdict = {"passed": passed, "score": verdict["score"], "reason": "" if passed else f"score {verdict['score']:.2f} < threshold {config.judge_panel.grounding_threshold}"}

        results[key] = verdict
        if not verdict["passed"]:
            failures[key] = verdict.get("reason") or "failed"

    if unavailable_count == len(JUDGE_NAMES):
        return {"judge_results": {}, "judge_failures": None, "extraction_status": "unjudged"}

    return {
        "judge_results": results,
        "judge_failures": failures or None,
    }
