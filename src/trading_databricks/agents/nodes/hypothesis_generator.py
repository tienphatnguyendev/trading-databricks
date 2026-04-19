"""Hypothesis generator node — replaces the monolithic causality_extractor LLM call.

Consumes evidence (from evidence_extractor) and precedents (from rag_retriever),
produces a single causal-chain extraction. On reflection retries, injects the
previous judge_failures into the user message.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import yaml

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.nodes._client import _get_client
from trading_databricks.agents.prompts import _PACKAGE_DIR, load_prompt, render_messages
from trading_databricks.agents.prompts._formatters import (
    format_few_shot,
    format_full_taxonomy,
    format_whitelist,
)
from trading_databricks.agents.state import MASState

logger = logging.getLogger(__name__)


def _parse_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _format_evidence_block(evidence: list[dict]) -> str:
    if not evidence:
        return "(no structured evidence extracted)"
    lines = []
    for i, ev in enumerate(evidence, 1):
        actor = ev.get("actor") or "?"
        action = ev.get("action") or "?"
        obj = ev.get("object") or ""
        span = ev.get("span") or ""
        lines.append(f"{i}. actor={actor} action={action} object={obj} span=\"{span}\"")
    return "\n".join(lines)


def _format_precedent_block(precedents: list[dict]) -> str:
    if not precedents:
        return "(no precedents found)"
    lines = ["headline | timestamp | return_24h | direction"]
    for p in precedents:
        lines.append(
            f"{p.get('headline', '?')} | {p.get('timestamp', '?')} | "
            f"{p.get('return_24h', '?')} | {p.get('direction', '?')}"
        )
    return "\n".join(lines)


def _format_failure_context(judge_failures: dict | None) -> str:
    if not judge_failures:
        return ""
    lines = ["", "PRIOR JUDGE FAILURES (address each in your new attempt):"]
    for judge_name, reason in judge_failures.items():
        lines.append(f"- {judge_name}: {reason}")
    return "\n".join(lines)


def _load_metadata() -> dict:
    """Load YAML metadata (reflection footer + parse retry suffix templates)."""
    doc = yaml.safe_load((_PACKAGE_DIR / "hypothesis_generator.yaml").read_text())
    return doc.get("metadata", {})


def hypothesis_generator(
    state: MASState,
    config: MASConfig,
    *,
    taxonomy: dict,
    whitelist: dict,
) -> dict[str, Any]:
    """Generate a causal-chain extraction grounded in evidence + precedents.

    taxonomy and whitelist are pre-loaded at graph construction time (see graph.py
    build_mas_graph) — not fetched on each invocation.

    Returns a dict with keys:
        extraction: dict | None
        extraction_status: str  "ok" | "ok_after_reflection" | "parse_failed" | "failed"
    """
    metadata = _load_metadata()
    judge_failures = state.get("judge_failures")
    reflection_count = state.get("reflection_count", 0)
    is_reflection = reflection_count > 0 and judge_failures

    messages = load_prompt("hypothesis_generator", version="production")
    rendered = render_messages(
        messages,
        {
            "taxonomy_injection": format_full_taxonomy(taxonomy),
            "whitelist_injection": format_whitelist(whitelist),
            "few_shot_injection": format_few_shot(state.get("few_shot_examples", []), 5),
            "reflection_footer": metadata.get("reflection_footer_template", "") if is_reflection else "",
            "headline": state.get("headline", ""),
            "body": state.get("body", ""),
            "evidence_block": _format_evidence_block(state.get("evidence", [])),
            "precedent_block": _format_precedent_block(state.get("precedents", [])),
            "failure_context": _format_failure_context(judge_failures),
        },
    )

    try:
        client = _get_client()
        response = client.predict(
            endpoint=config.llm_endpoint,
            inputs={"messages": rendered, "temperature": 0.0, "max_tokens": 1000},
        )
    except Exception as exc:
        logger.warning("hypothesis_generator LLM call failed: %s", exc)
        return {"extraction": None, "extraction_status": "failed"}

    text = (response.get("choices") or [{}])[0].get("message", {}).get("content", "")
    parsed = _parse_json(text)

    if parsed is None:
        # Retry once with stricter suffix
        retry_suffix = metadata.get("parse_retry_suffix", "")
        retry_messages = list(rendered)
        retry_messages.append({"role": "user", "content": retry_suffix})
        try:
            response = client.predict(
                endpoint=config.llm_endpoint,
                inputs={"messages": retry_messages, "temperature": 0.0, "max_tokens": 1000},
            )
        except Exception as exc:
            logger.warning("hypothesis_generator retry LLM call failed: %s", exc)
            return {"extraction": None, "extraction_status": "failed"}
        text = (response.get("choices") or [{}])[0].get("message", {}).get("content", "")
        parsed = _parse_json(text)

    if parsed is None:
        return {"extraction": None, "extraction_status": "parse_failed"}

    status = "ok_after_reflection" if is_reflection else "ok"
    return {"extraction": parsed, "extraction_status": status}
