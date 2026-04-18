"""Evidence extractor node — first step of the causality sub-pipeline.

Extracts structured (actor, action, object, span, timestamp_hint) tuples from
the article using a single LLM call. Output flows into rag_retriever and
hypothesis_generator.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import mlflow.deployments

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.prompts import load_prompt, render_messages
from trading_databricks.agents.state import MASState

logger = logging.getLogger(__name__)


def _parse_evidence(text: str) -> list[dict] | None:
    """Parse a JSON array from LLM response, stripping markdown fences."""
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
    if not isinstance(parsed, list):
        return None
    return [item for item in parsed if isinstance(item, dict)]


def _get_client():
    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE", os.environ.get("DATABRICKS_PROFILE", ""))
    target_uri = f"databricks://{profile}" if profile else "databricks"
    return mlflow.deployments.get_deploy_client(target_uri)


def evidence_extractor(state: MASState, config: MASConfig) -> dict[str, Any]:
    """Extract structured evidence tuples from the article.

    Returns a dict with keys:
        evidence: list[dict]   (empty list on failure or malformed JSON)
        evidence_status: str   "ok" | "empty" | "failed"
        extraction_status: str (only set to "failed" on LLM exception, propagates termination)
    """
    headline = state.get("headline", "")
    body = state.get("body", "")

    messages = load_prompt("evidence_extractor", version="production")
    rendered = render_messages(messages, {"headline": headline, "body": body})

    try:
        client = _get_client()
        response = client.predict(
            endpoint=config.llm_endpoint,
            inputs={"messages": rendered, "temperature": 0.0, "max_tokens": 800},
        )
    except Exception as exc:
        logger.warning("evidence_extractor LLM call failed: %s", exc)
        return {"evidence": [], "evidence_status": "failed", "extraction_status": "failed"}

    text = (response.get("choices") or [{}])[0].get("message", {}).get("content", "")
    parsed = _parse_evidence(text)
    if parsed is None or len(parsed) == 0:
        return {"evidence": [], "evidence_status": "empty"}
    return {"evidence": parsed, "evidence_status": "ok"}
