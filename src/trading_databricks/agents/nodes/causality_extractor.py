"""Causality extractor node for MAS."""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

import mlflow.deployments

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.nodes.few_shot_retriever import MASState, get_spark_session
from trading_databricks.agents.prompts import build_system_prompt
from trading_databricks.agents.taxonomy import load_taxonomy, load_whitelist

logger = logging.getLogger(__name__)


def parse_json_response(text: str) -> dict[str, Any] | None:
    """Parse JSON from LLM response, stripping markdown wrappers if present."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON from LLM: {text[:100]}...")
        return None


def majority_vote(extractions: list[dict]) -> dict:
    """Select the majority consensus extraction, or highest confidence if tie.

    Consensus key: (treatment_class, mechanism, expected_direction).
    If no consensus, penalizes confidence by 0.3.
    """
    valid = [e for e in extractions if e]
    if not valid:
        return {}

    # Build consensus keys
    keys = []
    for ex in valid:
        k = (ex.get("treatment_class"), ex.get("mechanism"), ex.get("expected_direction"))
        keys.append(k)

    counts = Counter(keys)
    most_common_key, count = counts.most_common(1)[0]

    if count >= 2:
        # Return first matching extraction
        for ex, k in zip(valid, keys):
            if k == most_common_key:
                return ex

    # Tie-breaker: highest confidence
    best_ex = max(valid, key=lambda x: float(x.get("confidence", 0.0)))
    penalized_conf = max(0.0, float(best_ex.get("confidence", 0.0)) - 0.3)
    best_ex["confidence"] = penalized_conf
    return best_ex


def causality_extractor(state: MASState, config: MASConfig) -> MASState:
    """Extract causal chain using LLM with path-based routing."""
    path = state.get("extraction_path", "standard")

    spark = get_spark_session()
    taxonomy = load_taxonomy(spark, catalog=config.catalog, schema=config.schema)
    whitelist = load_whitelist(spark, catalog=config.catalog, schema=config.schema)

    system_prompt = build_system_prompt(
        taxonomy, whitelist, state.get("few_shot_examples", []), path
    )

    user_msg = f"Headline: {state['headline']}\nBody excerpt: {state['body'][: config.gatekeeper.text_window]}"
    if path in ("standard", "careful"):
        user_msg = f"Headline: {state['headline']}\nBody: {state['body']}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    temperature = 0.7 if path == "careful" else 0.0
    n_choices = 3 if path == "careful" else 1

    client = mlflow.deployments.get_deploy_client("databricks")

    logger.info(
        f"Calling LLM {config.llm_endpoint} for path={path}, n={n_choices}, temp={temperature}"
    )
    response = client.predict(
        endpoint=config.llm_endpoint,
        inputs={
            "messages": messages,
            "temperature": temperature,
            "n": n_choices,
            "max_tokens": 1000,
        },
    )

    extractions = []
    for choice in response.get("choices", []):
        text = choice.get("message", {}).get("content", "")
        parsed = parse_json_response(text)
        if parsed:
            extractions.append(parsed)

    if not extractions:
        logger.error("LLM failed to produce valid JSON across all choices.")
        state["extraction"] = None
        return state

    if path == "careful":
        final_ext = majority_vote(extractions)
    else:
        final_ext = extractions[0]

    state["extraction"] = final_ext
    return state
