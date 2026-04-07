"""Causality extractor node for MAS."""
from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

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
        k = (
            ex.get("treatment_class"),
            ex.get("mechanism"),
            ex.get("expected_direction")
        )
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
