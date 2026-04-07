"""Prompt templates and formatting for the MAS causality extractor."""
from __future__ import annotations

def format_compact_taxonomy(taxonomy: dict[str, list[str]]) -> str:
    classes = []
    for l1, l2s in sorted(taxonomy.items()):
        for l2 in sorted(l2s):
            classes.append(f"{l1}.{l2}")
    return ", ".join(classes)

def format_full_taxonomy(taxonomy: dict[str, list[str]]) -> str:
    lines = []
    for l1, l2s in sorted(taxonomy.items()):
        lines.append(f"- {l1}:")
        for l2 in sorted(l2s):
            lines.append(f"  - {l1}.{l2}")
    return "\n".join(lines)

def format_whitelist(whitelist: dict[tuple[str, str], list[str]]) -> str:
    grouped: dict[str, list[str]] = {}
    for (l1, mechanism), _ in whitelist.items():
        grouped.setdefault(l1, []).append(mechanism)

    lines = []
    for l1, mechs in sorted(grouped.items()):
        lines.append(f"L1 Category: {l1}")
        for m in sorted(mechs):
            lines.append(f"  - {m}")
    return "\n".join(lines)

def format_few_shot(examples: list[dict], limit: int) -> str:
    selected = examples[:limit]
    if not selected:
        return "No examples available."
    formatted = []
    for i, ex in enumerate(selected, 1):
        chain_json = ex.get('causal_chain_json', '{}')
        if not isinstance(chain_json, str):
            import json
            chain_json = json.dumps(chain_json)
        formatted.append(f"Example {i}:\nHeadline: {ex.get('trigger_event', '')}\nExtraction:\n{chain_json}")
    return "\n\n".join(formatted)

_BASE_SYSTEM_PROMPT = """You are a financial news causality extractor for BTC/ETH markets.

TASK: Extract a single-hop causal chain from the article.
Respond only with the JSON extraction object. Do not include markdown formatting like ```json.

TREATMENT TAXONOMY (classify the event into one of these):
{taxonomy_injection}

MECHANISM WHITELIST (the mechanism MUST be one of these):
{whitelist_injection}

SIMILAR PAST EXTRACTIONS (reference, do not copy blindly):
{few_shot_injection}

RULES:
1. Copy evidence_spans VERBATIM from the article body — exact character sequences.
2. mechanism must exactly match a whitelist entry for the chosen L1 category.
3. If the article does not describe a clear causal event for BTC/ETH, respond with {{"is_impactful": false}}.
4. Use retrieved examples as reference patterns, but extract based on THIS article's actual content.
"""

_FAST_SYSTEM_PROMPT = """You are a financial news causality extractor for BTC/ETH markets.

TASK: Extract a single-hop causal chain from the article.
Respond only with the JSON extraction object. Do not include markdown formatting like ```json.

CRITICAL: `treatment_class` MUST be exactly one of the following 34 categories, or you will fail:
{taxonomy_injection}

SIMILAR PAST EXTRACTIONS (reference, do not copy blindly):
{few_shot_injection}

RULES:
1. Copy evidence_spans VERBATIM from the article body — exact character sequences.
2. If the article does not describe a clear causal event for BTC/ETH, respond with {{"is_impactful": false}}.
3. Use retrieved examples as reference patterns, but extract based on THIS article's actual content.
"""

_CO_T_INSTRUCTION = """
5. CAREFUL PATH: Please provide explicit chain-of-thought reasoning before your final extraction. Explain your classification and mechanism selection in the `reasoning` field.
"""

def build_system_prompt(
    taxonomy: dict[str, list[str]],
    whitelist: dict[tuple[str, str], list[str]],
    few_shot_examples: list[dict],
    path: str,
) -> str:
    if path == "fast":
        return _FAST_SYSTEM_PROMPT.format(
            taxonomy_injection=format_compact_taxonomy(taxonomy),
            few_shot_injection=format_few_shot(few_shot_examples, 3),
        )
    
    prompt = _BASE_SYSTEM_PROMPT.format(
        taxonomy_injection=format_full_taxonomy(taxonomy),
        whitelist_injection=format_whitelist(whitelist),
        few_shot_injection=format_few_shot(few_shot_examples, 5),
    )
    if path == "careful":
        prompt += _CO_T_INSTRUCTION
    return prompt
