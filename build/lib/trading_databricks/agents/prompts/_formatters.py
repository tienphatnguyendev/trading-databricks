"""Formatters for taxonomy, whitelist, and few-shot injection into prompts."""
from __future__ import annotations

import json


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
        chain_json = ex.get("causal_chain_json", "{}")
        if not isinstance(chain_json, str):
            chain_json = json.dumps(chain_json)
        formatted.append(
            f"Example {i}:\nHeadline: {ex.get('trigger_event', '')}\nExtraction:\n{chain_json}"
        )
    return "\n\n".join(formatted)
