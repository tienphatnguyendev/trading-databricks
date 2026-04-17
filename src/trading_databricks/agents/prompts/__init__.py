"""Prompt management for MAS agents."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from trading_databricks.agents.prompts._formatters import (
    format_compact_taxonomy,
    format_few_shot,
    format_full_taxonomy,
    format_whitelist,
)

_PACKAGE_DIR = Path(__file__).parent


def _load_from_yaml(name: str) -> list[dict[str, str]]:
    yaml_path = _PACKAGE_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No YAML source for prompt '{name}' at {yaml_path}")
    doc = yaml.safe_load(yaml_path.read_text())
    return [dict(m) for m in doc["messages"]]


def load_prompt(name: str, version: str | int = "production") -> list[dict[str, str]]:
    """Load a prompt as chat messages.

    Prefers MLflow Prompt Registry; falls back to packaged YAML when unavailable.
    """
    try:
        import mlflow

        prompt = mlflow.load_prompt(f"prompts:/{name}/{version}")
        if hasattr(prompt, "messages") and prompt.messages:
            return [dict(m) for m in prompt.messages]
        return [{"role": "system", "content": prompt.template}]
    except Exception:
        return _load_from_yaml(name)


def render_messages(
    messages: list[dict[str, str]],
    variables: dict[str, Any],
) -> list[dict[str, str]]:
    """Substitute ``{var}`` placeholders using ``str.format``.

    Double braces ``{{`` / ``}}`` are preserved as literal braces.
    """
    return [
        {"role": m["role"], "content": m["content"].format(**variables)}
        for m in messages
    ]


__all__ = [
    "load_prompt",
    "render_messages",
    "format_compact_taxonomy",
    "format_full_taxonomy",
    "format_whitelist",
    "format_few_shot",
]
