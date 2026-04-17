"""Prompt management for MAS agents.

Runtime loads prompts from MLflow Prompt Registry via ``load_prompt``.
YAML files in this package are the source of truth (synced via ``_sync.py``).
"""
from __future__ import annotations

from trading_databricks.agents.prompts._formatters import (
    format_compact_taxonomy,
    format_few_shot,
    format_full_taxonomy,
    format_whitelist,
)

__all__ = [
    "format_compact_taxonomy",
    "format_full_taxonomy",
    "format_whitelist",
    "format_few_shot",
]
