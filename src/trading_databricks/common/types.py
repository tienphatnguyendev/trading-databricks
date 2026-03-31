"""Shared type aliases and constants for trading_databricks."""
from __future__ import annotations

from pyspark.sql import DataFrame

# Type alias so all transform functions share the same signature shape
TransformFn = "Callable[[DataFrame], DataFrame]"
