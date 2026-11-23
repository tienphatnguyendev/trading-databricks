"""Shared DLT expectation helpers.

Expectation rule strings are plain SQL expressions evaluated by SDP's
@dp.expect_or_drop / @dp.expect_or_fail decorators.  Centralising them here
lets tests validate the expressions without touching pipeline.py files.
"""
from __future__ import annotations

# ── Generic helpers ──────────────────────────────────────────────────────────

def not_null(col: str) -> str:
    """Return an IS NOT NULL expectation expression for *col*."""
    return f"{col} IS NOT NULL"


def positive(col: str) -> str:
    """Return a > 0 expectation expression for *col*."""
    return f"{col} > 0"


def non_empty_string(col: str) -> str:
    """Return a non-null AND non-empty string expectation for *col*."""
    return f"{col} IS NOT NULL AND LENGTH(TRIM({col})) > 0"
