"""RAG retriever node — second step of the causality sub-pipeline.

Queries the news firehose Vector Search index for precedent articles, then
joins each hit to OHLC outcomes (±N hours) so the downstream LLM sees what
actually moved the market after similar past events.

Externals are wrapped in module-level helpers (`_query_firehose`,
`_join_market_outcomes`) so tests can monkeypatch them without standing up
Vector Search or Spark.
"""
from __future__ import annotations

import logging
from typing import Any

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.state import MASState

logger = logging.getLogger(__name__)


def _build_query_text(headline: str, evidence: list[dict]) -> str:
    """Build the semantic query string from headline + top evidence span."""
    if not evidence:
        return headline
    span = (evidence[0].get("span") or "").strip()
    return f"{headline}\n{span}" if span else headline


def _query_firehose(query_text: str, *, index_name: str, num_results: int, timeout_s: float) -> list[dict]:
    """Query the news firehose VS index. Real implementation calls Databricks VS.

    For local testing, this function is monkeypatched. The real implementation
    is added once the firehose index is provisioned (separate ticket).

    Returns a list of dicts with at least: headline, body_excerpt, timestamp.
    """
    raise NotImplementedError(
        "Firehose VS query requires the news_firehose_idx to be provisioned. "
        "Tests must monkeypatch this function."
    )


def _join_market_outcomes(
    precedents: list[dict],
    *,
    symbol: str,
    ohlc_table: str,
    window_hours: int,
) -> list[dict]:
    """Join each precedent to OHLC return over ±window_hours.

    Returns the precedents list with `return_24h` and `direction` ("up" |
    "down" | "flat") fields added. Precedents whose OHLC join fails are
    dropped from the returned list.

    Real implementation runs a Spark SQL query; tests monkeypatch.
    """
    raise NotImplementedError(
        "OHLC join requires Spark and ohlc_gold table. Tests must monkeypatch."
    )


def rag_retriever(state: MASState, config: MASConfig) -> dict[str, Any]:
    """Retrieve precedent news + market outcomes.

    Returns a dict with keys:
        precedents: list[dict]
        precedent_status: str  "ok" | "no_match" | "unavailable"
    """
    query_text = _build_query_text(state.get("headline", ""), state.get("evidence", []))
    symbol = state.get("symbol", "")

    try:
        hits = _query_firehose(
            query_text,
            index_name=config.rag.firehose_index_name,
            num_results=config.rag.num_precedents,
            timeout_s=config.rag.timeout_s,
        )
    except Exception as exc:
        logger.warning("rag_retriever firehose query failed: %s", exc)
        return {"precedents": [], "precedent_status": "unavailable"}

    if not hits:
        return {"precedents": [], "precedent_status": "no_match"}

    try:
        joined = _join_market_outcomes(
            hits,
            symbol=symbol,
            ohlc_table=config.rag.ohlc_table,
            window_hours=config.rag.outcome_window_hours,
        )
    except Exception as exc:
        logger.warning("rag_retriever OHLC join failed: %s", exc)
        return {"precedents": [], "precedent_status": "unavailable"}

    if not joined:
        return {"precedents": [], "precedent_status": "no_match"}

    return {"precedents": joined, "precedent_status": "ok"}
