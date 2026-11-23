"""RAG retriever node — precedent news retrieval with market outcomes.

Supports two retrieval modes (controlled by config.rag.use_chunk_index):
  1. Headline-level: query news_firehose_idx (original, 500-char body_truncated)
  2. Chunk-level (small-to-big): query news_chunks_idx → group by story_id
     → return full parent articles from news_cleaned

Optionally generates HyDE (Hypothetical Document Embeddings) queries for
improved semantic matching when config.rag.hyde_enabled=True.

Externals are wrapped in module-level helpers so tests can monkeypatch them
without standing up Vector Search or Spark.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.state import MASState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------


def _build_query_text(headline: str, evidence: list[dict]) -> str:
    """Build the semantic query string from headline + top evidence span."""
    if not evidence:
        return headline
    span = (evidence[0].get("span") or "").strip()
    return f"{headline}\n{span}" if span else headline


def _build_hyde_query(headline: str, evidence: list[dict]) -> str:
    """Generate a HyDE (Hypothetical Document Embeddings) query.

    Creates a synthetic causal summary that semantically resembles
    what a relevant precedent article would look like. This improves
    retrieval by matching document-space embeddings rather than
    query-space embeddings.
    """
    span = ""
    if evidence:
        span = (evidence[0].get("span") or "").strip()
    # Construct a hypothetical document in the style of our corpus
    return (
        f"{headline}. "
        f"{span + '. ' if span else ''}"
        f"This event caused significant market movement in cryptocurrency "
        f"markets through the identified causal mechanism."
    )


# ---------------------------------------------------------------------------
# External dependencies (monkeypatchable for tests)
# ---------------------------------------------------------------------------


def _query_vs_index(
    query_text: str,
    *,
    index_name: str,
    num_results: int,
    columns: list[str],
    filters_json: str | None = None,
) -> list[dict]:
    """Query a Databricks Vector Search index.

    Real implementation uses the Databricks SDK or MCP tool.
    Tests must monkeypatch this function.
    """
    raise NotImplementedError(
        "VS query requires provisioned index. Tests must monkeypatch."
    )


def _query_firehose(
    query_text: str,
    *,
    index_name: str,
    num_results: int,
    timeout_s: float,
) -> list[dict]:
    """Query headline precedents through the existing firehose helper contract.

    Keep ``timeout_s`` available to callers that supply a provisioned firehose
    implementation. The shared VS helper remains an infrastructure stub.
    """
    return _query_vs_index(
        query_text,
        index_name=index_name,
        num_results=num_results,
        columns=["story_id", "headline", "body_truncated", "published_at", "source"],
    )


def _get_parent_articles(
    story_ids: list[str],
    *,
    catalog: str,
    schema: str,
) -> dict[str, dict]:
    """Fetch full parent articles from silver.news_cleaned by story_id.

    Returns {story_id: {headline, body, published_at, source}}.
    Tests must monkeypatch this function.
    """
    raise NotImplementedError(
        "Parent article fetch requires Spark. Tests must monkeypatch."
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


# ---------------------------------------------------------------------------
# Small-to-big: chunk hits → deduplicated parent articles
# ---------------------------------------------------------------------------


def _chunks_to_parents(
    chunk_hits: list[dict],
    *,
    config: MASConfig,
    max_parents: int = 5,
) -> list[dict]:
    """Group chunk hits by story_id, rank parents by chunk score, fetch full articles.

    Small-to-big pattern:
      1. Chunk-level VS returns top-N chunks (may be from same article)
      2. Group by story_id, aggregate scores (max score per parent)
      3. Return top-K unique parent articles with full body

    Returns list of parent article dicts with:
      story_id, headline, body, published_at, source, retrieval_score
    """
    # Aggregate best score per story_id
    parent_scores: dict[str, float] = defaultdict(float)
    for hit in chunk_hits:
        sid = hit.get("story_id", "")
        score = hit.get("score", 0.0)
        parent_scores[sid] = max(parent_scores[sid], score)

    # Rank parents by best chunk score, take top-K
    ranked_ids = sorted(parent_scores, key=parent_scores.get, reverse=True)[:max_parents]

    if not ranked_ids:
        return []

    # Fetch full parent articles (small-to-big: expand chunk → full doc)
    if config.rag.chunk_to_parent:
        try:
            parents = _get_parent_articles(
                ranked_ids,
                catalog=config.catalog,
                schema="02_silver",
            )
        except Exception as exc:
            logger.warning("Parent article fetch failed: %s", exc)
            # Fall back to chunk text only
            return [
                {"story_id": sid, "retrieval_score": parent_scores[sid]}
                for sid in ranked_ids
            ]
    else:
        parents = {}

    # Build output with full parent context
    results = []
    for sid in ranked_ids:
        parent = parents.get(sid, {})
        results.append({
            "story_id": sid,
            "headline": parent.get("headline", ""),
            "body": parent.get("body", ""),
            "published_at": parent.get("published_at"),
            "source": parent.get("source", ""),
            "retrieval_score": parent_scores[sid],
        })
    return results


# ---------------------------------------------------------------------------
# Main node function
# ---------------------------------------------------------------------------


def rag_retriever(state: MASState, config: MASConfig) -> dict[str, Any]:
    """Retrieve precedent news + market outcomes.

    Supports two retrieval strategies:
      - Headline-level (use_chunk_index=False): queries news_firehose_idx
      - Chunk-level small-to-big (use_chunk_index=True): queries news_chunks_idx,
        deduplicates to parent articles, returns full context

    Optionally uses HyDE queries (hyde_enabled=True) for improved retrieval.

    Returns a dict with keys:
        precedents: list[dict]
        precedent_status: str  "ok" | "no_match" | "unavailable"
    """
    headline = state.get("headline", "")
    evidence = state.get("evidence", [])
    symbol = state.get("symbol", "")

    # Build query text (optionally with HyDE)
    if config.rag.hyde_enabled:
        query_text = _build_hyde_query(headline, evidence)
    else:
        query_text = _build_query_text(headline, evidence)

    # Query Vector Search
    try:
        if config.rag.use_chunk_index:
            hits = _query_vs_index(
                query_text,
                index_name=f"{config.catalog}.02_silver.news_chunks_idx",
                num_results=config.rag.num_precedents * 2,
                columns=[
                    "chunk_id", "story_id", "chunk_text", "chunk_type", "published_at",
                ],
            )
        else:
            hits = _query_firehose(
                query_text,
                index_name=config.rag.firehose_index_name,
                num_results=config.rag.num_precedents,
                timeout_s=config.rag.timeout_s,
            )
    except NotImplementedError:
        logger.info("rag_retriever: VS index not provisioned yet")
        return {"precedents": [], "precedent_status": "unavailable"}
    except Exception as exc:
        logger.warning("rag_retriever VS query failed: %s", exc)
        return {"precedents": [], "precedent_status": "unavailable"}

    if not hits:
        return {"precedents": [], "precedent_status": "no_match"}

    # Process results based on retrieval mode
    if config.rag.use_chunk_index:
        # Small-to-big: chunk hits → parent articles
        precedents = _chunks_to_parents(
            hits, config=config, max_parents=config.rag.num_precedents,
        )
    else:
        # Direct headline-level results
        precedents = hits[:config.rag.num_precedents]

    if not precedents:
        return {"precedents": [], "precedent_status": "no_match"}

    # Only return precedents whose market outcomes are available.
    try:
        joined = _join_market_outcomes(
            precedents,
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
