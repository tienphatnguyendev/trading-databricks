"""Formal information retrieval metrics for MAS Vector Search evaluation.

Metrics implemented:
    - precision_at_k: Fraction of top-k results that are relevant
    - recall_at_k: Fraction of relevant docs found in top-k
    - reciprocal_rank: 1 / rank of first relevant result (MRR component)
    - ndcg_at_k: Normalized Discounted Cumulative Gain
    - evaluate_retrieval: Batch evaluation over labeled query-document pairs

All functions accept plain Python structures (lists/dicts) and return floats.
Batch evaluation returns a Polars DataFrame.

Conventions:
    - Relevance is binary (0/1) — no graded relevance.
    - k defaults to 5 (matching VS `num_results` in few_shot_retriever).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True)
class RetrievalMetricsConfig:
    """Configuration for retrieval evaluation."""

    k: int = 5
    similarity_threshold: float = 0.3


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], *, k: int = 5) -> float:
    """Fraction of top-k retrieved documents that are relevant.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (best first).
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Number of results to consider.

    Returns:
        Precision@k in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for doc_id in top_k if doc_id in relevant_ids) / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], *, k: int = 5) -> float:
    """Fraction of relevant documents found in top-k results.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (best first).
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Number of results to consider.

    Returns:
        Recall@k in [0.0, 1.0]. Returns 0.0 if no relevant docs exist.
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for doc_id in top_k if doc_id in relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal of the rank of the first relevant result.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (best first).
        relevant_ids: Set of ground-truth relevant document IDs.

    Returns:
        1/rank of first relevant hit, or 0.0 if none found.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], *, k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance).

    DCG@k  = sum_{i=1}^{k} rel_i / log2(i + 1)
    IDCG@k = sum_{i=1}^{min(k, |relevant|)} 1 / log2(i + 1)
    NDCG@k = DCG@k / IDCG@k

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (best first).
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Number of results to consider.

    Returns:
        NDCG@k in [0.0, 1.0]. Returns 0.0 if no relevant docs exist.
    """
    if not relevant_ids or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    dcg = sum(
        (1.0 / math.log2(i + 2))  # i+2 because enumerate is 0-indexed, log2(rank+1)
        for i, doc_id in enumerate(top_k)
        if doc_id in relevant_ids
    )

    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    queries: list[dict[str, Any]],
    *,
    config: RetrievalMetricsConfig | None = None,
) -> pl.DataFrame:
    """Batch evaluation of retrieval quality over labeled query-document pairs.

    Args:
        queries: List of dicts, each with:
            - query_id: str — unique query identifier
            - query_text: str — the query headline/text
            - retrieved_ids: list[str] — ordered retrieved document IDs
            - relevant_ids: list[str] — ground-truth relevant document IDs
            - similarity_scores: list[float] (optional) — per-result scores
        config: Evaluation configuration. Defaults to k=5.

    Returns:
        Polars DataFrame with per-query metrics and summary row.
    """
    if config is None:
        config = RetrievalMetricsConfig()

    rows: list[dict[str, Any]] = []
    for q in queries:
        relevant = set(q["relevant_ids"])
        retrieved = q["retrieved_ids"]
        scores = q.get("similarity_scores", [])

        row = {
            "query_id": q["query_id"],
            "query_text": q.get("query_text", ""),
            "n_retrieved": len(retrieved),
            "n_relevant": len(relevant),
            "precision_at_k": precision_at_k(retrieved, relevant, k=config.k),
            "recall_at_k": recall_at_k(retrieved, relevant, k=config.k),
            "mrr": reciprocal_rank(retrieved, relevant),
            "ndcg_at_k": ndcg_at_k(retrieved, relevant, k=config.k),
            "mean_similarity": (
                sum(scores[:config.k]) / len(scores[:config.k])
                if scores
                else None
            ),
        }
        rows.append(row)

    return pl.DataFrame(rows)


def summarize_retrieval(metrics_df: pl.DataFrame) -> dict[str, float]:
    """Compute aggregate retrieval metrics from per-query results.

    Args:
        metrics_df: Output of evaluate_retrieval().

    Returns:
        Dict with mean_precision_at_k, mean_recall_at_k, mean_mrr,
        mean_ndcg_at_k, and n_queries.
    """
    return {
        "n_queries": metrics_df.height,
        "mean_precision_at_k": metrics_df["precision_at_k"].mean(),
        "mean_recall_at_k": metrics_df["recall_at_k"].mean(),
        "mean_mrr": metrics_df["mrr"].mean(),
        "mean_ndcg_at_k": metrics_df["ndcg_at_k"].mean(),
    }
