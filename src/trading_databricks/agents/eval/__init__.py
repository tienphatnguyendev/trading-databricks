"""Evaluation harness for the MAS causality pipeline.

Public API:
    - build_eval_dataset: load / materialize MLflow GenAI datasets
    - run_evaluation: one-shot evaluation driver
    - ALL_SCORERS, DETERMINISTIC_SCORERS, JUDGE_SCORERS, REFERENCE_SCORERS
    - evaluate_retrieval, summarize_retrieval: IR metrics for Vector Search
"""
from __future__ import annotations

from trading_databricks.agents.eval.dataset import build_eval_dataset
from trading_databricks.agents.eval.retrieval_scorers import (
    RetrievalMetricsConfig,
    evaluate_retrieval,
    summarize_retrieval,
)
from trading_databricks.agents.eval.runner import run_evaluation
from trading_databricks.agents.eval.scorers import (
    ALL_SCORERS,
    DETERMINISTIC_SCORERS,
    JUDGE_SCORERS,
    REFERENCE_SCORERS,
)

__all__ = [
    "build_eval_dataset",
    "run_evaluation",
    "ALL_SCORERS",
    "DETERMINISTIC_SCORERS",
    "JUDGE_SCORERS",
    "REFERENCE_SCORERS",
    "evaluate_retrieval",
    "summarize_retrieval",
    "RetrievalMetricsConfig",
]
