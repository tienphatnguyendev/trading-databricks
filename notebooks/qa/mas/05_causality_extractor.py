"""
mas/05_causality_extractor.py — QA notebook for Causality Extractor.

Run: uv run python notebooks/qa/mas/05_causality_extractor.py
"""
import os
import time

import polars as pl

from notebooks.qa.utils import (
    _prepare_local_graph_environment,
    build_label_distribution,
    build_scatter,
    load_article_sample,
    log_diag,
    log_plotly,
    mlflow_component_run,
)
from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.nodes.causality_extractor import (
    causality_extractor,
    majority_vote,
    parse_json_response,
)
from trading_databricks.agents.nodes.few_shot_retriever import STATIC_FEW_SHOT_EXAMPLES

# Default to mock LLM for QA tests to save tokens and ensure offline reliability
os.environ["MAS_QA_MOCK_LLM"] = os.environ.get("MAS_QA_MOCK_LLM", "1")

def _run_parse_unit_tests():
    tests = [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"b": 2}\n```', {"b": 2}),
        ('```\n{"c": 3}\n```', {"c": 3}),
        ('invalid', None),
        ('', None)
    ]
    return pl.DataFrame([
        {"input": t[0], "parsed": str(parse_json_response(t[0])), "match": parse_json_response(t[0]) == t[1]}
        for t in tests
    ])

def _run_majority_vote_unit_tests():
    ex1 = {"treatment_class": "A", "mechanism": "m", "expected_direction": "bullish", "confidence": 0.9}
    res = majority_vote([ex1, ex1, ex1])
    return pl.DataFrame([{"match": res == ex1}])

config = MASConfig()

with mlflow_component_run(
    layer="mas", component="causality_extractor",
    params={"sample_size": 6}
):
    with _prepare_local_graph_environment():
        # Section 1 — Load input
        articles = load_article_sample(n=6)
        results_by_path = {}

        # Section 2 — Exercise node, 3 paths
        for path in ("fast", "standard", "careful"):
            rows = []
            for a in articles:
                state = {
                    "headline": a["headline"],
                    "body": a["body"],
                    "few_shot_examples": STATIC_FEW_SHOT_EXAMPLES,
                    "extraction_path": path
                }
                t0 = time.time()
                out = causality_extractor(state, config)
                latency_ms = (time.time() - t0) * 1000

                ext = out.get("extraction") or {}
                rows.append({
                    **a,
                    "latency_ms": latency_ms,
                    "path": path,
                    "parse_success": bool(ext),
                    "treatment_class": ext.get("treatment_class"),
                    "mechanism": ext.get("mechanism"),
                    "confidence": ext.get("confidence")
                })
            results_by_path[path] = pl.DataFrame(rows)

    # Section 3 — Diagnostic tables
    for path, df in results_by_path.items():
        log_diag(f"extractor_{path}_outputs", df, metrics={f"{path}_parse_success_rate": df["parse_success"].mean()})

    log_diag("parse_json_response_unit", _run_parse_unit_tests())
    log_diag("majority_vote_unit", _run_majority_vote_unit_tests())

    # Section 4 — Visual inspection
    combined = pl.concat(list(results_by_path.values()))
    if combined.height > 0:
        log_plotly("latency_by_path", build_scatter(combined, x="path", y="latency_ms"))
        log_plotly("parse_success_by_path", build_label_distribution(combined["parse_success"]))

    # Section 5 — Sanity assertions
    assert all(df["parse_success"].sum() >= 1 for df in results_by_path.values())
