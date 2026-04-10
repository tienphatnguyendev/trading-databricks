"""
mas/06_mechanism_validator.py — QA notebook for Mechanism Validator.

Run: uv run python notebooks/qa/mas/06_mechanism_validator.py
"""
import time

import polars as pl

from notebooks.qa.utils import (
    _raise_embed_error,
    build_label_distribution,
    build_scatter,
    log_diag,
    log_plotly,
    mlflow_component_run,
)
from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.nodes.mechanism_validator import validate_extraction
from trading_databricks.agents.taxonomy import MECHANISM_WHITELIST_SEED


def _build_fixture_matrix():
    return pl.DataFrame([
        {
            "case": "S1_accept_happy_path",
            "extraction": {"treatment_class": "Regulation.etf_ruling", "trigger_event": "SEC approves spot Bitcoin ETF", "mechanism": "institutional_access", "expected_direction": "bullish", "evidence_spans": ["SEC approves"]},
            "article_body": "SEC approves spot Bitcoin ETF today",
            "expected_status": "accepted",
            "expected_rejected_raw": None
        },
        {
            "case": "S5_embedding_fail",
            "extraction": {"treatment_class": "Regulation.etf_ruling", "trigger_event": "SEC approves spot Bitcoin ETF", "mechanism": "institutional_access", "expected_direction": "bullish", "evidence_spans": ["SEC approves"]},
            "article_body": "SEC approves...",
            "expected_status": "accepted",
            "expected_rejected_raw": None
        }
    ])

def _run_cosine_unit_tests():
    from trading_databricks.agents.nodes.mechanism_validator import _cosine_similarity
    return pl.DataFrame([{"match": _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0}])

config = MASConfig()

with mlflow_component_run(layer="mas", component="mechanism_validator"):
    # Section 1 — Load whitelist + build fixture matrix
    whitelist = {
        (r["treatment_l1"], r["mechanism"]): r["valid_directions"]
        for r in MECHANISM_WHITELIST_SEED
    }
    fixtures = _build_fixture_matrix()

    # Section 2 — Exercise node
    def run_case(row):
        embed_fn = _raise_embed_error if row["case"] == "S5_embedding_fail" else (lambda t: [1.0, 0.0, 0.0])
        t0 = time.time()
        result = validate_extraction(
            row["extraction"], row["article_body"], whitelist,
            config=config.validator, embed_fn=embed_fn
        )
        return {**row, **result, "latency_ms": (time.time() - t0) * 1000}

    matrix_dicts = [run_case(r) for r in fixtures.iter_rows(named=True)]
    matrix = pl.DataFrame(matrix_dicts)
    matrix = matrix.with_columns([
        pl.col("validation_status").alias("actual_status"),
        pl.col("rejected_mechanism_raw").alias("actual_rejected_raw"),
        pl.col("evidence_relevance_score").alias("actual_score")
    ])

    # Section 3 — Diagnostic tables
    log_diag("validator_fixture_matrix", matrix)
    log_diag("cosine_similarity_unit", _run_cosine_unit_tests())

    # Section 4 — Visual inspection
    log_plotly("validation_status_distribution", build_label_distribution(matrix["actual_status"]))
    log_plotly("evidence_relevance_histogram", build_scatter(matrix, x="case", y="actual_score"))

    # Section 5 — Sanity assertions
    assert (matrix["actual_status"] == matrix["expected_status"]).all()

    s5_filter = matrix.filter(pl.col("case") == "S5_embedding_fail")
    if s5_filter.height > 0:
        assert s5_filter["actual_score"][0] is None
