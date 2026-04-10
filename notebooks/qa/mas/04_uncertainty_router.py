"""
mas/04_uncertainty_router.py — QA notebook for Uncertainty Router.

Run: uv run python notebooks/qa/mas/04_uncertainty_router.py
"""
import polars as pl

from notebooks.qa.utils import (
    build_path_distribution,
    build_scatter,
    log_diag,
    log_plotly,
    mlflow_component_run,
)
from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.nodes.uncertainty_router import uncertainty_router


def _build_examples(n, top_sim):
    if n == 0:
        return []
    ex = [{"similarity_score": top_sim}]
    for _ in range(n - 1):
        ex.append({"similarity_score": max(0, top_sim - 0.1)})
    return ex

config = MASConfig()

cases = pl.DataFrame([
    {"case": "empty", "n_examples": 0, "top_similarity": 0.0, "expected_path": "careful"},
    {"case": "single", "n_examples": 1, "top_similarity": 0.95, "expected_path": "careful"},
    {"case": "fast_high", "n_examples": 3, "top_similarity": 0.95, "expected_path": "fast"},
    {"case": "fast_boundary", "n_examples": 3, "top_similarity": 0.85, "expected_path": "fast"},
    {"case": "just_below_fast", "n_examples": 3, "top_similarity": 0.8499, "expected_path": "standard"},
    {"case": "standard_mid", "n_examples": 3, "top_similarity": 0.70, "expected_path": "standard"},
    {"case": "standard_low", "n_examples": 3, "top_similarity": 0.5001, "expected_path": "standard"},
    {"case": "careful_boundary", "n_examples": 3, "top_similarity": 0.50, "expected_path": "careful"},
    {"case": "careful_low", "n_examples": 3, "top_similarity": 0.20, "expected_path": "careful"},
    {"case": "careful_zero", "n_examples": 3, "top_similarity": 0.00, "expected_path": "careful"},
    {"case": "missing_score", "n_examples": 3, "top_similarity": 0.00, "expected_path": "careful"},
    {"case": "bulk_fast", "n_examples": 5, "top_similarity": 0.90, "expected_path": "fast"},
])

with mlflow_component_run(
    layer="mas", component="uncertainty_router",
    params={"high_similarity_threshold": config.router.high_similarity_threshold}
):
    # Section 1 & 2 — Load input & Exercise node
    actual = []
    for r in cases.iter_rows(named=True):
        state = {"few_shot_examples": _build_examples(r["n_examples"], r["top_similarity"])}
        if r["case"] == "missing_score":
            state["few_shot_examples"] = [{"other": 1}] * 3

        actual.append(uncertainty_router(state, config)["extraction_path"])

    df = cases.with_columns(actual_path=pl.Series(actual))
    df = df.with_columns(match=pl.col("actual_path") == pl.col("expected_path"))

    # Section 3 — Diagnostic tables
    log_diag("router_decision_matrix", df)

    # Section 4 — Visual inspection
    log_plotly("routing_path_distribution", build_path_distribution(actual))
    log_plotly("similarity_vs_path", build_scatter(df, x="top_similarity", y="actual_path", color="expected_path"))

    # Section 5 — Sanity assertions
    assert df["match"].all(), "Routing mismatch in decision matrix"
    assert set(actual) == {"fast", "standard", "careful"}
