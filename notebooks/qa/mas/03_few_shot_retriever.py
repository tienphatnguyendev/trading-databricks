"""
mas/03_few_shot_retriever.py — QA notebook for Few-Shot Retriever.

Run: uv run python notebooks/qa/mas/03_few_shot_retriever.py
"""
import polars as pl

from notebooks.qa.utils import (
    _patch_vs_index,
    build_label_distribution,
    load_article_sample,
    log_diag,
    log_plotly,
    mlflow_component_run,
)
from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.nodes.few_shot_retriever import (
    few_shot_retriever,
    merge_examples,
)


def _run_merge_unit_tests():
    tests = [
        (0, 5, 5, 5), (2, 5, 5, 5), (5, 5, 5, 5),
        (6, 5, 5, 5), (3, 5, 3, 3), (0, 0, 5, 0)
    ]
    rows = []
    for d, s, max_t, expected in tests:
        dyn = [{"similarity_score": 0.9}] * d
        stat = [{"similarity_score": 1.0}] * s
        res = merge_examples(dyn, stat, max_total=max_t)
        assert len(res) == expected
        rows.append({"n_dynamic": d, "n_static": s, "max_total": max_t, "expected_len": expected})
    return pl.DataFrame(rows)

def _to_df(results):
    return pl.DataFrame([{
        "headline": r.get("headline"),
        "n_examples": len(r.get("few_shot_examples", [])),
        "retrieval_story_ids": r.get("retrieval_story_ids", []),
    } for r in results])

config = MASConfig()

with mlflow_component_run(
    layer="mas", component="few_shot_retriever",
    params={"sample_size": 10}
):
    # Section 1 — Load input
    articles = load_article_sample(n=10)

    # Section 2 — Exercise node across 3 phases
    with _patch_vs_index(mode="cold"):
        cold = [few_shot_retriever({"headline": a["headline"]}, config=config) for a in articles]

    with _patch_vs_index(mode="success"):
        warm_ok = [few_shot_retriever({"headline": a["headline"]}, config=config) for a in articles]

    with _patch_vs_index(mode="fail"):
        warm_bad = [few_shot_retriever({"headline": a["headline"]}, config=config) for a in articles]

    # Section 3 — Diagnostic tables
    log_diag("few_shot_cold_start", _to_df(cold))
    log_diag("few_shot_warm_success", _to_df(warm_ok))
    log_diag("few_shot_warm_failure", _to_df(warm_bad))
    log_diag("merge_examples_unit_table", _run_merge_unit_tests())

    # Section 4 — Visual inspection
    # Count occurrences of n_examples by combining phase outputs
    composition_rows = []
    for run_list, phase in [(cold, "cold"), (warm_ok, "success"), (warm_bad, "fail")]:
        for r in run_list:
            composition_rows.append({"phase": phase, "n_examples": len(r.get("few_shot_examples", []))})

    comp_df = pl.DataFrame(composition_rows)
    if comp_df.height > 0:
        log_plotly("cold_vs_warm_composition", build_label_distribution(comp_df["phase"]))

    # Section 5 — Sanity assertions
    assert all(len(r["few_shot_examples"]) == 5 for r in cold)
    assert all(r["retrieval_story_ids"] == [] for r in cold + warm_bad)
