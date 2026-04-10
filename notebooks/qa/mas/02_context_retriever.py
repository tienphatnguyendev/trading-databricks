"""
mas/02_context_retriever.py — QA notebook for Context Retriever.

Run: uv run python notebooks/qa/mas/02_context_retriever.py
"""
import polars as pl

from notebooks.qa.utils import (
    build_label_distribution,
    load_article_sample,
    log_diag,
    log_plotly,
    mlflow_component_run,
)
from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.nodes.context_retriever import context_retriever

config = MASConfig()

with mlflow_component_run(
    layer="mas", component="context_retriever",
    params={"sample_size": 20, "stub_version": "v1"}
):
    # Section 1 — Load input
    articles = load_article_sample(n=20)
    initial_states = [
        {"headline": a["headline"], "body": a["body"], "published_at": a["published_at"]}
        for a in articles
    ]

    # Section 2 — Exercise node
    results = [context_retriever(s, config=config) for s in initial_states]
    df = pl.DataFrame(results)

    # Section 3 — Diagnostic tables
    log_diag("context_retriever_outputs", df)

    # Section 4 — Visual inspection
    if df.height > 0:
        # Show null coverage for context fields
        null_counts = pl.DataFrame({
            "market_vol": [df["market_vol_at_publish"].null_count()],
            "market_trend": [df["market_trend_at_publish"].null_count()],
            "dollar_bar_close": [df["dollar_bar_close"].null_count()]
        })
        log_plotly("null_coverage", build_label_distribution(null_counts.melt()["value"]))

    # Section 5 — Sanity assertions
    assert df["market_vol_at_publish"].null_count() == len(df)
    assert df["market_trend_at_publish"].null_count() == len(df)
    assert df["dollar_bar_close"].null_count() == len(df)

    for r, inp in zip(results, articles):
        assert r["headline"] == inp["headline"]
        assert "body" in r and "published_at" in r
