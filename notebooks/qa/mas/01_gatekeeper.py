"""
mas/01_gatekeeper.py — QA notebook for Gatekeeper (Stage 1) component.

Run: uv run python notebooks/qa/mas/01_gatekeeper.py
"""
import polars as pl

from notebooks.qa.utils import (
    build_histogram,
    build_label_distribution,
    load_article_sample,
    log_diag,
    log_plotly,
    mlflow_component_run,
    summarize,
)
from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.gatekeeper import (
    build_nlp,
    extract_ner_features,
    is_crypto_relevant,
)

config = MASConfig()

with mlflow_component_run(
    layer="mas", component="gatekeeper",
    params={"sample_size": 200, "text_window": config.gatekeeper.text_window, "ner_only_mode": True}
):
    # Section 1 — Load input
    articles = load_article_sample(n=200)

    # Section 2 — Exercise node
    nlp = build_nlp()
    rows = []
    for a in articles:
        doc = nlp(a["headline"] + " " + a["body"][:config.gatekeeper.text_window])
        feats = extract_ner_features(doc)
        is_impactful, conf = is_crypto_relevant(a["headline"], a["body"], config=config.gatekeeper)
        rows.append({**a, **feats, "is_impactful": is_impactful, "confidence": conf})

    df = pl.DataFrame(rows)

    # Section 3 — Diagnostic tables
    log_diag("ner_per_article", df)

    summary_cols = ["is_impactful", "entity_count", "has_crypto_asset", "has_exchange", "has_stablecoin", "has_protocol", "has_concept"]
    log_diag("ner_coverage_summary", summarize(df, summary_cols))

    # Section 4 — Visual inspection
    if df.height > 0:
        log_plotly("entity_distribution", build_label_distribution(df["has_crypto_asset"]))
        log_plotly("category_coverage", build_label_distribution(df["is_impactful"]))
        log_plotly("entity_count_histogram", build_histogram(df["entity_count"]))

    # Section 5 — Sanity assertions
    assert df["is_impactful"].dtype == pl.Boolean
    assert df["confidence"].min() >= 0.0 and df["confidence"].max() <= 1.0
    assert build_nlp() is build_nlp(), "build_nlp must be cached"
