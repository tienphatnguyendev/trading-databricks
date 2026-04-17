"""SDP pipeline definition for the MAS (agents) domain.

This file contains ONLY @dp.materialized_view decorators.
All transformation logic lives in agents/gold.py and is locally testable.

Pipeline:
  news_causal_features  — materialized view (full recompute, batch)
                          Reads silver.news_cleaned, applies gatekeeper,
                          calls ai_query('mas-news-causal'), expands 23 fields.

DABs pipeline resource: resources/agents_pipeline.pipeline.yml (created in TRD-33)

Note: ai_query() requires the mas-news-causal Model Serving endpoint to be live.
      Deploy the endpoint in TRD-33 before running this pipeline in production.
"""

from __future__ import annotations

from pyspark import pipelines as dp

from trading_databricks.agents.config import GatekeeperConfig
from trading_databricks.agents.gold import news_causal_features_transform
from trading_databricks.common.uc_identifiers import DEFAULT_CATALOG

_CATALOG = DEFAULT_CATALOG


@dp.materialized_view
def news_causal_features():  # noqa: ANN201
    """Gold materialized view: structured causal features from news articles.

    Applies gatekeeper (NER filter) then calls the MAS endpoint via ai_query.
    Full recompute on each pipeline run — no streaming state.
    """
    return news_causal_features_transform(
        spark,  # noqa: F821
        catalog=_CATALOG,
        gatekeeper_config=GatekeeperConfig(),
    )
