"""SDP pipeline definition for the MAS (agents) domain.

Pipeline DAG (all materialized views — full batch recompute):

  02_silver.news_chunks        ← silver.news_cleaned → hierarchical chunking
  03_gold.news_firehose        ← silver projection for VS firehose index
  03_gold.news_causal_features ← gatekeeper → ai_query('mas-news-causal') → 23 fields

External dependencies:
  - 02_silver.news_cleaned must exist (owned by the news pipeline)
  - mas-news-causal Model Serving endpoint must be live (for ai_query)
  - 03_gold.treatment_taxonomy + mechanism_whitelist must be seeded

DABs pipeline resource: resources/agents_pipeline.pipeline.yml
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Package path: SDP serverless doesn't add the package root to sys.path.
# The src_root config is injected via pipeline configuration in the YAML.
# ---------------------------------------------------------------------------
import sys

_src_root = spark.conf.get("src_root")  # noqa: F821
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from pyspark import pipelines as dp

from trading_databricks.agents.gold import (
    news_causal_features_transform,
    news_firehose_transform,
)
from trading_databricks.agents.silver import news_chunks_transform

# ── Pipeline configuration (injected by DABs at runtime) ─────────────────────

catalog = spark.conf.get("catalog")  # noqa: F821
silver_schema = spark.conf.get("silver_schema")
gold_schema = spark.conf.get("gold_schema")

# ── Data quality expectations ─────────────────────────────────────────────

_VALID_STORY_ID = "story_id IS NOT NULL"
_VALID_CHUNK_TEXT = "chunk_text IS NOT NULL AND LENGTH(TRIM(chunk_text)) > 0"

# ── Silver: News Chunks ───────────────────────────────────────────────────


@dp.materialized_view(
    name=f"{silver_schema}.news_chunks",
    comment="Hierarchical chunks from news articles for VS embedding (~4.8 chunks/article).",
    cluster_by=["published_at"],
)
@dp.expect_or_drop("valid_story_id", _VALID_STORY_ID)
@dp.expect_or_drop("valid_chunk_text", _VALID_CHUNK_TEXT)
def silver_news_chunks():  # noqa: ANN201
    return news_chunks_transform(
        spark.table(f"`{catalog}`.`{silver_schema}`.news_cleaned")  # noqa: F821
    )


# ── Gold: VS Firehose ────────────────────────────────────────────────────


@dp.materialized_view(
    name=f"{gold_schema}.news_firehose",
    comment="News projection for Vector Search firehose index (headline-embedded).",
    cluster_by=["published_at"],
)
def gold_news_firehose():  # noqa: ANN201
    return news_firehose_transform(
        spark.table(f"`{catalog}`.`{silver_schema}`.news_cleaned")  # noqa: F821
    )


# ── Gold: Causal Features ────────────────────────────────────────────────


@dp.materialized_view(
    name=f"{gold_schema}.news_causal_features",
    comment="Structured causal features from news via MAS agent (23 fields).",
    cluster_by=["published_at"],
)
def gold_news_causal_features():  # noqa: ANN201
    """Gold MV: gatekeeper filter → ai_query('mas-news-causal') → 23 fields.

    Uses SQL RLIKE gatekeeper (no spacy dependency in SDP serverless).
    Reads silver.news_cleaned, filters crypto-relevant articles via RLIKE,
    calls MAS Model Serving endpoint via ai_query, expands JSON response into
    CAUSAL_FEATURES_SCHEMA (23 fields).  Requires mas-news-causal endpoint.
    """
    return news_causal_features_transform(
        spark,  # noqa: F821
        catalog=catalog,
        gatekeeper_mode="sql",
    )
