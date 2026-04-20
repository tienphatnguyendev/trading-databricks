"""MLflow PyFunc model for the MAS news-causal agent.

Wraps the LangGraph MAS pipeline as a structured-I/O PyFunc model:
  Input:  {headline: str, body: str, published_at: str}
  Output: {treatment_class, trigger_event, mechanism, asset, expected_direction,
           confidence, causal_chain_json, evidence_spans, validation_status,
           rejected_mechanism_raw, retrieval_story_ids, extraction_path,
           evidence_relevance_score}

Standard Databricks deployment:
  1. Log:      from trading_databricks.agents.serve import log_model
               log_model()  # inside an mlflow.start_run()
  2. Register: Auto-registers in UC as smart-trading-dev.03_gold.mas_news_causal
  3. Serve:    Deploy endpoint with inference tables enabled
  4. Call:     ai_query('mas-news-causal', struct(headline, body, published_at))

Model Serving note:
  Containers have no Spark / UC table access.  log_model() bundles the
  taxonomy and whitelist as JSON artifacts so load_context() can hydrate
  the graph without a SparkSession.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.pyfunc import PythonModel
from mlflow.types.schema import ColSpec, Schema

logger = logging.getLogger(__name__)

# ── I/O contract (matches gold.py get_json_object keys) ──────────────────────

INPUT_SCHEMA = Schema([
    ColSpec("string", "headline"),
    ColSpec("string", "body"),
    ColSpec("string", "published_at"),
])

OUTPUT_SCHEMA = Schema([
    ColSpec("string", "treatment_class"),
    ColSpec("string", "trigger_event"),
    ColSpec("string", "mechanism"),
    ColSpec("string", "asset"),
    ColSpec("string", "expected_direction"),
    ColSpec("double", "confidence"),
    ColSpec("string", "causal_chain_json"),
    ColSpec("string", "evidence_spans"),         # JSON-encoded array<string>
    ColSpec("string", "validation_status"),
    ColSpec("string", "rejected_mechanism_raw"),
    ColSpec("string", "retrieval_story_ids"),     # JSON-encoded array<string>
    ColSpec("string", "extraction_path"),
    ColSpec("double", "evidence_relevance_score"),
])

MODEL_SIGNATURE = ModelSignature(inputs=INPUT_SCHEMA, outputs=OUTPUT_SCHEMA)

# ── Whitelist serialization helpers ───────────────────────────────────────

_WL_KEY_SEP = "|||"


def _serialize_whitelist(whitelist: dict[tuple[str, str], list[str]]) -> dict[str, list[str]]:
    """Convert tuple-keyed whitelist to JSON-safe string keys.

    ``(l1, mechanism)`` → ``"l1|||mechanism"``.
    """
    return {f"{k[0]}{_WL_KEY_SEP}{k[1]}": v for k, v in whitelist.items()}


def _deserialize_whitelist(raw: dict[str, list[str]]) -> dict[tuple[str, str], list[str]]:
    """Restore tuple keys from JSON-safe string keys.

    ``"l1|||mechanism"`` → ``(l1, mechanism)``.
    """
    return {tuple(k.split(_WL_KEY_SEP, 1)): v for k, v in raw.items()}


# ── Output helpers ────────────────────────────────────────────────────────


def _extract_causal_features(state: dict[str, Any]) -> dict[str, Any]:
    """Map LangGraph MASState → ai_query JSON contract.

    Keys match the ``$.field`` paths used by ``gold.py`` ``get_json_object()``
    calls.  Array fields are JSON-encoded so that the full prediction dict
    round-trips cleanly through ``ai_query()`` STRING serialisation.
    """
    extraction = state.get("extraction") or {}
    evidence = extraction.get("evidence_spans")
    retrieval_ids = state.get("retrieval_story_ids")
    return {
        "treatment_class": extraction.get("treatment_class"),
        "trigger_event": extraction.get("trigger_event"),
        "mechanism": extraction.get("mechanism"),
        "asset": extraction.get("asset"),
        # Prompt outputs "expected_direction"; legacy CausalChain used "direction".
        # Try new key first, fall back to old key for backward compatibility.
        "expected_direction": extraction.get("expected_direction") or extraction.get("direction"),
        "confidence": extraction.get("confidence"),  # LLM outputs float [0,1]
        "causal_chain_json": json.dumps(extraction) if extraction else None,
        "evidence_spans": json.dumps(evidence) if evidence else None,
        "validation_status": state.get("validation_status"),
        "rejected_mechanism_raw": state.get("rejected_mechanism_raw"),
        "retrieval_story_ids": json.dumps(retrieval_ids) if retrieval_ids else None,
        "extraction_path": state.get("extraction_path"),
        "evidence_relevance_score": state.get("evidence_relevance_score"),
    }


def _error_result() -> dict[str, Any]:
    """Null-filled result dict for articles that fail processing."""
    return dict.fromkeys(
        [
            "treatment_class", "trigger_event", "mechanism", "asset",
            "expected_direction", "confidence", "causal_chain_json",
            "evidence_spans", "rejected_mechanism_raw",
            "retrieval_story_ids", "extraction_path",
            "evidence_relevance_score",
        ],
    ) | {"validation_status": "failed"}


# ── PyFunc Model ────────────────────────────────────────────────────────


class MASPyfuncModel(PythonModel):
    """MLflow PyFunc wrapper for the MAS LangGraph agent.

    ``load_context`` hydrates the compiled StateGraph from bundled artifacts
    (taxonomy + whitelist JSON) so the model runs without Spark in Model
    Serving containers.  ``predict`` invokes the graph per-row and returns
    the 13-field causal feature dict.
    """

    def load_context(self, context: Any) -> None:
        """Build the MAS LangGraph agent from bundled artifacts.

        Model Serving containers have no Spark session or UC table access.
        We monkey-patch taxonomy loaders and Spark helpers BEFORE importing
        the graph module so that all ``from X import Y`` bindings in
        graph.py (and nodes imported transitively) resolve to the patched
        functions.

        Patch order matters:
          1. taxonomy  — causality_extractor imports load_taxonomy/load_whitelist
          2. few_shot_retriever — nodes/__init__ triggers causality_extractor
          3. graph      — binds get_spark_session & load_whitelist from above
        """
        # ── Step 0: Load bundled artifacts (pure JSON, no imports). ───
        taxonomy = self._load_artifact(context, "taxonomy")
        whitelist_raw = self._load_artifact(context, "whitelist")
        whitelist = _deserialize_whitelist(whitelist_raw) if whitelist_raw else None

        # ── Step 1: Patch taxonomy loaders (before nodes/__init__). ───
        import trading_databricks.agents.taxonomy as _tax
        _tax.load_taxonomy = lambda *a, **kw: taxonomy
        _tax.load_whitelist = lambda *a, **kw: whitelist

        # ── Step 2: Patch Spark helpers (triggers nodes/__init__). ────
        import trading_databricks.agents.nodes.few_shot_retriever as _fsr
        _fsr.get_spark_session = lambda: None
        _fsr.get_accepted_chain_count = lambda *a, **kw: 0

        # ── Step 3: Import graph (binds patched functions). ───────────
        from trading_databricks.agents.config import MASConfig
        from trading_databricks.agents.graph import build_mas_graph

        self._config = MASConfig()
        self._graph = build_mas_graph(
            config=self._config,
            taxonomy=taxonomy,
            whitelist=whitelist,
        )
        logger.info(
            "MAS agent loaded: catalog=%s, version=%s",
            self._config.catalog,
            self._config.agent_model_version,
        )

    @mlflow.trace(name="mas_predict")
    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Process news articles through the MAS agent.

        Args:
            model_input: DataFrame with columns headline, body, published_at.
            params: Unused — reserved for future config overrides.

        Returns:
            DataFrame with 13 causal feature columns per article.
        """
        results: list[dict[str, Any]] = []
        for _, row in model_input.iterrows():
            try:
                state = self._parse_input(row)
                output = self._graph.invoke(state)
                results.append(_extract_causal_features(output))
            except Exception:
                logger.exception(
                    "MAS agent failed for article: %.80s",
                    row.get("headline", ""),
                )
                results.append(_error_result())
        return pd.DataFrame(results)

    @staticmethod
    def _parse_input(row: pd.Series) -> dict[str, str]:
        """Extract headline/body/published_at from model input row.

        Handles both structured input (named columns from ``struct()``) and
        a single JSON string column (from ``to_json(struct(...))``).
        """
        if "headline" in row.index:
            return {
                "headline": str(row["headline"]) if row["headline"] is not None else "",
                "body": str(row["body"]) if row.get("body") is not None else "",
                "published_at": str(row["published_at"]) if row.get("published_at") is not None else "",
            }
        # Fallback: single column with a JSON string.
        data = json.loads(str(row.iloc[0]))
        return {
            "headline": data.get("headline", ""),
            "body": data.get("body", ""),
            "published_at": data.get("published_at", ""),
        }

    @staticmethod
    def _load_artifact(context: Any, name: str) -> dict | None:
        """Load a JSON artifact bundled at log time, or ``None``."""
        artifacts = getattr(context, "artifacts", None) or {}
        path = artifacts.get(name)
        if path is None:
            return None
        with open(path) as f:
            return json.load(f)

    def __repr__(self) -> str:
        config = getattr(self, "_config", None)
        version = config.agent_model_version if config else "unloaded"
        return f"{type(self).__name__}(version={version!r})"


# ── Logging helper ────────────────────────────────────────────────────────


def log_model(
    *,
    registered_model_name: str = "smart-trading-dev.03_gold.mas_news_causal",
    artifact_path: str = "mas-news-causal",
) -> mlflow.models.ModelInfo:
    """Log the MAS agent to MLflow and register in Unity Catalog.

    Must be called inside an ``mlflow.start_run()`` context **with Spark
    available** (notebook or SDP environment).  Reads taxonomy and whitelist
    from UC, serialises them as JSON artifacts so the model can load without
    Spark at serving time.

    Returns:
        ModelInfo with ``model_uri`` for deployment.
    """
    import tempfile
    from pathlib import Path

    from mlflow.models.resources import (
        DatabricksServingEndpoint,
        DatabricksVectorSearchIndex,
    )

    from trading_databricks.agents.config import MASConfig
    from trading_databricks.agents.nodes.few_shot_retriever import get_spark_session
    from trading_databricks.agents.taxonomy import load_taxonomy, load_whitelist

    config = MASConfig()
    spark = get_spark_session()

    # Bundle reference tables as JSON artifacts for Model Serving.
    taxonomy = load_taxonomy(spark, catalog=config.catalog, schema=config.schema)
    whitelist = load_whitelist(spark, catalog=config.catalog, schema=config.schema)

    tmp = Path(tempfile.mkdtemp())
    taxonomy_path = tmp / "taxonomy.json"
    whitelist_path = tmp / "whitelist.json"
    taxonomy_path.write_text(json.dumps(taxonomy, default=str))
    # Whitelist has tuple keys (l1, mechanism) → serialize to string keys.
    whitelist_path.write_text(json.dumps(_serialize_whitelist(whitelist), default=str))

    input_example = pd.DataFrame([{
        "headline": "Fed cuts interest rates by 50 basis points",
        "body": (
            "The Federal Reserve announced a 50 basis point rate cut, "
            "signaling a shift toward monetary easing."
        ),
        "published_at": "2026-04-14T00:00:00Z",
    }])

    return mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=MASPyfuncModel(),
        registered_model_name=registered_model_name,
        signature=MODEL_SIGNATURE,
        input_example=input_example,
        artifacts={
            "taxonomy": str(taxonomy_path),
            "whitelist": str(whitelist_path),
        },
        pip_requirements=[
            "langgraph>=1.1.6",
            "spacy==3.8.4",
            "pydantic>=2.0",
            "mlflow>=2.18",
        ],
        resources=[
            DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3-1-8b-instruct"),
            DatabricksServingEndpoint(endpoint_name="databricks-gte-large-en"),
            DatabricksVectorSearchIndex(index_name="smart-trading-dev.03_gold.causal_chains_idx"),
        ],
    )
