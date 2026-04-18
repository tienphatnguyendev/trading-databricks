"""Provision Databricks Vector Search infrastructure for few-shot retrieval.

This script creates the Vector Search endpoint and Delta Sync index required
by the Multi-Agent System's RAG component.

Usage:
    # Dev environment
    uv run python scripts/provision_vector_search.py --env dev

    # Production
    uv run python scripts/provision_vector_search.py --env prod
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def provision_vs_infrastructure(env: str) -> None:
    """Provision VS endpoint and index for the given environment.

    Args:
        env: Environment ('dev' or 'prod')

    Raises:
        ValueError: If environment is not 'dev' or 'prod'
        ImportError: If Databricks MCP tools are not available
    """
    # Import config
    from trading_databricks.agents.config import MASConfig
    from trading_databricks.common.uc_identifiers import (
        CAUSAL_CHAINS_INDEX_TABLE,
        EMBEDDING_ENDPOINT_NAME,
        GOLD_SCHEMA,
        NEWS_CAUSAL_FEATURES_TABLE,
        quoted_table_fqn,
        table_fqn,
    )

    # Create config (catalog and index name will be env-specific)
    if env == "dev":
        config = MASConfig()
    elif env == "prod":
        config = MASConfig(
            catalog="trading-signal-gen-prod",
            schema=GOLD_SCHEMA,
            vs_index_name=table_fqn(
                "trading-signal-gen-prod",
                GOLD_SCHEMA,
                CAUSAL_CHAINS_INDEX_TABLE,
            ),
        )
    else:
        raise ValueError(f"Invalid environment: {env}. Must be 'dev' or 'prod'")

    logger.info(f"Provisioning VS infrastructure for environment: {env}")
    logger.info(f"Catalog: {config.catalog}, Schema: {config.schema}")

    from databricks.connect import DatabricksSession
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.vectorsearch import (
        DeltaSyncVectorIndexSpecResponse,
        EndpointType,
    )

    from trading_databricks.agents.schemas import CAUSAL_FEATURES_SCHEMA

    w = WorkspaceClient()

    # Ensure source table exists before index creation
    logger.info("Ensuring source table exists...")
    spark = DatabricksSession.builder.serverless(True).getOrCreate()
    source_table = table_fqn(config.catalog, config.schema, NEWS_CAUSAL_FEATURES_TABLE)
    source_table_quoted = quoted_table_fqn(
        config.catalog,
        config.schema,
        NEWS_CAUSAL_FEATURES_TABLE,
    )
    spark.createDataFrame([], CAUSAL_FEATURES_SCHEMA).write.format("delta").option(
        "mergeSchema", "true"
    ).mode("ignore").saveAsTable(source_table_quoted)

    # Step 1: Create/update VS endpoint
    endpoint_name = config.vs_endpoint_name  # "mas-few-shot"
    logger.info(f"Step 1: Creating/updating VS endpoint: {endpoint_name}")

    try:
        logger.info(f"Creating/updating VS endpoint: {endpoint_name}...")
        w.vector_search_endpoints.create_endpoint_and_wait(
            name=endpoint_name, endpoint_type=EndpointType.STANDARD
        )
        logger.info(f"  ✓ Endpoint '{endpoint_name}' ready")
    except Exception as e:
        logger.error(f"Failed to create endpoint: {e}")
        raise

    # Step 2: Create/update VS Delta Sync index
    index_name = config.vs_index_name

    logger.info(f"Step 2: Creating/updating VS index: {index_name}")
    logger.info(f"  Source table: {source_table}")

    try:
        logger.info(f"Creating/updating VS index: {index_name}...")
        w.vector_search_indexes.create_index_and_wait(
            name=index_name,
            endpoint_name=endpoint_name,
            primary_key="story_id",
            index_type="DELTA_SYNC",
            delta_sync_index_spec=DeltaSyncVectorIndexSpecResponse(
                source_table=source_table,
                embedding_source_columns=[
                    {
                        "name": "trigger_event",
                        "embedding_model_endpoint_name": EMBEDDING_ENDPOINT_NAME,
                    }
                ],
                pipeline_type="TRIGGERED",
            ),
        )
        logger.info(f"  ✓ Index '{index_name}' configured")
    except Exception as e:
        logger.error(f"Failed to create index: {e}")
        raise

    # Step 3: Log verification steps
    logger.info("Step 3: Verify provisioning")
    logger.info(
        f"  Monitor endpoint status: "
        f"databricks vector-search endpoints get {endpoint_name}"
    )
    logger.info(
        f"  Monitor index status: databricks vector-search indexes get {index_name}"
    )

    # Step 4: Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ VS INFRASTRUCTURE PROVISIONING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Environment:     {env}")
    logger.info(f"Catalog:         {config.catalog}")
    logger.info(f"Endpoint:        {endpoint_name} (STANDARD type)")
    logger.info(f"Index:           {index_name}")
    logger.info(f"Source table:    {source_table}")
    logger.info(f"Embedding model: {EMBEDDING_ENDPOINT_NAME}")
    logger.info("Sync mode:       TRIGGERED (manual refresh)")
    logger.info("")
    logger.info("Next steps:")
    logger.info(
        "1. Verify endpoint is ONLINE: "
        "databricks vector-search endpoints get mas-few-shot"
    )
    logger.info(
        f"2. Verify index is READY: databricks vector-search indexes get {index_name}"
    )
    logger.info("3. Trigger initial sync after news_causal_features table is populated")
    logger.info("4. Test few-shot retrieval with sample headline")
    logger.info("")


def main() -> None:
    """Parse arguments and provision VS infrastructure."""
    parser = argparse.ArgumentParser(
        description=(
            "Provision Databricks Vector Search infrastructure for MAS few-shot retrieval"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Provision dev environment\n"
            "  uv run python scripts/provision_vector_search.py --env dev\n"
            "\n"
            "  # Provision production environment\n"
            "  uv run python scripts/provision_vector_search.py --env prod\n"
        ),
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=["dev", "prod"],
        help="Environment to provision (dev or prod)",
    )

    args = parser.parse_args()

    try:
        provision_vs_infrastructure(args.env)
        sys.exit(0)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Provisioning failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
