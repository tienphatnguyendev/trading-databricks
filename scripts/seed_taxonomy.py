"""One-shot script to seed treatment taxonomy and mechanism whitelist tables.

Usage:
    uv run python scripts/seed_taxonomy.py
    uv run python scripts/seed_taxonomy.py --catalog trading-signal-gen-prod --schema 03_gold
"""
from __future__ import annotations

import argparse

from databricks.connect import DatabricksSession

from trading_databricks.agents.taxonomy import seed_taxonomy
from trading_databricks.common.uc_identifiers import (
    DEFAULT_CATALOG,
    GOLD_SCHEMA,
    MECHANISM_WHITELIST_TABLE,
    TREATMENT_TAXONOMY_TABLE,
    quoted_table_fqn,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--schema", default=GOLD_SCHEMA)
    args = parser.parse_args()

    spark = DatabricksSession.builder.serverless(True).getOrCreate()
    seed_taxonomy(spark, catalog=args.catalog, schema=args.schema)

    taxonomy_count = spark.table(
        quoted_table_fqn(args.catalog, args.schema, TREATMENT_TAXONOMY_TABLE)
    ).count()
    whitelist_count = spark.table(
        quoted_table_fqn(args.catalog, args.schema, MECHANISM_WHITELIST_TABLE)
    ).count()
    print(f"treatment_taxonomy: {taxonomy_count} rows (expected 34)")
    print(f"mechanism_whitelist: {whitelist_count} rows (expected 40)")
    assert taxonomy_count == 34, f"Expected 34, got {taxonomy_count}"
    assert whitelist_count == 40, f"Expected 40, got {whitelist_count}"


if __name__ == "__main__":
    main()
