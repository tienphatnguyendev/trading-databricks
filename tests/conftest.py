"""Shared pytest fixtures."""
from __future__ import annotations

import pytest
from databricks.connect import DatabricksSession
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """Session-scoped SparkSession backed by Databricks serverless compute.

    Uses DatabricksSession.builder.serverless() so tests run against the same
    runtime as production pipelines without requiring a classic cluster.
    Credentials are resolved from ~/.databrickscfg or environment variables.
    """
    return DatabricksSession.builder.serverless().getOrCreate()
