"""Shared MLflow Deployments client factory for MAS nodes."""
from __future__ import annotations

import os

import mlflow.deployments


def _get_client():
    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE", os.environ.get("DATABRICKS_PROFILE", ""))
    target_uri = f"databricks://{profile}" if profile else "databricks"
    return mlflow.deployments.get_deploy_client(target_uri)
