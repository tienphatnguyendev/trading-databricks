# Databricks notebook source
# MAGIC %md
# MAGIC # MAS E2E Evaluation
# MAGIC Thin driver. Edit params and run.

# COMMAND ----------
# MAGIC %pip install -U mlflow>=3.8.0
# MAGIC %restart_python

# COMMAND ----------
import os

os.environ.setdefault("MLFLOW_EXPERIMENT_ID", "/Shared/mas-evaluation")

from trading_databricks.agents.eval import run_evaluation

# COMMAND ----------
# Baseline: e2e against the primary unlabeled dataset
result = run_evaluation(
    dataset_name="smart-trading-dev.04_eval.mas_eval_v1",
    dataset_version="latest",
    predict_target="e2e",
    scorers="all",
    run_kind="baseline",
)

# COMMAND ----------
result.tables["eval_results"].describe()
