# Databricks notebook source
# MAGIC %md
# MAGIC # Causality Extractor Prompt Tuning
# MAGIC Candidate-only. Merge the YAML edit on `main` to promote.

# COMMAND ----------
# MAGIC %pip install -U mlflow>=3.8.0
# MAGIC %restart_python

# COMMAND ----------
import os
from pathlib import Path

import yaml
import mlflow

os.environ.setdefault("MLFLOW_EXPERIMENT_ID", "/Shared/mas-evaluation")

from trading_databricks.agents.eval import run_evaluation

PROMPT_NAME = "causality_extractor_base"
YAML_PATH = (
    Path("src/trading_databricks/agents/prompts") / f"{PROMPT_NAME}.yaml"
)

doc = yaml.safe_load(YAML_PATH.read_text())
print(doc["messages"][0]["content"])

# COMMAND ----------
# EDIT ME: scratch prompt content
candidate_content = doc["messages"][0]["content"] + "\n\n# experimental: be more terse on standard path."
candidate_messages = [{"role": "system", "content": candidate_content}]

registered = mlflow.register_prompt(
    name=PROMPT_NAME,
    template=candidate_messages,
    commit_message="scratch tuning candidate",
)
mlflow.set_prompt_alias(name=PROMPT_NAME, alias="candidate", version=registered.version)
print(f"registered v{registered.version} as candidate")

# COMMAND ----------
result = run_evaluation(
    dataset_name="smart-trading-dev.04_eval.mas_eval_v1",
    predict_target="causality_extractor",
    scorers="all",
    run_kind="tuning",
    tags={"tuning_hypothesis": "terse-standard-path"},
)
result.tables["eval_results"].describe()
