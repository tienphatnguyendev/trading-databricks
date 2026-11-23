"""Unified runner for MAS evaluation.

Use:
    from trading_databricks.agents.eval import run_evaluation
    run_evaluation(dataset_name="mas_eval_v1", predict_target="e2e")
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from typing import Any, Literal

PredictTarget = Literal[
    "e2e",
    "causality_extractor",
    "uncertainty_router",
    "mechanism_validator",
]
ScorerSelector = Literal["all", "deterministic", "judges", "reference"]


def _git_sha7() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip())
        return sha, dirty
    except Exception:
        return "nogit", False


def _mas_config_hash(config) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _run_name(
    *,
    run_kind: str,
    predict_target: str,
    dataset_name: str,
    dataset_version: str | int,
    git_sha7: str,
) -> str:
    return f"{run_kind}__{predict_target}__{dataset_name}@{dataset_version}__{git_sha7}"


def _build_tags(
    *,
    predict_target: str,
    dataset_name: str,
    dataset_version: str | int,
    mas_config_hash: str,
    prompt_versions: dict[str, str],
    extra: dict[str, str] | None,
    run_kind: str,
) -> dict[str, str]:
    tags = {
        "predict_target": predict_target,
        "dataset_name": dataset_name,
        "dataset_version": str(dataset_version),
        "mas_config_hash": mas_config_hash,
        "prompt_versions": json.dumps(prompt_versions, sort_keys=True),
        "run_kind": run_kind,
    }
    if extra:
        tags.update({k: str(v) for k, v in extra.items()})
    return tags


def _filter_scorers(
    *,
    requested: list,
    reference_scorers: list,
    data,
) -> list:
    """Drop reference-based scorers when the dataset has no populated expectations."""
    has_expectations = any(
        bool(row)
        for row in data["expectations"].tolist()
    )
    if has_expectations:
        return requested
    ref_names = {s.name for s in reference_scorers}
    return [s for s in requested if s.name not in ref_names]


def _build_predict_fn(predict_target: PredictTarget, config):
    """Return a ``predict_fn(**inputs)`` wrapper for the selected target."""
    from trading_databricks.agents.graph import build_mas_graph

    if predict_target == "e2e":
        graph = build_mas_graph(config=config)

        def wrapper(headline: str, body: str, symbol: str) -> dict:
            state = {"headline": headline, "body": body, "published_at": ""}
            return dict(graph.invoke(state))

        return wrapper

    if predict_target == "causality_extractor":
        from trading_databricks.agents.nodes import causality_extractor

        def wrapper(headline: str, body: str, symbol: str) -> dict:
            state = {
                "headline": headline,
                "body": body,
                "extraction_path": "standard",
                "few_shot_examples": [],
            }
            return dict(causality_extractor(state, config=config))

        return wrapper

    if predict_target == "uncertainty_router":
        from trading_databricks.agents.nodes import uncertainty_router

        def wrapper(headline: str, body: str, symbol: str) -> dict:
            state = {"headline": headline, "body": body, "few_shot_examples": []}
            return dict(uncertainty_router(state, config=config))

        return wrapper

    if predict_target == "mechanism_validator":
        from trading_databricks.agents.graph import mechanism_validator_node

        def wrapper(headline: str, body: str, symbol: str, extraction: dict | None = None) -> dict:
            state = {"headline": headline, "body": body, "extraction": extraction or {}}
            return dict(mechanism_validator_node(state, config))

        return wrapper

    raise ValueError(f"unknown predict_target: {predict_target!r}")


def run_evaluation(
    *,
    dataset_name: str,
    dataset_version: str | int = "latest",
    scorers: list | ScorerSelector = "all",
    predict_target: PredictTarget = "e2e",
    mas_config_overrides: dict | None = None,
    experiment_id: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    run_kind: str = "baseline",
    prompt_versions: dict[str, str] | None = None,
):
    """Run a MAS evaluation against an MLflow GenAI dataset.

    Returns ``mlflow.models.EvaluationResult``.
    """
    import mlflow

    from trading_databricks.agents.config import MASConfig
    from trading_databricks.agents.eval import scorers as scorer_module
    from trading_databricks.agents.eval.dataset import _resolve_corpus, build_eval_dataset

    exp_id = experiment_id or os.environ.get("MLFLOW_EXPERIMENT_ID")
    if exp_id:
        mlflow.set_experiment(experiment_id=exp_id)

    config = MASConfig(**(mas_config_overrides or {}))
    data = build_eval_dataset(corpus=_resolve_corpus(dataset_name))

    scorer_module.ensure_judges_registered(experiment_id=exp_id)

    if scorers == "all":
        resolved = list(scorer_module.ALL_SCORERS)
    elif scorers == "deterministic":
        resolved = list(scorer_module.DETERMINISTIC_SCORERS)
    elif scorers == "judges":
        resolved = list(scorer_module.JUDGE_SCORERS)
    elif scorers == "reference":
        resolved = list(scorer_module.REFERENCE_SCORERS)
    elif isinstance(scorers, list):
        resolved = scorers
    else:
        raise ValueError(f"invalid scorers arg: {scorers!r}")

    resolved = _filter_scorers(
        requested=resolved,
        reference_scorers=list(scorer_module.REFERENCE_SCORERS),
        data=data,
    )

    sha7, dirty = _git_sha7()
    built_tags = _build_tags(
        predict_target=predict_target,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        mas_config_hash=_mas_config_hash(config),
        prompt_versions=prompt_versions or {},
        extra={**(tags or {}), "git_sha": sha7, "git_dirty": str(dirty)},
        run_kind=run_kind,
    )

    final_run_name = run_name or _run_name(
        run_kind=run_kind,
        predict_target=predict_target,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        git_sha7=sha7,
    )

    predict_fn = _build_predict_fn(predict_target, config)

    with mlflow.start_run(run_name=final_run_name, tags=built_tags):
        return mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=resolved,
        )
