"""Sync YAML prompts with MLflow Prompt Registry.

Usage:
    uv run python -m trading_databricks.agents.prompts._sync push
    uv run python -m trading_databricks.agents.prompts._sync push --name judge_direction
    uv run python -m trading_databricks.agents.prompts._sync pull --name judge_direction --version 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import yaml

_PACKAGE_DIR = Path(__file__).parent


def _iter_yaml_paths(name: str | None) -> list[Path]:
    if name:
        path = _PACKAGE_DIR / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(path)
        return [path]
    return sorted(_PACKAGE_DIR.glob("*.yaml"))


def push(name: str | None, stage: str) -> None:
    for path in _iter_yaml_paths(name):
        doc = yaml.safe_load(path.read_text())
        prompt_name = doc["name"]
        messages = doc["messages"]
        registered = mlflow.register_prompt(
            name=prompt_name,
            template=messages,
            commit_message=f"sync from {path.name}",
        )
        if stage:
            mlflow.set_prompt_alias(
                name=prompt_name,
                alias=stage,
                version=registered.version,
            )
        print(f"pushed {prompt_name} v{registered.version} (alias={stage or 'none'})")


def pull(name: str, version: str | int) -> None:
    prompt = mlflow.load_prompt(f"prompts:/{name}/{version}")
    path = _PACKAGE_DIR / f"{name}.yaml"
    existing = yaml.safe_load(path.read_text()) if path.exists() else {"name": name}
    messages = [dict(m) for m in prompt.messages] if hasattr(prompt, "messages") else [
        {"role": "system", "content": prompt.template}
    ]
    existing["messages"] = messages
    path.write_text(yaml.safe_dump(existing, sort_keys=False))
    print(f"pulled {name} v{version} → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="YAML ↔ Registry sync")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_push = sub.add_parser("push")
    p_push.add_argument("--name", default=None)
    p_push.add_argument("--stage", default="candidate")

    p_pull = sub.add_parser("pull")
    p_pull.add_argument("--name", required=True)
    p_pull.add_argument("--version", required=True)

    args = parser.parse_args()
    if args.cmd == "push":
        push(args.name, args.stage)
    else:
        pull(args.name, args.version)


if __name__ == "__main__":
    main()
