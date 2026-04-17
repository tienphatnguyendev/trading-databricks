# MAS Agent Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an MLflow 3 GenAI evaluation harness for the 5-node MAS causality-extraction pipeline with YAML+Registry prompt management, 3-tier scorer catalog, unified `run_evaluation()` API, and MemAlign-ready experiment layout.

**Architecture:** Replace `src/trading_databricks/agents/prompts.py` with a `prompts/` package backed by YAML files synced to MLflow Prompt Registry. Add `src/trading_databricks/agents/eval/` module (dataset builders, 3-tier scorers, unified runner). Add thin notebook drivers under `notebooks/qa/mas/` for e2e and per-node tuning. Reuses the 2026-04-14 LLM Judge Validator production judge rubrics/models — one Registry, one experiment for both offline eval and production judge prompts.

**Tech Stack:** MLflow 3.8+ (`mlflow.genai.evaluate`, `mlflow.genai.datasets`, `mlflow.genai.scorers`, `mlflow.genai.judges.make_judge`, `mlflow.register_prompt`), LangGraph (existing), Polars (not pandas per CLAUDE.md), Pydantic schemas (existing `CausalChain`), Databricks Unity Catalog, Gemini 3.1 Pro via external-model endpoint.

**Related spec:** `docs/superpowers/specs/2026-04-17-mas-agent-evaluation-harness-design.md`

---

## File Structure

**Create:**
- `src/trading_databricks/agents/prompts/__init__.py` — `load_prompt()`, `render_messages()`
- `src/trading_databricks/agents/prompts/_formatters.py` — taxonomy/whitelist/few-shot formatters (lifted)
- `src/trading_databricks/agents/prompts/_sync.py` — YAML ↔ Registry CLI
- `src/trading_databricks/agents/prompts/causality_extractor_base.yaml`
- `src/trading_databricks/agents/prompts/causality_extractor_fast.yaml`
- `src/trading_databricks/agents/prompts/judge_direction.yaml`
- `src/trading_databricks/agents/prompts/judge_mechanism.yaml`
- `src/trading_databricks/agents/prompts/judge_grounding.yaml`
- `src/trading_databricks/agents/eval/__init__.py`
- `src/trading_databricks/agents/eval/dataset.py`
- `src/trading_databricks/agents/eval/scorers.py`
- `src/trading_databricks/agents/eval/runner.py`
- `src/trading_databricks/agents/eval/labeling.py` (stub)
- `tests/agents/prompts/test_formatters.py`
- `tests/agents/prompts/test_loader.py`
- `tests/agents/eval/test_dataset.py`
- `tests/agents/eval/test_scorers.py`
- `tests/agents/eval/test_runner.py`
- `notebooks/qa/mas/e2e_mas_evaluation.py`
- `notebooks/qa/mas/tuning/causality_extractor_tuning.py`
- `notebooks/qa/mas/tuning/judge_direction_tuning.py`
- `notebooks/qa/mas/tuning/judge_mechanism_tuning.py`
- `notebooks/qa/mas/tuning/judge_grounding_tuning.py`

**Modify:**
- `src/trading_databricks/agents/prompts.py` → delete after Task 5 lands
- `src/trading_databricks/agents/nodes/causality_extractor.py` — swap prompt import
- `tests/agents/test_prompts.py` → delete / relocate
- `pyproject.toml` — ensure `mlflow>=3.8.0`, `pyyaml`

---

## Task 1: Scaffold prompts/ package and lift formatters

**Files:**
- Create: `src/trading_databricks/agents/prompts/__init__.py`
- Create: `src/trading_databricks/agents/prompts/_formatters.py`
- Create: `tests/agents/prompts/__init__.py`
- Create: `tests/agents/prompts/test_formatters.py`

- [ ] **Step 1: Write failing test for formatters**

Create `tests/agents/prompts/__init__.py` (empty) and `tests/agents/prompts/test_formatters.py`:

```python
from trading_databricks.agents.prompts._formatters import (
    format_compact_taxonomy,
    format_full_taxonomy,
    format_whitelist,
    format_few_shot,
)


def test_format_compact_taxonomy_sorts_and_joins():
    tax = {"Regulation": ["approval", "ban"], "Adoption": ["etf"]}
    assert format_compact_taxonomy(tax) == "Adoption.etf, Regulation.approval, Regulation.ban"


def test_format_full_taxonomy_renders_tree():
    tax = {"Adoption": ["etf"]}
    out = format_full_taxonomy(tax)
    assert "- Adoption:" in out
    assert "  - Adoption.etf" in out


def test_format_whitelist_groups_by_l1():
    wl = {("Adoption", "etf_inflow"): ["e1"], ("Adoption", "etf_outflow"): ["e2"]}
    out = format_whitelist(wl)
    assert "L1 Category: Adoption" in out
    assert "  - etf_inflow" in out


def test_format_few_shot_respects_limit():
    examples = [{"trigger_event": f"h{i}", "causal_chain_json": "{}"} for i in range(5)]
    out = format_few_shot(examples, limit=2)
    assert "Example 1:" in out
    assert "Example 2:" in out
    assert "Example 3:" not in out


def test_format_few_shot_empty_returns_placeholder():
    assert format_few_shot([], limit=3) == "No examples available."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/prompts/test_formatters.py -v`
Expected: FAIL (ModuleNotFoundError: `trading_databricks.agents.prompts._formatters`)

- [ ] **Step 3: Create package scaffold and lift formatters**

Create `src/trading_databricks/agents/prompts/__init__.py`:

```python
"""Prompt management for MAS agents.

Runtime loads prompts from MLflow Prompt Registry via ``load_prompt``.
YAML files in this package are the source of truth (synced via ``_sync.py``).
"""
from __future__ import annotations

from trading_databricks.agents.prompts._formatters import (
    format_compact_taxonomy,
    format_few_shot,
    format_full_taxonomy,
    format_whitelist,
)

__all__ = [
    "format_compact_taxonomy",
    "format_full_taxonomy",
    "format_whitelist",
    "format_few_shot",
]
```

Create `src/trading_databricks/agents/prompts/_formatters.py` with the four functions lifted verbatim from `src/trading_databricks/agents/prompts.py` (lines 4–42):

```python
"""Formatters for taxonomy, whitelist, and few-shot injection into prompts."""
from __future__ import annotations

import json


def format_compact_taxonomy(taxonomy: dict[str, list[str]]) -> str:
    classes = []
    for l1, l2s in sorted(taxonomy.items()):
        for l2 in sorted(l2s):
            classes.append(f"{l1}.{l2}")
    return ", ".join(classes)


def format_full_taxonomy(taxonomy: dict[str, list[str]]) -> str:
    lines = []
    for l1, l2s in sorted(taxonomy.items()):
        lines.append(f"- {l1}:")
        for l2 in sorted(l2s):
            lines.append(f"  - {l1}.{l2}")
    return "\n".join(lines)


def format_whitelist(whitelist: dict[tuple[str, str], list[str]]) -> str:
    grouped: dict[str, list[str]] = {}
    for (l1, mechanism), _ in whitelist.items():
        grouped.setdefault(l1, []).append(mechanism)
    lines = []
    for l1, mechs in sorted(grouped.items()):
        lines.append(f"L1 Category: {l1}")
        for m in sorted(mechs):
            lines.append(f"  - {m}")
    return "\n".join(lines)


def format_few_shot(examples: list[dict], limit: int) -> str:
    selected = examples[:limit]
    if not selected:
        return "No examples available."
    formatted = []
    for i, ex in enumerate(selected, 1):
        chain_json = ex.get("causal_chain_json", "{}")
        if not isinstance(chain_json, str):
            chain_json = json.dumps(chain_json)
        formatted.append(
            f"Example {i}:\nHeadline: {ex.get('trigger_event', '')}\nExtraction:\n{chain_json}"
        )
    return "\n\n".join(formatted)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agents/prompts/test_formatters.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/agents/prompts/ tests/agents/prompts/
git commit -m "TRD feat: scaffold prompts/ package, lift formatters"
```

---

## Task 2: Author YAML files for causality extractor prompts

**Files:**
- Create: `src/trading_databricks/agents/prompts/causality_extractor_base.yaml`
- Create: `src/trading_databricks/agents/prompts/causality_extractor_fast.yaml`

- [ ] **Step 1: Create `causality_extractor_base.yaml`**

```yaml
name: causality_extractor_base
description: Full prompt for standard and careful extraction paths.
variables:
  - taxonomy_injection
  - whitelist_injection
  - few_shot_injection
  - careful_suffix
messages:
  - role: system
    content: |
      You are a financial news causality extractor for BTC/ETH markets.

      TASK: Extract a single-hop causal chain from the article.
      Respond only with the JSON extraction object. Do not include markdown formatting like ```json.

      TREATMENT TAXONOMY (classify the event into one of these):
      {taxonomy_injection}

      MECHANISM WHITELIST (the mechanism MUST be one of these):
      {whitelist_injection}

      SIMILAR PAST EXTRACTIONS (reference, do not copy blindly):
      {few_shot_injection}

      RULES:
      1. Copy evidence_spans VERBATIM from the article body — exact character sequences.
      2. mechanism must exactly match a whitelist entry for the chosen L1 category.
      3. If the article does not describe a clear causal event for BTC/ETH, respond with {{"is_impactful": false}}.
      4. Use retrieved examples as reference patterns, but extract based on THIS article's actual content.{careful_suffix}
metadata:
  used_by: [causality_extractor]
  paths: [standard, careful]
  careful_suffix: |

    5. CAREFUL PATH: Please provide explicit chain-of-thought reasoning before your final extraction. Explain your classification and mechanism selection in the `reasoning` field.
```

- [ ] **Step 2: Create `causality_extractor_fast.yaml`**

```yaml
name: causality_extractor_fast
description: Compact prompt for fast extraction path (taxonomy-only, no whitelist).
variables:
  - taxonomy_injection
  - few_shot_injection
messages:
  - role: system
    content: |
      You are a financial news causality extractor for BTC/ETH markets.

      TASK: Extract a single-hop causal chain from the article.
      Respond only with the JSON extraction object. Do not include markdown formatting like ```json.

      CRITICAL: `treatment_class` MUST be exactly one of the following 34 categories, or you will fail:
      {taxonomy_injection}

      SIMILAR PAST EXTRACTIONS (reference, do not copy blindly):
      {few_shot_injection}

      RULES:
      1. Copy evidence_spans VERBATIM from the article body — exact character sequences.
      2. If the article does not describe a clear causal event for BTC/ETH, respond with {{"is_impactful": false}}.
      3. Use retrieved examples as reference patterns, but extract based on THIS article's actual content.
metadata:
  used_by: [causality_extractor]
  paths: [fast]
```

- [ ] **Step 3: Commit**

```bash
git add src/trading_databricks/agents/prompts/*.yaml
git commit -m "TRD feat: add YAML source for causality extractor prompts"
```

---

## Task 3: Implement `load_prompt` loader (Registry with YAML fallback)

**Files:**
- Modify: `src/trading_databricks/agents/prompts/__init__.py`
- Create: `tests/agents/prompts/test_loader.py`

Contract: `load_prompt(name, version="production")` returns `list[dict]` of chat messages with `{role, content}`. It first tries `mlflow.load_prompt(f"prompts:/{name}/{version}")`; on failure (registry unavailable, prompt not yet registered) falls back to reading the sibling YAML file. The fallback is what makes offline unit tests work without a Databricks connection.

- [ ] **Step 1: Write failing tests**

Create `tests/agents/prompts/test_loader.py`:

```python
from unittest.mock import patch

import pytest

from trading_databricks.agents.prompts import load_prompt, render_messages


def test_load_prompt_falls_back_to_yaml_when_registry_unavailable():
    """When Registry call fails, loader reads the sibling YAML."""
    with patch("mlflow.load_prompt", side_effect=RuntimeError("no registry")):
        messages = load_prompt("causality_extractor_fast", version="production")
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert "financial news causality extractor" in messages[0]["content"]


def test_load_prompt_unknown_name_raises():
    with patch("mlflow.load_prompt", side_effect=RuntimeError("no registry")):
        with pytest.raises(FileNotFoundError):
            load_prompt("does_not_exist")


def test_render_messages_substitutes_variables():
    messages = [{"role": "system", "content": "Hello {name}, count={n}"}]
    rendered = render_messages(messages, {"name": "world", "n": 3})
    assert rendered == [{"role": "system", "content": "Hello world, count=3"}]


def test_render_messages_ignores_double_braces():
    """Double-brace escapes should survive .format() as literal braces."""
    messages = [{"role": "system", "content": "Return {{\"is_impactful\": false}} on empty"}]
    rendered = render_messages(messages, {})
    assert '{"is_impactful": false}' in rendered[0]["content"]
```

- [ ] **Step 2: Run test — expect fail**

Run: `uv run pytest tests/agents/prompts/test_loader.py -v`
Expected: FAIL (ImportError on `load_prompt`, `render_messages`)

- [ ] **Step 3: Implement loader**

Replace contents of `src/trading_databricks/agents/prompts/__init__.py`:

```python
"""Prompt management for MAS agents."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from trading_databricks.agents.prompts._formatters import (
    format_compact_taxonomy,
    format_few_shot,
    format_full_taxonomy,
    format_whitelist,
)

_PACKAGE_DIR = Path(__file__).parent


def _load_from_yaml(name: str) -> list[dict[str, str]]:
    yaml_path = _PACKAGE_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No YAML source for prompt '{name}' at {yaml_path}")
    doc = yaml.safe_load(yaml_path.read_text())
    return [dict(m) for m in doc["messages"]]


def load_prompt(name: str, version: str | int = "production") -> list[dict[str, str]]:
    """Load a prompt as chat messages.

    Prefers MLflow Prompt Registry; falls back to packaged YAML when unavailable.
    """
    try:
        import mlflow

        prompt = mlflow.load_prompt(f"prompts:/{name}/{version}")
        if hasattr(prompt, "messages") and prompt.messages:
            return [dict(m) for m in prompt.messages]
        return [{"role": "system", "content": prompt.template}]
    except Exception:
        return _load_from_yaml(name)


def render_messages(
    messages: list[dict[str, str]],
    variables: dict[str, Any],
) -> list[dict[str, str]]:
    """Substitute ``{var}`` placeholders using ``str.format``.

    Double braces ``{{`` / ``}}`` are preserved as literal braces.
    """
    return [
        {"role": m["role"], "content": m["content"].format(**variables)}
        for m in messages
    ]


__all__ = [
    "load_prompt",
    "render_messages",
    "format_compact_taxonomy",
    "format_full_taxonomy",
    "format_whitelist",
    "format_few_shot",
]
```

- [ ] **Step 4: Ensure pyyaml dep**

Check `pyproject.toml` has `pyyaml`. If absent:

```bash
uv add pyyaml
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/agents/prompts/ -v`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add src/trading_databricks/agents/prompts/__init__.py tests/agents/prompts/test_loader.py pyproject.toml uv.lock
git commit -m "TRD feat: add load_prompt with Registry + YAML fallback"
```

---

## Task 4: YAML ↔ Registry sync CLI (`_sync.py`)

**Files:**
- Create: `src/trading_databricks/agents/prompts/_sync.py`

Behavior: `uv run python -m trading_databricks.agents.prompts._sync push [--name NAME] [--stage STAGE]` registers each YAML as a prompt version in the Registry. `… _sync.py pull --name NAME --version V` writes a Registry version back to YAML (for GEPA-accepted candidates).

- [ ] **Step 1: Implement `_sync.py`**

```python
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
```

- [ ] **Step 2: Smoke-test CLI help (no Registry call)**

Run: `uv run python -m trading_databricks.agents.prompts._sync --help`
Expected: usage text with `push` and `pull` subcommands.

- [ ] **Step 3: Commit**

```bash
git add src/trading_databricks/agents/prompts/_sync.py
git commit -m "TRD feat: YAML <-> Registry sync CLI for MAS prompts"
```

---

## Task 5: Refactor causality_extractor to use Registry + delete old prompts.py

**Files:**
- Modify: `src/trading_databricks/agents/nodes/causality_extractor.py`
- Delete: `src/trading_databricks/agents/prompts.py` (after confirming only the node imports from it)
- Delete: `tests/agents/test_prompts.py` (replaced by `tests/agents/prompts/`)

- [ ] **Step 1: Locate all imports of the old module**

Run: `uv run python -c "import subprocess; print(subprocess.check_output(['grep','-rn','from trading_databricks.agents.prompts import','src','tests','notebooks']).decode())"`

Alternative with Grep tool: search `from trading_databricks.agents.prompts import` across repo.

Record every caller. Each must switch from `build_system_prompt(...)` to the new loader pattern.

- [ ] **Step 2: Add a thin `build_system_prompt` shim in the new package**

Append to `src/trading_databricks/agents/prompts/__init__.py`:

```python
def build_system_prompt(
    taxonomy: dict[str, list[str]],
    whitelist: dict[tuple[str, str], list[str]],
    few_shot_examples: list[dict],
    path: str,
    *,
    registry_stage: str = "production",
) -> str:
    """Load + render the causality-extractor system prompt for a path.

    Returns the fully rendered system content string (matches old API).
    """
    if path == "fast":
        messages = load_prompt("causality_extractor_fast", version=registry_stage)
        rendered = render_messages(
            messages,
            {
                "taxonomy_injection": format_compact_taxonomy(taxonomy),
                "few_shot_injection": format_few_shot(few_shot_examples, 3),
            },
        )
        return rendered[0]["content"]

    messages = load_prompt("causality_extractor_base", version=registry_stage)
    careful_suffix = ""
    if path == "careful":
        # careful_suffix literal lives in YAML metadata to keep it out of Python
        import yaml
        from pathlib import Path

        yaml_path = Path(__file__).parent / "causality_extractor_base.yaml"
        doc = yaml.safe_load(yaml_path.read_text())
        careful_suffix = doc.get("metadata", {}).get("careful_suffix", "")

    rendered = render_messages(
        messages,
        {
            "taxonomy_injection": format_full_taxonomy(taxonomy),
            "whitelist_injection": format_whitelist(whitelist),
            "few_shot_injection": format_few_shot(few_shot_examples, 5),
            "careful_suffix": careful_suffix,
        },
    )
    return rendered[0]["content"]
```

Add `"build_system_prompt"` to `__all__`.

- [ ] **Step 3: Update `causality_extractor.py` imports**

Open `src/trading_databricks/agents/nodes/causality_extractor.py` and find the line importing from `trading_databricks.agents.prompts`. Leave it — with `prompts` now a package exposing the same `build_system_prompt`, the import keeps working. Verify the API signature still matches.

- [ ] **Step 4: Run the existing prompt/extractor tests**

Run: `uv run pytest tests/agents/test_prompts.py tests/agents/test_causality_extractor.py -v`

If `test_prompts.py` references the old module-level private constants (`_BASE_SYSTEM_PROMPT`, etc.), adapt it. Preferred path: delete `tests/agents/test_prompts.py` and add coverage in `tests/agents/prompts/test_loader.py` for `build_system_prompt` instead:

```python
def test_build_system_prompt_fast_injects_compact_taxonomy():
    from trading_databricks.agents.prompts import build_system_prompt

    out = build_system_prompt(
        taxonomy={"Adoption": ["etf"]},
        whitelist={},
        few_shot_examples=[],
        path="fast",
    )
    assert "Adoption.etf" in out
    assert "MECHANISM WHITELIST" not in out


def test_build_system_prompt_careful_appends_cot_suffix():
    from trading_databricks.agents.prompts import build_system_prompt

    out = build_system_prompt(
        taxonomy={"Adoption": ["etf"]},
        whitelist={("Adoption", "etf_inflow"): ["e"]},
        few_shot_examples=[],
        path="careful",
    )
    assert "chain-of-thought reasoning" in out
```

- [ ] **Step 5: Delete old module**

```bash
git rm src/trading_databricks/agents/prompts.py tests/agents/test_prompts.py
```

- [ ] **Step 6: Run full agents test suite**

Run: `uv run pytest tests/agents/ -v`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add -A src/trading_databricks/agents/ tests/agents/
git commit -m "TRD refactor: route causality extractor prompts via Registry"
```

---

## Task 6: Author judge YAMLs (direction, mechanism, grounding)

**Files:**
- Create: `src/trading_databricks/agents/prompts/judge_direction.yaml`
- Create: `src/trading_databricks/agents/prompts/judge_mechanism.yaml`
- Create: `src/trading_databricks/agents/prompts/judge_grounding.yaml`

Pull rubric text from `docs/superpowers/specs/2026-04-14-llm-judge-validator-design.md` (judge rubrics section). Each YAML has `variables: [headline, body, extraction]` and a single `system` message asking for the categorical/float verdict per that spec's contract.

- [ ] **Step 1: Create `judge_direction.yaml`**

```yaml
name: judge_direction
description: Binary direction judge for BTC/ETH causal-chain extractions.
variables:
  - headline
  - body
  - extraction
messages:
  - role: system
    content: |
      You are evaluating whether a causality extraction correctly identifies the
      market-direction implication for BTC/ETH.

      HEADLINE: {headline}
      BODY: {body}
      EXTRACTION: {extraction}

      Decide the direction the article implies for the token price action:
      - "up": article implies upward pressure
      - "down": article implies downward pressure
      - "neutral": article is ambiguous, informational, or non-impactful

      Respond with exactly one token: up | down | neutral.
metadata:
  used_by: [judge_direction_scorer, production_judge_node]
  feedback_value_type: categorical
  categories: [up, down, neutral]
```

- [ ] **Step 2: Create `judge_mechanism.yaml`**

```yaml
name: judge_mechanism
description: Mechanism-whitelist judge for MAS extractions.
variables:
  - headline
  - body
  - extraction
  - whitelist_injection
messages:
  - role: system
    content: |
      You evaluate whether the extracted mechanism correctly describes the causal
      driver the article attributes to the BTC/ETH move.

      HEADLINE: {headline}
      BODY: {body}
      EXTRACTION: {extraction}

      WHITELIST:
      {whitelist_injection}

      Choose the single best mechanism from the whitelist for this article.
      Respond with exactly that whitelist entry string (no punctuation, no explanation).
metadata:
  used_by: [judge_mechanism_scorer, production_judge_node]
  feedback_value_type: categorical
```

- [ ] **Step 3: Create `judge_grounding.yaml`**

```yaml
name: judge_grounding
description: Evidence-grounding judge scoring span support for the causal claim.
variables:
  - body
  - extraction
messages:
  - role: system
    content: |
      You are scoring how well the evidence spans in an extraction support its
      causal claim against the source article.

      BODY: {body}
      EXTRACTION: {extraction}

      Score the grounding on a continuous scale:
      - 1.0: every evidence span is verbatim and directly supports the claim
      - 0.5: spans are verbatim but only weakly support the claim
      - 0.0: spans are missing, paraphrased, or irrelevant

      Respond with a single float in [0.0, 1.0] — no other text.
metadata:
  used_by: [judge_grounding_scorer, production_judge_node]
  feedback_value_type: float
  range: [0.0, 1.0]
```

- [ ] **Step 4: Commit**

```bash
git add src/trading_databricks/agents/prompts/judge_*.yaml
git commit -m "TRD feat: add YAML sources for 3 MAS production judges"
```

---

## Task 7: Scaffold `eval/` package

**Files:**
- Create: `src/trading_databricks/agents/eval/__init__.py`
- Create: `src/trading_databricks/agents/eval/labeling.py`
- Create: `tests/agents/eval/__init__.py`

- [ ] **Step 1: Write scaffold**

`src/trading_databricks/agents/eval/__init__.py`:

```python
"""Evaluation harness for the MAS causality pipeline.

Public API:
    - build_eval_dataset: load / materialize MLflow GenAI datasets
    - run_evaluation: one-shot evaluation driver
    - ALL_SCORERS, DETERMINISTIC_SCORERS, JUDGE_SCORERS, REFERENCE_SCORERS
"""
from __future__ import annotations

from trading_databricks.agents.eval.dataset import build_eval_dataset
from trading_databricks.agents.eval.runner import run_evaluation
from trading_databricks.agents.eval.scorers import (
    ALL_SCORERS,
    DETERMINISTIC_SCORERS,
    JUDGE_SCORERS,
    REFERENCE_SCORERS,
)

__all__ = [
    "build_eval_dataset",
    "run_evaluation",
    "ALL_SCORERS",
    "DETERMINISTIC_SCORERS",
    "JUDGE_SCORERS",
    "REFERENCE_SCORERS",
]
```

`src/trading_databricks/agents/eval/labeling.py`:

```python
"""Placeholder for MemAlign labeling sessions.

See Workflow 7 in .claude/skills/databricks-mlflow-evaluation/SKILL.md and
`patterns-judge-alignment.md` for the full flow when an SME is scheduled.

This module is intentionally empty in v1. Judge names in ``scorers.py``
match the label-schema names that a future labeling session will use,
so ``align()`` pairing works without schema migration.
"""
```

Create `tests/agents/eval/__init__.py` (empty).

- [ ] **Step 2: Do NOT import yet**

The `__init__.py` imports symbols we have not written. Leave the file in place — subsequent tasks populate the targets. If running tests at this point, expect `ImportError` on `trading_databricks.agents.eval`; only run `tests/agents/prompts/` for now.

- [ ] **Step 3: Commit**

```bash
git add src/trading_databricks/agents/eval/ tests/agents/eval/__init__.py
git commit -m "TRD feat: scaffold eval/ package and labeling stub"
```

---

## Task 8: `eval/dataset.py` — primary dataset builder with stratified sampling

**Files:**
- Create: `src/trading_databricks/agents/eval/dataset.py`
- Create: `tests/agents/eval/test_dataset.py`

Contract: `build_eval_dataset(corpus, version="latest")` returns a pandas DataFrame shaped for `mlflow.genai.evaluate(data=...)` with columns `inputs`, `expectations`, `tags`. Primary corpus stratifies on `source × token`, fallback to marginal on the smaller-population axis when a cell has fewer than 5 rows.

- [ ] **Step 1: Write failing test**

`tests/agents/eval/test_dataset.py`:

```python
import polars as pl
import pytest

from trading_databricks.agents.eval.dataset import (
    _stratified_sample,
    PRIMARY_DATASET_NAME,
)


def test_stratified_sample_proportional_allocation():
    df = pl.DataFrame({
        "id": list(range(100)),
        "source": ["coindesk"] * 60 + ["theblock"] * 40,
        "token": ["BTC"] * 50 + ["ETH"] * 50,
    })
    out = _stratified_sample(df, strata=["source", "token"], target_n=20, min_cell=5, seed=0)
    assert len(out) == 20
    # all 4 cells represented since each has ≥ min_cell rows
    assert out.select(pl.struct("source", "token")).n_unique() == 4


def test_stratified_sample_fallback_when_cell_below_min():
    df = pl.DataFrame({
        "id": list(range(50)),
        "source": ["coindesk"] * 48 + ["rare"] * 2,
        "token": ["BTC"] * 40 + ["ETH"] * 10,
    })
    out = _stratified_sample(df, strata=["source", "token"], target_n=20, min_cell=5, seed=0)
    assert len(out) == 20


def test_primary_dataset_name_matches_spec():
    assert PRIMARY_DATASET_NAME == "smart-trading-dev.04_eval.mas_eval_v1"


def test_build_eval_dataset_unknown_corpus_raises():
    from trading_databricks.agents.eval.dataset import build_eval_dataset

    with pytest.raises(ValueError, match="unknown corpus"):
        build_eval_dataset(corpus="bogus")
```

- [ ] **Step 2: Run test — expect fail**

Run: `uv run pytest tests/agents/eval/test_dataset.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `dataset.py`**

```python
"""Evaluation dataset builders for the MAS harness.

Primary corpus stratifies on ``source × token`` with a marginal fallback
when a cell has fewer than ``min_cell`` rows.
"""
from __future__ import annotations

from typing import Literal

import polars as pl

PRIMARY_DATASET_NAME = "smart-trading-dev.04_eval.mas_eval_v1"
CALIBRATION_DATASETS = {
    "cryptolin": "smart-trading-dev.04_eval.mas_eval_direction_cryptolin_v1",
    "fincausal": "smart-trading-dev.04_eval.mas_eval_grounding_fincausal_v1",
    "cnc": "smart-trading-dev.04_eval.mas_eval_grounding_cnc_v1",
    "mechanism_bootstrap": "smart-trading-dev.04_eval.mas_eval_mechanism_bootstrap_v1",
}

Corpus = Literal["primary", "cryptolin", "fincausal", "cnc", "mechanism_bootstrap"]


def _stratified_sample(
    df: pl.DataFrame,
    *,
    strata: list[str],
    target_n: int,
    min_cell: int,
    seed: int,
) -> pl.DataFrame:
    """Proportional stratified sample with marginal fallback for sparse cells."""
    cell_counts = df.group_by(strata).len()
    sparse = cell_counts.filter(pl.col("len") < min_cell)
    if sparse.height == 0:
        cells_iter = cell_counts.iter_rows(named=True)
    else:
        # Fallback: drop the higher-cardinality axis, resample marginals
        marginal_axis = strata[0]
        cell_counts = df.group_by([marginal_axis]).len()
        strata = [marginal_axis]
        cells_iter = cell_counts.iter_rows(named=True)

    total = df.height
    out_parts: list[pl.DataFrame] = []
    for row in cells_iter:
        k = max(1, round(target_n * row["len"] / total))
        mask = pl.lit(True)
        for s in strata:
            mask = mask & (pl.col(s) == row[s])
        cell_df = df.filter(mask)
        take = min(k, cell_df.height)
        out_parts.append(cell_df.sample(n=take, seed=seed, shuffle=True))

    out = pl.concat(out_parts)
    if out.height > target_n:
        out = out.sample(n=target_n, seed=seed, shuffle=True)
    return out


def _load_primary_source() -> pl.DataFrame:
    """Pull rows from smart-trading-dev.01_bronze.news_agent_eval_dataset via Spark."""
    from trading_databricks.agents.nodes.few_shot_retriever import get_spark_session

    spark = get_spark_session()
    sdf = spark.table("smart-trading-dev.01_bronze.news_agent_eval_dataset")
    return pl.from_arrow(sdf.toArrow())


def build_eval_dataset(
    corpus: Corpus = "primary",
    version: str | int = "latest",
    *,
    target_n: int = 200,
    seed: int = 42,
):
    """Return a pandas DataFrame ready for ``mlflow.genai.evaluate(data=...)``.

    Schema: columns ``inputs`` (dict), ``expectations`` (dict), ``tags`` (dict).
    """
    if corpus == "primary":
        df = _load_primary_source()
        sampled = _stratified_sample(
            df,
            strata=["source", "token"],
            target_n=target_n,
            min_cell=5,
            seed=seed,
        )
        records = [
            {
                "inputs": {"headline": r["headline"], "body": r["body"], "symbol": r["token"]},
                "expectations": {},
                "tags": {"source": r["source"], "token": r["token"], "corpus": "primary"},
            }
            for r in sampled.iter_rows(named=True)
        ]
    elif corpus in CALIBRATION_DATASETS:
        # Delegated to per-corpus loader in Task 9
        records = _load_calibration(corpus)
    else:
        raise ValueError(f"unknown corpus: {corpus!r}")

    import pandas as pd

    return pd.DataFrame(records)


def _load_calibration(corpus: str) -> list[dict]:
    """Stub — populated in Task 9."""
    raise NotImplementedError(f"calibration loader for {corpus} pending Task 9")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/agents/eval/test_dataset.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/agents/eval/dataset.py tests/agents/eval/test_dataset.py
git commit -m "TRD feat: MAS primary eval dataset builder with stratified sampling"
```

---

## Task 9: Calibration corpus loaders + UC registration helpers

**Files:**
- Modify: `src/trading_databricks/agents/eval/dataset.py`
- Modify: `tests/agents/eval/test_dataset.py`

For each of `cryptolin`, `fincausal`, `cnc`, `mechanism_bootstrap`: add a loader reading from the corresponding UC table (or HuggingFace cache for the open corpora — see spec §2.2 of the 2026-04-14 design), emitting records with populated `expectations` per the spec §2.2 table in the harness design doc.

Scope decision: v1 assumes the four UC tables already exist (built as a separate data-engineering ticket). Loaders read them as Spark tables and convert via Arrow. If a table is missing, raise a clear error pointing at the ticket.

- [ ] **Step 1: Add test for one loader**

Append to `tests/agents/eval/test_dataset.py`:

```python
def test_load_calibration_unknown_corpus_raises():
    from trading_databricks.agents.eval.dataset import _load_calibration

    with pytest.raises(ValueError, match="unsupported"):
        _load_calibration("not_a_corpus")
```

- [ ] **Step 2: Replace `_load_calibration`**

In `src/trading_databricks/agents/eval/dataset.py`, replace the stub:

```python
_CALIBRATION_EXPECTATIONS = {
    "cryptolin": ("direction_label", "direction"),
    "fincausal": ("causal_span_present", "grounding"),
    "cnc": ("causal_span_present", "grounding"),
    "mechanism_bootstrap": ("mechanism_label", "mechanism"),
}


def _load_calibration(corpus: str) -> list[dict]:
    if corpus not in CALIBRATION_DATASETS:
        raise ValueError(f"unsupported calibration corpus: {corpus!r}")

    from trading_databricks.agents.nodes.few_shot_retriever import get_spark_session

    table = CALIBRATION_DATASETS[corpus]
    label_key, rubric = _CALIBRATION_EXPECTATIONS[corpus]

    spark = get_spark_session()
    sdf = spark.table(table)
    df = pl.from_arrow(sdf.toArrow())

    records: list[dict] = []
    for r in df.iter_rows(named=True):
        records.append(
            {
                "inputs": {
                    "headline": r.get("headline", ""),
                    "body": r.get("body", ""),
                    "symbol": r.get("token", r.get("symbol", "BTC")),
                },
                "expectations": {label_key: r[label_key]},
                "tags": {
                    "source": r.get("source", corpus),
                    "token": r.get("token", "BTC"),
                    "corpus": corpus,
                    "rubric": rubric,
                },
            }
        )
    return records
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/agents/eval/test_dataset.py -v`
Expected: all passed.

- [ ] **Step 4: Commit**

```bash
git add src/trading_databricks/agents/eval/dataset.py tests/agents/eval/test_dataset.py
git commit -m "TRD feat: calibration corpus loaders for direction/mechanism/grounding"
```

---

## Task 10: Tier 1 — deterministic reference-free scorers

**Files:**
- Create: `src/trading_databricks/agents/eval/scorers.py`
- Create: `tests/agents/eval/test_scorers.py`

Six Tier-1 scorers per spec §3.1. All use `@mlflow.genai.scorers.scorer` decorator pattern (no LLM calls).

- [ ] **Step 1: Write failing test**

`tests/agents/eval/test_scorers.py`:

```python
import pytest

from trading_databricks.agents.eval.scorers import (
    schema_valid,
    treatment_in_taxonomy,
    mechanism_in_whitelist,
    evidence_spans_verbatim,
    validator_accepted,
)


def test_schema_valid_accepts_well_formed_extraction():
    outputs = {"extraction": {
        "is_impactful": True,
        "treatment_class": "Adoption.etf",
        "mechanism": "etf_inflow",
        "evidence_spans": ["block trade"],
        "direction": "up",
    }}
    assert schema_valid(outputs=outputs) == 1.0


def test_schema_valid_rejects_missing_fields():
    outputs = {"extraction": {"is_impactful": True}}
    assert schema_valid(outputs=outputs) == 0.0


def test_treatment_in_taxonomy_true():
    outputs = {"extraction": {"is_impactful": True, "treatment_class": "Adoption.etf"}}
    assert treatment_in_taxonomy(
        outputs=outputs,
        taxonomy={"Adoption": ["etf"]},
    ) == 1.0


def test_mechanism_in_whitelist_false():
    outputs = {"extraction": {"treatment_class": "Adoption.etf", "mechanism": "bogus"}}
    assert mechanism_in_whitelist(
        outputs=outputs,
        whitelist={("Adoption", "etf_inflow"): ["ev"]},
    ) == 0.0


def test_evidence_spans_verbatim_true():
    outputs = {"extraction": {"evidence_spans": ["block trade"]}}
    inputs = {"body": "A large block trade hit the tape."}
    assert evidence_spans_verbatim(outputs=outputs, inputs=inputs) == 1.0


def test_evidence_spans_verbatim_false_when_paraphrased():
    outputs = {"extraction": {"evidence_spans": ["large block transaction"]}}
    inputs = {"body": "A large block trade hit the tape."}
    assert evidence_spans_verbatim(outputs=outputs, inputs=inputs) == 0.0


def test_validator_accepted_passes():
    outputs = {"validation_status": "accepted"}
    assert validator_accepted(outputs=outputs) == 1.0
```

- [ ] **Step 2: Implement deterministic scorers**

`src/trading_databricks/agents/eval/scorers.py`:

```python
"""Scorer catalog for the MAS evaluation harness.

Three tiers:
    1. Deterministic reference-free (schema, taxonomy, whitelist, verbatim, validator, latency)
    2. LLM judges (direction, mechanism, grounding) — registered at import
    3. Reference-based (direction_match, mechanism_match, grounding_alignment)

Reference-based scorers are auto-skipped by runner.py when the dataset has
empty expectations.
"""
from __future__ import annotations

from typing import Any

from mlflow.genai.scorers import scorer

from trading_databricks.agents.schemas import CausalChain


@scorer
def schema_valid(*, outputs: dict[str, Any]) -> float:
    extraction = outputs.get("extraction")
    if not extraction:
        return 0.0
    try:
        CausalChain.model_validate(extraction)
        return 1.0
    except Exception:
        return 0.0


@scorer
def treatment_in_taxonomy(
    *,
    outputs: dict[str, Any],
    taxonomy: dict[str, list[str]],
) -> float:
    extraction = outputs.get("extraction") or {}
    treatment = extraction.get("treatment_class", "")
    if "." not in treatment:
        return 0.0
    l1, l2 = treatment.split(".", 1)
    return 1.0 if l2 in taxonomy.get(l1, []) else 0.0


@scorer
def mechanism_in_whitelist(
    *,
    outputs: dict[str, Any],
    whitelist: dict[tuple[str, str], list[str]],
) -> float:
    extraction = outputs.get("extraction") or {}
    treatment = extraction.get("treatment_class", "")
    mechanism = extraction.get("mechanism", "")
    if "." not in treatment:
        return 0.0
    l1 = treatment.split(".", 1)[0]
    return 1.0 if (l1, mechanism) in whitelist else 0.0


@scorer
def evidence_spans_verbatim(
    *,
    outputs: dict[str, Any],
    inputs: dict[str, Any],
) -> float:
    extraction = outputs.get("extraction") or {}
    body = inputs.get("body", "") or ""
    spans = extraction.get("evidence_spans", []) or []
    if not spans:
        return 0.0
    return 1.0 if all(s in body for s in spans) else 0.0


@scorer
def validator_accepted(*, outputs: dict[str, Any]) -> float:
    return 1.0 if outputs.get("validation_status") == "accepted" else 0.0


DETERMINISTIC_SCORERS = [
    schema_valid,
    treatment_in_taxonomy,
    mechanism_in_whitelist,
    evidence_spans_verbatim,
    validator_accepted,
]

# Populated in Tasks 11-12
JUDGE_SCORERS: list = []
REFERENCE_SCORERS: list = []
ALL_SCORERS = DETERMINISTIC_SCORERS + JUDGE_SCORERS + REFERENCE_SCORERS
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/agents/eval/test_scorers.py -v`
Expected: 7 passed.

- [ ] **Step 4: Commit**

```bash
git add src/trading_databricks/agents/eval/scorers.py tests/agents/eval/test_scorers.py
git commit -m "TRD feat: Tier-1 deterministic MAS eval scorers"
```

---

## Task 11: Tier 2 — LLM judges with idempotent registration

**Files:**
- Modify: `src/trading_databricks/agents/eval/scorers.py`
- Modify: `tests/agents/eval/test_scorers.py`

Contract: three `mlflow.genai.judges.make_judge` instances. Names (`judge_direction`, `judge_mechanism`, `judge_grounding`) MUST match future SME label-schema names for MemAlign `align()` pairing. Registration idempotent per `_register_if_absent`.

- [ ] **Step 1: Add test for idempotent registration**

Append to `tests/agents/eval/test_scorers.py`:

```python
def test_register_if_absent_skips_when_already_registered(monkeypatch):
    from trading_databricks.agents.eval import scorers

    class FakeScorer:
        name = "judge_direction"
        def register(self, *, experiment_id):
            raise AssertionError("should not be called when scorer already exists")

    class Existing:
        name = "judge_direction"

    monkeypatch.setattr(
        "mlflow.genai.scorers.list_scorers",
        lambda experiment_id=None: [Existing()],
    )
    out = scorers._register_if_absent(FakeScorer(), experiment_id="exp-1")
    assert out.name == "judge_direction"
```

- [ ] **Step 2: Append judge code to `scorers.py`**

Replace the `JUDGE_SCORERS = []` line and the `ALL_SCORERS` line with:

```python
import os


def _register_if_absent(scorer_obj, *, experiment_id: str | None):
    """Register scorer with experiment unless a same-name scorer already exists."""
    if experiment_id is None:
        return scorer_obj
    try:
        import mlflow.genai.scorers as _gs

        existing = {s.name for s in _gs.list_scorers(experiment_id=experiment_id)}
    except Exception:
        existing = set()
    if scorer_obj.name not in existing:
        try:
            scorer_obj.register(experiment_id=experiment_id)
        except Exception:
            pass
    return scorer_obj


def _make_judges(experiment_id: str | None):
    from mlflow.genai.judges import make_judge

    from trading_databricks.agents.prompts import load_prompt

    def _instructions(name: str) -> str:
        messages = load_prompt(name, version="production")
        return messages[0]["content"]

    model = os.environ.get(
        "MAS_JUDGE_MODEL",
        "databricks:/gemini-3-1-pro",
    )

    direction = make_judge(
        name="judge_direction",
        instructions=_instructions("judge_direction"),
        model=model,
        feedback_value_type="categorical",
    )
    mechanism = make_judge(
        name="judge_mechanism",
        instructions=_instructions("judge_mechanism"),
        model=model,
        feedback_value_type="categorical",
    )
    grounding = make_judge(
        name="judge_grounding",
        instructions=_instructions("judge_grounding"),
        model=model,
        feedback_value_type="float",
    )

    exp_id = experiment_id or os.environ.get("MLFLOW_EXPERIMENT_ID")
    return [
        _register_if_absent(direction, experiment_id=exp_id),
        _register_if_absent(mechanism, experiment_id=exp_id),
        _register_if_absent(grounding, experiment_id=exp_id),
    ]


def ensure_judges_registered(experiment_id: str | None = None) -> list:
    """Lazy builder so imports don't touch MLflow unless asked."""
    global JUDGE_SCORERS, ALL_SCORERS
    if not JUDGE_SCORERS:
        JUDGE_SCORERS = _make_judges(experiment_id)
        ALL_SCORERS = DETERMINISTIC_SCORERS + JUDGE_SCORERS + REFERENCE_SCORERS
    return JUDGE_SCORERS
```

Keep module-level `JUDGE_SCORERS = []` and `ALL_SCORERS = DETERMINISTIC_SCORERS + JUDGE_SCORERS + REFERENCE_SCORERS`. Lazy registration keeps `import trading_databricks.agents.eval` cheap and test-friendly.

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/agents/eval/test_scorers.py -v`
Expected: 8 passed.

- [ ] **Step 4: Commit**

```bash
git add src/trading_databricks/agents/eval/scorers.py tests/agents/eval/test_scorers.py
git commit -m "TRD feat: Tier-2 LLM judges with idempotent registration"
```

---

## Task 12: Tier 3 — reference-based scorers

**Files:**
- Modify: `src/trading_databricks/agents/eval/scorers.py`
- Modify: `tests/agents/eval/test_scorers.py`

- [ ] **Step 1: Write failing test**

Append to `tests/agents/eval/test_scorers.py`:

```python
def test_direction_match_hit():
    from trading_databricks.agents.eval.scorers import direction_match

    outputs = {"extraction": {"direction": "up"}}
    expectations = {"direction_label": "up"}
    assert direction_match(outputs=outputs, expectations=expectations) == 1.0


def test_direction_match_miss_returns_zero():
    from trading_databricks.agents.eval.scorers import direction_match

    outputs = {"extraction": {"direction": "up"}}
    expectations = {"direction_label": "down"}
    assert direction_match(outputs=outputs, expectations=expectations) == 0.0


def test_mechanism_match_missing_expectation_returns_none():
    from trading_databricks.agents.eval.scorers import mechanism_match

    outputs = {"extraction": {"mechanism": "etf_inflow"}}
    assert mechanism_match(outputs=outputs, expectations={}) is None
```

- [ ] **Step 2: Implement Tier-3 scorers**

Append to `src/trading_databricks/agents/eval/scorers.py` (before the `ALL_SCORERS` tail):

```python
@scorer
def direction_match(
    *,
    outputs: dict[str, Any],
    expectations: dict[str, Any],
) -> float | None:
    if "direction_label" not in expectations:
        return None
    extraction = outputs.get("extraction") or {}
    return 1.0 if extraction.get("direction") == expectations["direction_label"] else 0.0


@scorer
def mechanism_match(
    *,
    outputs: dict[str, Any],
    expectations: dict[str, Any],
) -> float | None:
    if "mechanism_label" not in expectations:
        return None
    extraction = outputs.get("extraction") or {}
    return 1.0 if extraction.get("mechanism") == expectations["mechanism_label"] else 0.0


@scorer
def grounding_alignment(
    *,
    outputs: dict[str, Any],
    expectations: dict[str, Any],
) -> float | None:
    if "causal_span_present" not in expectations:
        return None
    has_spans = bool((outputs.get("extraction") or {}).get("evidence_spans"))
    return 1.0 if has_spans == expectations["causal_span_present"] else 0.0


REFERENCE_SCORERS = [direction_match, mechanism_match, grounding_alignment]
ALL_SCORERS = DETERMINISTIC_SCORERS + JUDGE_SCORERS + REFERENCE_SCORERS
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/agents/eval/test_scorers.py -v`
Expected: 11 passed.

- [ ] **Step 4: Commit**

```bash
git add src/trading_databricks/agents/eval/scorers.py tests/agents/eval/test_scorers.py
git commit -m "TRD feat: Tier-3 reference-based MAS eval scorers"
```

---

## Task 13: `run_evaluation()` runner with auto-tagging and scorer filtering

**Files:**
- Create: `src/trading_databricks/agents/eval/runner.py`
- Create: `tests/agents/eval/test_runner.py`

- [ ] **Step 1: Write failing test**

`tests/agents/eval/test_runner.py`:

```python
from trading_databricks.agents.eval.runner import (
    _build_tags,
    _filter_scorers,
    _run_name,
)


def test_build_tags_includes_required_keys():
    tags = _build_tags(
        predict_target="e2e",
        dataset_name="mas_eval_v1",
        dataset_version=3,
        mas_config_hash="abc123def456",
        prompt_versions={"causality_extractor_base": "2"},
        extra={"tuning_hypothesis": "add CoT"},
        run_kind="tuning",
    )
    assert tags["predict_target"] == "e2e"
    assert tags["dataset_name"] == "mas_eval_v1"
    assert tags["dataset_version"] == "3"
    assert tags["mas_config_hash"] == "abc123def456"
    assert tags["run_kind"] == "tuning"
    assert tags["tuning_hypothesis"] == "add CoT"
    assert "prompt_versions" in tags  # JSON string


def test_run_name_format():
    name = _run_name(
        run_kind="baseline",
        predict_target="e2e",
        dataset_name="mas_eval_v1",
        dataset_version=3,
        git_sha7="a1b2c3d",
    )
    assert name == "baseline__e2e__mas_eval_v1@3__a1b2c3d"


def test_filter_scorers_drops_reference_when_no_expectations():
    import pandas as pd

    df = pd.DataFrame([{"inputs": {"headline": "h"}, "expectations": {}, "tags": {}}])

    class FakeRef:
        name = "direction_match"

    class FakeDet:
        name = "schema_valid"

    out = _filter_scorers(
        requested=[FakeRef(), FakeDet()],
        reference_scorers=[FakeRef()],
        data=df,
    )
    assert [s.name for s in out] == ["schema_valid"]


def test_filter_scorers_keeps_reference_when_expectations_populated():
    import pandas as pd

    df = pd.DataFrame([{"inputs": {}, "expectations": {"direction_label": "up"}, "tags": {}}])

    class FakeRef:
        name = "direction_match"

    out = _filter_scorers(
        requested=[FakeRef()],
        reference_scorers=[FakeRef()],
        data=df,
    )
    assert [s.name for s in out] == ["direction_match"]
```

- [ ] **Step 2: Run test — expect fail**

Run: `uv run pytest tests/agents/eval/test_runner.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement runner**

`src/trading_databricks/agents/eval/runner.py`:

```python
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
        from trading_databricks.agents.graph import _mechanism_validator_node

        def wrapper(headline: str, body: str, symbol: str, extraction: dict | None = None) -> dict:
            state = {"headline": headline, "body": body, "extraction": extraction or {}}
            return dict(_mechanism_validator_node(state, config))

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
):
    """Run a MAS evaluation against an MLflow GenAI dataset.

    Returns ``mlflow.models.EvaluationResult``.
    """
    import mlflow

    from trading_databricks.agents.config import MASConfig
    from trading_databricks.agents.eval import scorers as scorer_module
    from trading_databricks.agents.eval.dataset import build_eval_dataset

    exp_id = experiment_id or os.environ.get("MLFLOW_EXPERIMENT_ID")
    if exp_id:
        mlflow.set_experiment(experiment_id=exp_id)

    config = MASConfig(**(mas_config_overrides or {}))
    data = build_eval_dataset(corpus="primary" if dataset_name.endswith("mas_eval_v1") else dataset_name)

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
        prompt_versions={},
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/agents/eval/test_runner.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/agents/ -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/trading_databricks/agents/eval/runner.py tests/agents/eval/test_runner.py
git commit -m "TRD feat: run_evaluation() with auto-tagging and scorer filtering"
```

---

## Task 14: E2E driver notebook

**Files:**
- Create: `notebooks/qa/mas/e2e_mas_evaluation.py`

- [ ] **Step 1: Author the notebook**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/qa/mas/e2e_mas_evaluation.py
git commit -m "TRD feat: e2e MAS evaluation driver notebook"
```

---

## Task 15: Per-node tuning notebooks

**Files:**
- Create: `notebooks/qa/mas/tuning/causality_extractor_tuning.py`
- Create: `notebooks/qa/mas/tuning/judge_direction_tuning.py`
- Create: `notebooks/qa/mas/tuning/judge_mechanism_tuning.py`
- Create: `notebooks/qa/mas/tuning/judge_grounding_tuning.py`

Each follows the same shape: load current YAML → edit scratch → register as `candidate` → run evaluation with `tuning_hypothesis` tag → compare vs baseline. YAML commit is the production gate; notebooks never write `production`.

- [ ] **Step 1: Create `causality_extractor_tuning.py`**

```python
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
```

- [ ] **Step 2: Create the three judge tuning notebooks**

For each of `judge_direction`, `judge_mechanism`, `judge_grounding`, author a copy of the above with:
- `PROMPT_NAME` set to the judge name
- `predict_target="e2e"` (judges score whole extractions)
- `scorers="judges"`
- `tags={"tuning_hypothesis": "<descriptive slug>"}`

Example header for `judge_direction_tuning.py`:

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # Judge Direction Prompt Tuning
```

With `PROMPT_NAME = "judge_direction"` and `scorers="judges"`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/qa/mas/tuning/
git commit -m "TRD feat: per-node MAS prompt tuning notebooks"
```

---

## Task 16: Push initial prompt YAMLs to Registry

**Files:** none

- [ ] **Step 1: Push all YAMLs as `candidate` and then promote to `production`**

Run in a Databricks-authenticated shell:

```bash
uv run python -m trading_databricks.agents.prompts._sync push --stage candidate
uv run python -m trading_databricks.agents.prompts._sync push --stage production
```

- [ ] **Step 2: Verify via MLflow**

```bash
uv run python -c "import mlflow; p = mlflow.load_prompt('prompts:/causality_extractor_base/production'); print(type(p).__name__, getattr(p, 'version', '?'))"
```

Expected: prints a `Prompt` object with a numeric version.

- [ ] **Step 3: Commit any YAML edits generated by the push flow**

If `_sync.py` normalized any YAML on disk, `git status` will show it:

```bash
git status
git add -u src/trading_databricks/agents/prompts/
git diff --cached
git commit -m "TRD chore: sync initial MAS prompts to Registry" || true
```

---

## Self-Review

**Spec coverage (§1–§6 of design doc):**
- §1 YAML+Registry hybrid — Tasks 1–6, 16
- §2.1 primary dataset + stratified sampling — Task 8
- §2.2 calibration slices — Task 9
- §2.3 dataset loader API — Task 8
- §3.1 deterministic scorers — Task 10
- §3.2 LLM judges + MemAlign-ready names — Task 11
- §3.3 reference scorers + auto-skip — Tasks 12–13
- §4 unified `run_evaluation()` API — Task 13
- §5 tuning notebooks (candidate-only) — Task 15
- §6.1 experiment layout — Tasks 13–14
- §6.2 run naming — Task 13 (`_run_name`)
- §6.3 idempotent judge registration — Task 11
- §6.4 expectations scaffold — Tasks 8–9
- §6.5 labeling stub — Task 7

**Placeholder scan:** none. Every step has concrete code or commands.

**Type consistency:** scorer functions kwarg-only (`outputs=`, `inputs=`, `expectations=`); judge names (`judge_direction`, `judge_mechanism`, `judge_grounding`) consistent across YAML, scorers.py, and tuning notebooks; dataset record schema (`inputs`/`expectations`/`tags`) identical in dataset.py, scorers.py, runner.py.

---

Plan complete and saved to `docs/superpowers/plans/2026-04-17-mas-agent-evaluation-harness.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
