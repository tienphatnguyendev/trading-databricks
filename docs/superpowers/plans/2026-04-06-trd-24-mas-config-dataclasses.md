# TRD-24: MAS Config Dataclasses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement four frozen dataclasses (`GatekeeperConfig`, `UncertaintyRouterConfig`, `MechanismValidatorConfig`, `MASConfig`) in `src/trading_databricks/agents/config.py` so all downstream MAS Wave 2+ agents have a single source of truth for thresholds and feature flags.

**Architecture:** Four `@dataclass(frozen=True)` classes in a single file. `MASConfig` composes the other three via `field(default_factory=...)`. No Spark dependency — pure Python. Merges before TRD-23 (TRD-23 is developed in parallel in a separate worktree).

**Tech Stack:** Python 3.12, `dataclasses` stdlib, `pytest`, `uv`

**Spec:** `docs/superpowers/specs/2026-04-06-mas-w1-schema-config-design.md` §2

---

## File Map

| Action | Path |
|--------|------|
| Create | `src/trading_databricks/agents/config.py` |
| Create | `tests/agents/__init__.py` |
| Create | `tests/agents/test_config.py` |

---

## Task 1: Branch Setup

- [ ] **Step 1.1: Create branch from main**

```bash
git checkout main && git pull
git checkout -b feat/TRD-24-mas-config-dataclasses
```

Expected: Switched to a new branch 'feat/TRD-24-mas-config-dataclasses'

- [ ] **Step 1.2: Create tests/agents package**

```bash
touch tests/agents/__init__.py
```

- [ ] **Step 1.3: Create empty test file to verify import structure**

Create `tests/agents/test_config.py` with only the import block:

```python
"""Tests for trading_databricks.agents.config frozen dataclasses."""
from __future__ import annotations

import pytest

from trading_databricks.agents.config import (
    GatekeeperConfig,
    MASConfig,
    MechanismValidatorConfig,
    UncertaintyRouterConfig,
)
```

- [ ] **Step 1.4: Run to confirm ImportError (file doesn't exist yet)**

```bash
uv run pytest tests/agents/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'trading_databricks.agents.config'`

---

## Task 2: GatekeeperConfig

- [ ] **Step 2.1: Write failing tests for GatekeeperConfig**

Replace `tests/agents/test_config.py` with:

```python
"""Tests for trading_databricks.agents.config frozen dataclasses."""
from __future__ import annotations

import pytest

from trading_databricks.agents.config import (
    GatekeeperConfig,
    MASConfig,
    MechanismValidatorConfig,
    UncertaintyRouterConfig,
)


class TestGatekeeperConfig:
    def test_defaults(self):
        cfg = GatekeeperConfig()
        assert cfg.model_uri == "models:/news-gatekeeper/Production"
        assert cfg.confidence_threshold == 0.5
        assert cfg.ner_only_mode is True
        assert cfg.text_window == 500

    def test_frozen(self):
        cfg = GatekeeperConfig()
        with pytest.raises(Exception):
            cfg.model_uri = "other"  # type: ignore[misc]

    def test_custom_values(self):
        cfg = GatekeeperConfig(confidence_threshold=0.7, ner_only_mode=False)
        assert cfg.confidence_threshold == 0.7
        assert cfg.ner_only_mode is False
        assert cfg.text_window == 500  # other defaults preserved
```

- [ ] **Step 2.2: Run to confirm tests fail (ImportError)**

```bash
uv run pytest tests/agents/test_config.py::TestGatekeeperConfig -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 2.3: Implement GatekeeperConfig**

Create `src/trading_databricks/agents/config.py`:

```python
"""Frozen configuration dataclasses for all MAS components.

All values match the spec defaults exactly. Pass instances through
agent nodes rather than threading raw parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GatekeeperConfig:
    model_uri: str = "models:/news-gatekeeper/Production"
    confidence_threshold: float = 0.5
    ner_only_mode: bool = True   # V1: NER only, no MLflow dependency
    text_window: int = 500       # chars of body concatenated with headline
```

- [ ] **Step 2.4: Run to confirm GatekeeperConfig tests pass**

```bash
uv run pytest tests/agents/test_config.py::TestGatekeeperConfig -v
```

Expected: 3 passed

- [ ] **Step 2.5: Commit**

```bash
git add src/trading_databricks/agents/config.py tests/agents/__init__.py tests/agents/test_config.py
git commit -m "TRD-24 feat: add GatekeeperConfig frozen dataclass"
```

---

## Task 3: UncertaintyRouterConfig

- [ ] **Step 3.1: Add UncertaintyRouterConfig tests to test_config.py**

Append after `TestGatekeeperConfig` class:

```python
class TestUncertaintyRouterConfig:
    def test_defaults(self):
        cfg = UncertaintyRouterConfig()
        assert cfg.high_similarity_threshold == 0.85
        assert cfg.low_similarity_threshold == 0.50
        assert cfg.self_consistency_k == 3
        assert cfg.min_vs_results == 2
        assert cfg.min_index_size == 50

    def test_frozen(self):
        cfg = UncertaintyRouterConfig()
        with pytest.raises(Exception):
            cfg.self_consistency_k = 10  # type: ignore[misc]

    def test_custom_thresholds(self):
        cfg = UncertaintyRouterConfig(high_similarity_threshold=0.9, self_consistency_k=5)
        assert cfg.high_similarity_threshold == 0.9
        assert cfg.self_consistency_k == 5
        assert cfg.low_similarity_threshold == 0.50  # default preserved
```

- [ ] **Step 3.2: Run to confirm new tests fail**

```bash
uv run pytest tests/agents/test_config.py::TestUncertaintyRouterConfig -v
```

Expected: `ImportError` (UncertaintyRouterConfig not yet defined)

- [ ] **Step 3.3: Add UncertaintyRouterConfig to config.py**

Append after `GatekeeperConfig`:

```python

@dataclass(frozen=True)
class UncertaintyRouterConfig:
    high_similarity_threshold: float = 0.85   # fast path (VS cosine ≥ this)
    low_similarity_threshold: float = 0.50    # careful path (VS cosine < this)
    self_consistency_k: int = 3               # LLM calls on careful path
    min_vs_results: int = 2                   # minimum VS matches for fast path
    min_index_size: int = 50                  # below this: force careful regardless
```

- [ ] **Step 3.4: Run to confirm all tests pass**

```bash
uv run pytest tests/agents/test_config.py -v
```

Expected: 6 passed

- [ ] **Step 3.5: Commit**

```bash
git add src/trading_databricks/agents/config.py tests/agents/test_config.py
git commit -m "TRD-24 feat: add UncertaintyRouterConfig frozen dataclass"
```

---

## Task 4: MechanismValidatorConfig

- [ ] **Step 4.1: Add MechanismValidatorConfig tests**

Append after `TestUncertaintyRouterConfig`:

```python
class TestMechanismValidatorConfig:
    def test_defaults(self):
        cfg = MechanismValidatorConfig()
        assert cfg.evidence_grounding_reject_threshold is None
        assert cfg.embedding_model == "databricks-gte-large-en"

    def test_frozen(self):
        cfg = MechanismValidatorConfig()
        with pytest.raises(Exception):
            cfg.embedding_model = "other"  # type: ignore[misc]

    def test_none_threshold_means_log_only(self):
        # None = informational only in V3; rejection threshold added in V4
        cfg = MechanismValidatorConfig()
        assert cfg.evidence_grounding_reject_threshold is None

    def test_custom_threshold(self):
        cfg = MechanismValidatorConfig(evidence_grounding_reject_threshold=0.3)
        assert cfg.evidence_grounding_reject_threshold == 0.3
```

- [ ] **Step 4.2: Run to confirm new tests fail**

```bash
uv run pytest tests/agents/test_config.py::TestMechanismValidatorConfig -v
```

Expected: `ImportError` (MechanismValidatorConfig not yet defined)

- [ ] **Step 4.3: Add MechanismValidatorConfig to config.py**

Append after `UncertaintyRouterConfig`:

```python

@dataclass(frozen=True)
class MechanismValidatorConfig:
    evidence_grounding_reject_threshold: float | None = None  # None = log only (V3)
    embedding_model: str = "databricks-gte-large-en"
```

- [ ] **Step 4.4: Run to confirm all tests pass**

```bash
uv run pytest tests/agents/test_config.py -v
```

Expected: 10 passed

- [ ] **Step 4.5: Commit**

```bash
git add src/trading_databricks/agents/config.py tests/agents/test_config.py
git commit -m "TRD-24 feat: add MechanismValidatorConfig frozen dataclass"
```

---

## Task 5: MASConfig

- [ ] **Step 5.1: Add MASConfig tests**

Append after `TestMechanismValidatorConfig`:

```python
class TestMASConfig:
    def test_constructs_with_all_nested_defaults(self):
        cfg = MASConfig()
        assert isinstance(cfg.gatekeeper, GatekeeperConfig)
        assert isinstance(cfg.router, UncertaintyRouterConfig)
        assert isinstance(cfg.validator, MechanismValidatorConfig)

    def test_nested_default_values_correct(self):
        cfg = MASConfig()
        assert cfg.gatekeeper.confidence_threshold == 0.5
        assert cfg.gatekeeper.ner_only_mode is True
        assert cfg.router.high_similarity_threshold == 0.85
        assert cfg.router.min_index_size == 50
        assert cfg.validator.evidence_grounding_reject_threshold is None
        assert cfg.validator.embedding_model == "databricks-gte-large-en"

    def test_scalar_defaults(self):
        cfg = MASConfig()
        assert cfg.catalog == "trading-signal-gen-dev"
        assert cfg.schema == "gold"
        assert cfg.vs_index_name == "trading-signal-gen-dev.gold.causal_chains_idx"
        assert cfg.vs_endpoint_name == "mas-few-shot"
        assert cfg.llm_endpoint == "databricks-claude-sonnet"
        assert cfg.agent_model_version == "v1.0"

    def test_frozen(self):
        cfg = MASConfig()
        with pytest.raises(Exception):
            cfg.catalog = "other"  # type: ignore[misc]

    def test_custom_gatekeeper_preserves_other_defaults(self):
        cfg = MASConfig(gatekeeper=GatekeeperConfig(ner_only_mode=False))
        assert cfg.gatekeeper.ner_only_mode is False
        assert cfg.gatekeeper.confidence_threshold == 0.5
        assert cfg.router.high_similarity_threshold == 0.85  # unrelated config unaffected

    def test_catalog_override(self):
        cfg = MASConfig(catalog="trading-signal-gen-prod")
        assert cfg.catalog == "trading-signal-gen-prod"
        assert cfg.gatekeeper.confidence_threshold == 0.5  # nested unaffected

    def test_two_instances_are_independent(self):
        cfg_a = MASConfig(catalog="trading-signal-gen-dev")
        cfg_b = MASConfig(catalog="trading-signal-gen-prod")
        assert cfg_a.catalog != cfg_b.catalog
        # nested objects are independent instances
        assert cfg_a.gatekeeper is not cfg_b.gatekeeper
```

- [ ] **Step 5.2: Run to confirm MASConfig tests fail**

```bash
uv run pytest tests/agents/test_config.py::TestMASConfig -v
```

Expected: `ImportError` (MASConfig not yet defined)

- [ ] **Step 5.3: Add MASConfig to config.py**

Append after `MechanismValidatorConfig`:

```python

@dataclass(frozen=True)
class MASConfig:
    gatekeeper: GatekeeperConfig = field(default_factory=GatekeeperConfig)
    router: UncertaintyRouterConfig = field(default_factory=UncertaintyRouterConfig)
    validator: MechanismValidatorConfig = field(default_factory=MechanismValidatorConfig)
    catalog: str = "trading-signal-gen-dev"
    schema: str = "gold"
    vs_index_name: str = "trading-signal-gen-dev.gold.causal_chains_idx"
    vs_endpoint_name: str = "mas-few-shot"
    llm_endpoint: str = "databricks-claude-sonnet"
    agent_model_version: str = "v1.0"
```

- [ ] **Step 5.4: Run full test suite to confirm all pass**

```bash
uv run pytest tests/agents/test_config.py -v
```

Expected: 17 passed

- [ ] **Step 5.5: Run entire project test suite to confirm no regressions**

```bash
uv run pytest -v
```

Expected: All existing tests pass + 17 new tests pass

- [ ] **Step 5.6: Commit**

```bash
git add src/trading_databricks/agents/config.py tests/agents/test_config.py
git commit -m "TRD-24 feat: add MASConfig frozen dataclass with nested component configs"
```

---

## Task 6: PR and Coordination

- [ ] **Step 6.1: Update coordination file Section 1 (in_progress)**

Edit `docs/coordination/TRD-6-gen-ai-agent-layer.md` — update Section 1:

```markdown
## Section 1: Schema & Config — TRD-23, TRD-24 (Wave 1)
- **Status:** in_progress
- **Branch:** feat/TRD-24-mas-config-dataclasses
- **Modified files:**
  - src/trading_databricks/agents/config.py (NEW)
  - tests/agents/__init__.py (NEW)
  - tests/agents/test_config.py (NEW)
- **Exports (public APIs):**
  - config.py: GatekeeperConfig, UncertaintyRouterConfig, MechanismValidatorConfig, MASConfig
- **Needs from others:** None (foundation)
- **Notes:** TRD-24 done. TRD-23 in parallel. Both must merge before Wave 2 starts.
```

- [ ] **Step 6.2: Commit coordination update**

```bash
git add docs/coordination/TRD-6-gen-ai-agent-layer.md
git commit -m "TRD-24 chore: update coordination file with in_progress status"
```

- [ ] **Step 6.3: Push branch**

```bash
git push -u origin feat/TRD-24-mas-config-dataclasses
```

- [ ] **Step 6.4: Create PR**

```bash
gh pr create \
  --title "TRD-24: Add frozen config dataclasses for all MAS components" \
  --body "$(cat <<'EOF'
## Summary

- Adds `GatekeeperConfig`, `UncertaintyRouterConfig`, `MechanismValidatorConfig`, `MASConfig` as `@dataclass(frozen=True)` to `src/trading_databricks/agents/config.py`
- All defaults match `2026-04-06-mas-w1-schema-config-design.md` spec exactly
- `MASConfig` uses `field(default_factory=...)` for nested component configs
- No Spark dependency — pure Python, all tests run locally

## Test plan

- [x] `uv run pytest tests/agents/test_config.py -v` — 17 tests pass
- [x] `uv run pytest -v` — full suite, no regressions
- [ ] Merge before TRD-23 (parallel development, TRD-24 merges first)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
