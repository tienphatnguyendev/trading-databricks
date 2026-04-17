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
