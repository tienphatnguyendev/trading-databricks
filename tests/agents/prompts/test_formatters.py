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
