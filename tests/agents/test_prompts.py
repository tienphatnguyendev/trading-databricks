"""Tests for causality extractor prompts."""
import pytest
from trading_databricks.agents.prompts import (
    format_compact_taxonomy,
    format_full_taxonomy,
    format_whitelist,
    format_few_shot,
    build_system_prompt
)

def test_format_compact_taxonomy():
    taxonomy = {"Macro": ["rate_decision", "inflation"], "Security": ["exchange_hack"]}
    res = format_compact_taxonomy(taxonomy)
    assert res == "Macro.inflation, Macro.rate_decision, Security.exchange_hack"

def test_format_full_taxonomy():
    taxonomy = {"Macro": ["rate_decision"]}
    res = format_full_taxonomy(taxonomy)
    assert "- Macro:" in res
    assert "  - Macro.rate_decision" in res

def test_format_whitelist():
    whitelist = {("Macro", "risk_off"): ["bearish"]}
    res = format_whitelist(whitelist)
    assert "L1 Category: Macro" in res
    assert "  - risk_off" in res

def test_format_few_shot():
    examples = [{"trigger_event": "Hack", "causal_chain_json": '{"a": 1}'}]
    res = format_few_shot(examples, 1)
    assert "Headline: Hack" in res
    assert "Extraction:" in res
    
def test_build_system_prompt_fast():
    taxonomy = {"A": ["b"]}
    whitelist = {("A", "c"): ["bullish"]}
    prompt = build_system_prompt(taxonomy, whitelist, [], "fast")
    assert "A.b" in prompt
    assert "MECHANISM WHITELIST" not in prompt

def test_build_system_prompt_careful():
    taxonomy = {"A": ["b"]}
    whitelist = {("A", "c"): ["bullish"]}
    prompt = build_system_prompt(taxonomy, whitelist, [], "careful")
    assert "- A:" in prompt
    assert "CAREFUL PATH: Please provide explicit chain-of-thought" in prompt
