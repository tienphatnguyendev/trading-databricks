import json

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


def test_build_tags_prompt_versions_serialized():
    tags = _build_tags(
        predict_target="causality_extractor",
        dataset_name="mas_eval_v1",
        dataset_version="latest",
        mas_config_hash="abc123",
        prompt_versions={"causality_extractor_base": "3"},
        extra=None,
        run_kind="tuning",
    )
    assert json.loads(tags["prompt_versions"]) == {"causality_extractor_base": "3"}


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
