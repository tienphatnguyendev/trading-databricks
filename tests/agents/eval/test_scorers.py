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
    outputs = {"extraction": {"treatment_class": "missing_is_impactful"}}
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

def test_register_if_absent_skips_when_already_registered(monkeypatch):
    from trading_databricks.agents.eval import scorers

    class FakeScorer:
        name = "judge_direction"
        def register(self, *, experiment_id):
            raise AssertionError("should not be called when scorer already exists")

    class Existing:
        name = "judge_direction"

    monkeypatch.setattr(
        "mlflow.genai.scorers.get_all_scorers",
        lambda experiment_id=None: [Existing()],
    )
    out = scorers._register_if_absent(FakeScorer(), experiment_id="exp-1")
    assert out.name == "judge_direction"

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
