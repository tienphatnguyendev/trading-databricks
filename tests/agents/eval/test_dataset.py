import polars as pl
import pytest

from trading_databricks.agents.eval.dataset import (
    _resolve_corpus,
    _stratified_sample,
    CALIBRATION_DATASETS,
    PRIMARY_DATASET_NAME,
)


def test_stratified_sample_proportional_allocation():
    df = pl.DataFrame({
        "id": list(range(100)),
        "source": ["coindesk"] * 40 + ["coindesk"] * 20 + ["theblock"] * 20 + ["theblock"] * 20,
        "token": ["BTC"] * 40 + ["ETH"] * 20 + ["BTC"] * 20 + ["ETH"] * 20,
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

def test_resolve_corpus_primary():
    assert _resolve_corpus(PRIMARY_DATASET_NAME) == "primary"


def test_resolve_corpus_calibration():
    assert _resolve_corpus(CALIBRATION_DATASETS["cryptolin"]) == "cryptolin"


def test_resolve_corpus_unknown_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        _resolve_corpus("smart-trading-dev.04_eval.nonexistent_v1")


def test_load_calibration_unknown_corpus_raises():
    from trading_databricks.agents.eval.dataset import _load_calibration

    with pytest.raises(ValueError, match="unsupported"):
        _load_calibration("not_a_corpus")
