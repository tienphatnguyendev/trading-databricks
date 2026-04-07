import pytest

from trading_databricks.agents.config import MASConfig
from trading_databricks.agents.nodes.uncertainty_router import uncertainty_router


@pytest.fixture
def config():
    return MASConfig()

def test_router_cold_start(config):
    # Empty examples
    state = {"few_shot_examples": []}
    result = uncertainty_router(state, config)
    assert result == {"extraction_path": "careful"}

def test_router_insufficient_results(config):
    # Less than min_vs_results (which is 2 by default)
    state = {"few_shot_examples": [{"similarity_score": 0.99}]}
    result = uncertainty_router(state, config)
    assert result == {"extraction_path": "careful"}

def test_router_fast_path(config):
    # top_similarity >= 0.85
    state = {
        "few_shot_examples": [
            {"similarity_score": 0.90},
            {"similarity_score": 0.88}
        ]
    }
    result = uncertainty_router(state, config)
    assert result == {"extraction_path": "fast"}

def test_router_standard_path(config):
    # 0.50 <= top_similarity < 0.85
    state = {
        "few_shot_examples": [
            {"similarity_score": 0.70},
            {"similarity_score": 0.65}
        ]
    }
    result = uncertainty_router(state, config)
    assert result == {"extraction_path": "standard"}

def test_router_careful_path_low_similarity(config):
    # top_similarity < 0.50
    state = {
        "few_shot_examples": [
            {"similarity_score": 0.30},
            {"similarity_score": 0.25}
        ]
    }
    result = uncertainty_router(state, config)
    assert result == {"extraction_path": "careful"}

def test_uncertainty_router_is_exported():
    import trading_databricks.agents.nodes as nodes
    assert hasattr(nodes, "uncertainty_router")
    assert "uncertainty_router" in nodes.__all__

