"""Tests for agents/tools.py — UC function tools."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from trading_databricks.agents.config import MASConfig


@patch("trading_databricks.agents.tools.get_spark_session")
def test_fetch_mechanism_whitelist_returns_json_string(mock_spark: MagicMock) -> None:
    """fetch_mechanism_whitelist returns a valid JSON array string."""
    from trading_databricks.agents.tools import fetch_mechanism_whitelist

    mock_row_1 = MagicMock()
    mock_row_1.asDict.return_value = {
        "treatment_l1": "Regulation",
        "mechanism": "regulatory_clarity",
        "valid_directions": ["bullish"],
        "description": "Clear regulation enables access",
    }
    mock_row_2 = MagicMock()
    mock_row_2.asDict.return_value = {
        "treatment_l1": "Macro",
        "mechanism": "liquidity_contraction",
        "valid_directions": ["bearish"],
        "description": "Tight monetary policy drains risk assets",
    }

    mock_spark_instance = MagicMock()
    mock_spark_instance.table.return_value.collect.return_value = [mock_row_1, mock_row_2]
    mock_spark.return_value = mock_spark_instance

    result = fetch_mechanism_whitelist()

    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["treatment_l1"] == "Regulation"
    assert parsed[1]["mechanism"] == "liquidity_contraction"

    mock_spark_instance.table.assert_called_once_with(
        "trading-signal-gen-dev.gold.mechanism_whitelist"
    )


@patch("trading_databricks.agents.tools.get_spark_session")
def test_fetch_mechanism_whitelist_uses_config_catalog(mock_spark: MagicMock) -> None:
    """fetch_mechanism_whitelist respects the catalog from MASConfig."""
    from trading_databricks.agents.tools import fetch_mechanism_whitelist

    mock_spark_instance = MagicMock()
    mock_spark_instance.table.return_value.collect.return_value = []
    mock_spark.return_value = mock_spark_instance

    custom_config = MASConfig(catalog="my-catalog", schema="gold")
    result = fetch_mechanism_whitelist(config=custom_config)

    assert json.loads(result) == []
    mock_spark_instance.table.assert_called_once_with("my-catalog.gold.mechanism_whitelist")
