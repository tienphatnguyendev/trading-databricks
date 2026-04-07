"""Tests for agents/gold.py — news_causal_features_transform."""
from __future__ import annotations

from unittest.mock import MagicMock

from trading_databricks.agents.config import GatekeeperConfig


def test_gatekeeper_udf_passes_crypto_article() -> None:
    """_make_gatekeeper_udf returns True for articles mentioning Bitcoin."""
    import pandas as pd

    from trading_databricks.agents.gold import _make_gatekeeper_udf

    udf_fn = _make_gatekeeper_udf(GatekeeperConfig())
    result = udf_fn.func(
        pd.Series(["Bitcoin ETF approved by SEC"]),
        pd.Series(["The SEC has approved the first spot Bitcoin ETF."]),
    )
    assert result.iloc[0] is True


def test_gatekeeper_udf_rejects_non_crypto_article() -> None:
    """_make_gatekeeper_udf returns False for articles with no crypto entities."""
    import pandas as pd

    from trading_databricks.agents.gold import _make_gatekeeper_udf

    udf_fn = _make_gatekeeper_udf(GatekeeperConfig())
    result = udf_fn.func(
        pd.Series(["Fed raises interest rates"]),
        pd.Series(["The Federal Reserve raised rates by 25bp in a unanimous vote."]),
    )
    assert result.iloc[0] is False


def test_news_causal_features_transform_calls_spark_and_filters() -> None:
    """news_causal_features_transform reads silver table, filters, adds ai_query col."""
    from trading_databricks.agents.gold import news_causal_features_transform

    spark = MagicMock()
    mock_df = MagicMock()
    mock_df.filter.return_value = mock_df
    mock_df.withColumn.return_value = mock_df
    mock_df.select.return_value = mock_df
    spark.table.return_value = mock_df

    result = news_causal_features_transform(
        spark,
        catalog="trading-signal-gen-dev",
        gatekeeper_config=GatekeeperConfig(),
    )

    spark.table.assert_called_once_with("trading-signal-gen-dev.silver.news_cleaned")
    mock_df.filter.assert_called_once()
    mock_df.withColumn.assert_called_once()
    mock_df.select.assert_called_once()
    assert result is mock_df
