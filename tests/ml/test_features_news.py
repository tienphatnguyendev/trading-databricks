from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from trading_databricks.ml.features.news import compute_news

_L1_CATEGORIES = [
    "Sentiment", "DeFi", "Supply", "Exchange",
    "Macro", "Adoption", "Security", "Regulation",
]
_EXPECTED_NEWS_COLS = (
    {"event_id", "symbol", "event_ts", "computed_at",
     "news_volume_24h", "news_sentiment_score", "has_news_24h"}
    | {f"news_count_{l1}" for l1 in _L1_CATEGORIES}
    | {f"news_max_conf_{l1}" for l1 in _L1_CATEGORIES}
    | {f"news_direction_{l1}" for l1 in _L1_CATEGORIES}
    | {f"news_accepted_ratio_{l1}" for l1 in _L1_CATEGORIES}
)


def _make_events_lf(bar_indices: list[int]) -> pl.LazyFrame:
    base = datetime(2025, 1, 2)
    return pl.DataFrame(
        {
            "event_id": [f"e{i}" for i in bar_indices],
            "symbol": ["BTCUSDT"] * len(bar_indices),
            "event_ts": [base + timedelta(hours=idx) for idx in bar_indices],
            "bar_index": bar_indices,
        }
    ).lazy()


def test_zero_fallback_when_no_news_data() -> None:
    """compute_news(news_features=None) returns zero-filled LazyFrame."""
    events = _make_events_lf([5, 10, 15])
    result = compute_news(events, None).collect()

    assert set(result.columns) == _EXPECTED_NEWS_COLS
    assert len(result) == 3
    # All numeric news features must be 0 or False
    assert result["news_volume_24h"].sum() == 0
    assert result["news_sentiment_score"].sum() == 0.0
    assert result["has_news_24h"].sum() == 0


def test_zero_fallback_column_types() -> None:
    """Zero-fallback has correct Polars dtypes."""
    events = _make_events_lf([5])
    result = compute_news(events, None).collect()

    assert result.dtypes[result.columns.index("news_volume_24h")] == pl.Int32
    assert result.dtypes[result.columns.index("has_news_24h")] == pl.Boolean
    assert result.dtypes[result.columns.index("news_sentiment_score")] == pl.Float64


def test_news_count_aggregated_correctly() -> None:
    """When news_features provided: counts aggregated per category."""
    events = _make_events_lf([12])  # event at 2025-01-02 12:00 → window [11:00, 12:00)
    # One impactful news article 30 min before event
    news = pl.DataFrame(
        {
            "published_at": [datetime(2025, 1, 2, 11, 30)],
            "l1_category": ["Sentiment"],
            "confidence": [0.9],
            "expected_direction": [1],
            "validation_status": ["accepted"],
            "is_impactful": [True],
        }
    ).lazy()
    result = compute_news(events, news).collect()

    assert result["news_count_Sentiment"][0] == 1
    assert abs(result["news_max_conf_Sentiment"][0] - 0.9) < 1e-9
    assert result["news_direction_Sentiment"][0] == 1
    assert result["news_volume_24h"][0] == 1
    assert result["has_news_24h"][0] is True
