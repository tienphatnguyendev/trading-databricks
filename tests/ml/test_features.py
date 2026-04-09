# tests/ml/test_features.py
from __future__ import annotations

from datetime import datetime

import polars as pl

from trading_databricks.ml.features import (
    compute_news,
    compute_technical,
    compute_volatility,
    compute_volume,
)


def test_feature_functions_return_expected_columns() -> None:
    events = pl.DataFrame({
        "event_id": ["e1"],
        "symbol": ["BTCUSDT"],
        "event_ts": [datetime(2026, 1, 1)],
        "bar_index": [1],
        "avg_bars_per_day": [10.0],
    }).lazy()
    bars = pl.DataFrame({
        "bar_index": [1, 2],
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "open": [98.0, 108.0],
        "close": [100.0, 110.0],
        "high": [105.0, 115.0],
        "low": [95.0, 105.0],
        "volume": [1.0, 2.0],
        "dollar_volume": [100.0, 220.0],
        "bar_end_time": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
    }).lazy()

    assert "rsi_14" in compute_technical(events, bars).collect().columns
    assert "realized_vol_20" in compute_volatility(events, bars).collect().columns
    assert "relative_volume_20" in compute_volume(events, bars).collect().columns
    # compute_news accepts None gracefully (returns zero-filled frame)
    assert "news_sentiment_score" in compute_news(events, None).collect().columns
