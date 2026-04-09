import polars as pl
from datetime import datetime, timedelta, UTC
from trading_databricks.ml.config import FeatureConfig
from trading_databricks.ml.transformers.feature_union import PolarsFeatureUnion
from trading_databricks.ml.transformers.feature_transformers import TechnicalFeatureTransformer

def test_features_do_not_use_future_bars():
    """Verify that features computed at time T do not incorporate data from T+1."""
    now = datetime.now(UTC)
    
    # Create 50 bars
    bars = pl.DataFrame({
        "bar_index": range(50),
        "symbol": ["BTCUSDT"] * 50,
        "close": [100.0 + i for i in range(50)],
        "bar_end_time": [now + timedelta(minutes=i) for i in range(50)],
    })
    
    # Create an event at bar 25
    event_ts = now + timedelta(minutes=25)
    events = pl.DataFrame({
        "event_id": ["event_1"],
        "event_ts": [event_ts],
        "bar_index": [25],
        "symbol": ["BTCUSDT"]
    })
    
    # Setup transformer
    config = FeatureConfig(technical_windows=(5,))
    union = PolarsFeatureUnion([
        ("tech", TechnicalFeatureTransformer(config))
    ])
    
    # Scenario A: Fit with ALL bars (simulating a leakage bug where fit sees future data)
    union.fit(events, y=None, bars=bars)
    features_all = union.transform(events)
    
    # Scenario B: Fit with only bars up to the event
    past_bars = bars.filter(pl.col("bar_end_time") <= event_ts)
    union.fit(events, y=None, bars=past_bars)
    features_past = union.transform(events)
    
    # The output MUST be identical. If Scenario A used future data to calculate
    # moving averages, it would differ from Scenario B.
    from polars.testing import assert_frame_equal
    assert_frame_equal(
        features_all.drop(["computed_at"]), 
        features_past.drop(["computed_at"])
    )
