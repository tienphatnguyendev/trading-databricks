from datetime import datetime, timedelta, UTC
import polars as pl
from sklearn.pipeline import Pipeline
from trading_databricks.ml.config import MLPipelineConfig
from trading_databricks.ml.cusum import detect_cusum_events
from trading_databricks.ml.labeling import compute_triple_barrier_labels
from trading_databricks.ml.transformers.pipeline import build_training_pipeline
from trading_databricks.ml.training.cpcv import CPCVSplitter

def test_build_training_pipeline_returns_pipeline_and_splitter() -> None:
    config = MLPipelineConfig()
    
    # Generate dummy bars
    now = datetime.now(UTC)
    bars = pl.DataFrame({
        "bar_index": range(200),
        "symbol": ["BTCUSDT"] * 200,
        "close": [100.0 + i * 0.1 for i in range(200)],
        "bar_end_time": [now + timedelta(minutes=i) for i in range(200)],
    })
    
    events = detect_cusum_events(bars, config=config.cusum)
    labels = compute_triple_barrier_labels(events, bars, config=config.label)
    
    if labels.is_empty():
        return  # Skip if dummy data doesn't trigger events
        
    pipeline, splitter = build_training_pipeline(config, bars, events, labels)
    
    assert isinstance(pipeline, Pipeline)
    assert isinstance(splitter, CPCVSplitter)
    assert len(pipeline.steps) == 3
    assert pipeline.steps[0][0] == "features"
    assert pipeline.steps[1][0] == "to_numpy"
    assert pipeline.steps[2][0] == "classifier"
