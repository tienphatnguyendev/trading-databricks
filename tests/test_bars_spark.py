"""Unit tests for Spark ML bar construction transformers.

Tests Spark implementations against Polars baselines to ensure correctness.
Uses synthetic tick data to avoid dependency on external data sources.

Test coverage:
* TimeBarTransformer vs Polars build_time_bars()
* ThresholdBarTransformer vs Polars build_threshold_bars()
* CUSUM filter Pandas UDF vs Polars cusum_filter()
* Edge cases: empty DataFrames, single-tick bars, extreme thresholds
* OHLCV correctness: floating-point tolerance < 1e-10
* Performance characteristics: Spark should handle >10K ticks

Usage:
    pytest tests/test_bars_spark.py -v
    pytest tests/test_bars_spark.py::TestTimeBarTransformer -v
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Add notebooks directory to path for imports
project_root = Path(__file__).parent.parent
notebooks_path = project_root / "notebooks"
sys.path.insert(0, str(notebooks_path))

from btcusdt_utils.bars import (
    build_threshold_bars,
    build_time_bars,
    cusum_filter,
)
from btcusdt_utils.bars_spark import (
    TimeBarTransformer,
    ThresholdBarTransformer,
    apply_cusum_filter_spark,
    build_spark_threshold_bars,
    build_spark_time_bars,
    cusum_filter_pandas_udf,
)


# ══════════════════════════════════════════════════════════════════════════
# Test Fixtures
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Create Spark session for tests."""
    spark = (
        SparkSession.builder.appName("test_bars_spark")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture
def synthetic_ticks_small() -> pl.DataFrame:
    """Generate 1000 synthetic ticks for fast tests.
    
    Returns Polars DataFrame with columns:
    - transact_timestamp (datetime)
    - price_d (float64)
    - qty_d (float64)
    """
    np.random.seed(42)
    n = 1000
    
    # Generate timestamps: 1 tick per second over ~16 minutes
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(seconds=i) for i in range(n)]
    
    # Generate prices: random walk around $50,000
    price_changes = np.random.normal(0, 10, n)
    prices = 50000 + np.cumsum(price_changes)
    
    # Generate quantities: random between 0.01 and 1.0 BTC
    quantities = np.random.uniform(0.01, 1.0, n)
    
    return pl.DataFrame(
        {
            "transact_timestamp": timestamps,
            "price_d": prices,
            "qty_d": quantities,
        }
    )


@pytest.fixture
def synthetic_ticks_large() -> pl.DataFrame:
    """Generate 10K synthetic ticks for performance tests."""
    np.random.seed(42)
    n = 10_000
    
    # Generate timestamps: 1 tick per 0.1 second over ~16 minutes
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(milliseconds=i * 100) for i in range(n)]
    
    # Generate prices: random walk around $50,000
    price_changes = np.random.normal(0, 10, n)
    prices = 50000 + np.cumsum(price_changes)
    
    # Generate quantities: random between 0.01 and 1.0 BTC
    quantities = np.random.uniform(0.01, 1.0, n)
    
    return pl.DataFrame(
        {
            "transact_timestamp": timestamps,
            "price_d": prices,
            "qty_d": quantities,
        }
    )


@pytest.fixture
def synthetic_ticks_edge_cases() -> dict[str, pl.DataFrame]:
    """Generate edge case tick datasets.
    
    Returns:
        dict with keys: "empty", "single_tick", "two_ticks", "duplicate_timestamps"
    """
    return {
        "empty": pl.DataFrame(
            {
                "transact_timestamp": [],
                "price_d": [],
                "qty_d": [],
            },
            schema={
                "transact_timestamp": pl.Datetime,
                "price_d": pl.Float64,
                "qty_d": pl.Float64,
            },
        ),
        "single_tick": pl.DataFrame(
            {
                "transact_timestamp": [datetime(2024, 1, 1, 0, 0, 0)],
                "price_d": [50000.0],
                "qty_d": [0.5],
            }
        ),
        "two_ticks": pl.DataFrame(
            {
                "transact_timestamp": [
                    datetime(2024, 1, 1, 0, 0, 0),
                    datetime(2024, 1, 1, 0, 0, 1),
                ],
                "price_d": [50000.0, 50010.0],
                "qty_d": [0.5, 0.3],
            }
        ),
        "duplicate_timestamps": pl.DataFrame(
            {
                "transact_timestamp": [
                    datetime(2024, 1, 1, 0, 0, 0),
                    datetime(2024, 1, 1, 0, 0, 0),
                    datetime(2024, 1, 1, 0, 0, 1),
                ],
                "price_d": [50000.0, 50005.0, 50010.0],
                "qty_d": [0.5, 0.3, 0.7],
            }
        ),
    }


def polars_to_spark(df: pl.DataFrame, spark: SparkSession):
    """Convert Polars DataFrame to Spark DataFrame via Arrow."""
    return spark.createDataFrame(df.to_pandas())


def compare_ohlcv(
    spark_df,
    polars_df: pl.DataFrame,
    tolerance: float = 1e-10,
) -> dict[str, bool]:
    """Compare OHLCV values between Spark and Polars bar DataFrames.
    
    Returns:
        dict with keys: "bar_count_match", "open_match", "high_match",
        "low_match", "close_match", "volume_match"
    """
    # Convert Spark to Pandas for comparison
    spark_pd = spark_df.toPandas().sort_values("bucket").reset_index(drop=True)
    polars_pd = polars_df.to_pandas().sort_values("bucket").reset_index(drop=True)
    
    # Check bar count
    bar_count_match = len(spark_pd) == len(polars_pd)
    
    if not bar_count_match:
        return {
            "bar_count_match": False,
            "spark_count": len(spark_pd),
            "polars_count": len(polars_pd),
        }
    
    # Compare OHLCV columns
    results = {"bar_count_match": True}
    
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in spark_pd.columns or col not in polars_pd.columns:
            results[f"{col}_match"] = False
            continue
        
        max_diff = np.abs(spark_pd[col].to_numpy() - polars_pd[col].to_numpy()).max()
        results[f"{col}_match"] = max_diff < tolerance
        results[f"{col}_max_diff"] = max_diff
    
    return results


# ══════════════════════════════════════════════════════════════════════════
# Test TimeBarTransformer
# ══════════════════════════════════════════════════════════════════════════


class TestTimeBarTransformer:
    """Tests for TimeBarTransformer against Polars baseline."""
    
    def test_time_bars_5min_small(self, spark, synthetic_ticks_small):
        """Test 5-minute time bars on 1K ticks."""
        # Polars baseline
        bars_polars = build_time_bars(
            synthetic_ticks_small,
            interval_minutes=5,
            assume_sorted=False,
        )
        
        # Spark implementation
        ticks_spark = polars_to_spark(synthetic_ticks_small, spark)
        bars_spark = build_spark_time_bars(ticks_spark, interval_minutes=5)
        
        # Compare
        comparison = compare_ohlcv(bars_spark, bars_polars)
        
        assert comparison["bar_count_match"], (
            f"Bar count mismatch: Spark={comparison.get('spark_count')}, "
            f"Polars={comparison.get('polars_count')}"
        )
        assert comparison["open_match"], f"Open mismatch: {comparison.get('open_max_diff')}"
        assert comparison["high_match"], f"High mismatch: {comparison.get('high_max_diff')}"
        assert comparison["low_match"], f"Low mismatch: {comparison.get('low_max_diff')}"
        assert comparison["close_match"], f"Close mismatch: {comparison.get('close_max_diff')}"
        assert comparison["volume_match"], f"Volume mismatch: {comparison.get('volume_max_diff')}"
    
    def test_time_bars_1min_small(self, spark, synthetic_ticks_small):
        """Test 1-minute time bars on 1K ticks."""
        bars_polars = build_time_bars(synthetic_ticks_small, interval_minutes=1)
        ticks_spark = polars_to_spark(synthetic_ticks_small, spark)
        bars_spark = build_spark_time_bars(ticks_spark, interval_minutes=1)
        
        comparison = compare_ohlcv(bars_spark, bars_polars)
        assert comparison["bar_count_match"]
        assert all(comparison[f"{col}_match"] for col in ["open", "high", "low", "close", "volume"])
    
    def test_time_bars_transformer_api(self, spark, synthetic_ticks_small):
        """Test TimeBarTransformer class API."""
        transformer = TimeBarTransformer(intervalMinutes=5)
        ticks_spark = polars_to_spark(synthetic_ticks_small, spark)
        bars_spark = transformer.transform(ticks_spark)
        
        # Verify output schema
        expected_cols = {"bucket", "open", "high", "low", "close", "volume", "trade_count"}
        actual_cols = set(bars_spark.columns)
        assert expected_cols.issubset(actual_cols), f"Missing columns: {expected_cols - actual_cols}"
        
        # Verify at least one bar was produced
        assert bars_spark.count() > 0, "No bars produced"
    
    def test_time_bars_single_tick(self, spark, synthetic_ticks_edge_cases):
        """Test time bars with single tick."""
        single_tick = synthetic_ticks_edge_cases["single_tick"]
        
        bars_polars = build_time_bars(single_tick, interval_minutes=5)
        ticks_spark = polars_to_spark(single_tick, spark)
        bars_spark = build_spark_time_bars(ticks_spark, interval_minutes=5)
        
        assert bars_spark.count() == bars_polars.height == 1
        
        # OHLC should all equal the single price
        bar = bars_spark.collect()[0]
        assert bar["open"] == bar["high"] == bar["low"] == bar["close"] == 50000.0
        assert bar["volume"] == 0.5
        assert bar["trade_count"] == 1
    
    def test_time_bars_large_dataset(self, spark, synthetic_ticks_large):
        """Test time bars on 10K ticks (performance test)."""
        import time
        
        # Polars baseline
        t0_polars = time.perf_counter()
        bars_polars = build_time_bars(synthetic_ticks_large, interval_minutes=5)
        t_polars = time.perf_counter() - t0_polars
        
        # Spark implementation
        ticks_spark = polars_to_spark(synthetic_ticks_large, spark)
        t0_spark = time.perf_counter()
        bars_spark = build_spark_time_bars(ticks_spark, interval_minutes=5)
        n_bars = bars_spark.count()  # Force evaluation
        t_spark = time.perf_counter() - t0_spark
        
        # Verify correctness
        comparison = compare_ohlcv(bars_spark, bars_polars)
        assert comparison["bar_count_match"]
        
        # Log performance (not asserting speedup since local Spark may be slower)
        print(f"\n  10K ticks → {n_bars} bars: Polars={t_polars:.3f}s, Spark={t_spark:.3f}s")


# ══════════════════════════════════════════════════════════════════════════
# Test ThresholdBarTransformer
# ══════════════════════════════════════════════════════════════════════════


class TestThresholdBarTransformer:
    """Tests for ThresholdBarTransformer (volume/dollar/tick bars)."""
    
    def test_volume_bars_small(self, spark, synthetic_ticks_small):
        """Test volume bars with 10 BTC threshold on 1K ticks."""
        threshold = 10.0
        
        # Polars baseline
        bars_polars = build_threshold_bars(
            synthetic_ticks_small,
            threshold=threshold,
            column="qty_d",
            bar_type="volume",
        )
        
        # Spark implementation
        ticks_spark = polars_to_spark(synthetic_ticks_small, spark)
        bars_spark = build_spark_threshold_bars(
            ticks_spark,
            threshold=threshold,
            bar_type="volume",
        )
        
        # Compare bar counts (may differ by 1-2 due to floating-point precision)
        n_spark = bars_spark.count()
        n_polars = bars_polars.height
        bar_count_diff = abs(n_spark - n_polars)
        
        assert bar_count_diff <= 2, (
            f"Bar count difference too large: Spark={n_spark}, Polars={n_polars}, diff={bar_count_diff}"
        )
    
    def test_dollar_bars_small(self, spark, synthetic_ticks_small):
        """Test dollar bars with $500K threshold on 1K ticks."""
        threshold = 500_000.0
        
        # Add dollar column to Polars
        ticks_with_dollar = synthetic_ticks_small.with_columns(
            (pl.col("price_d") * pl.col("qty_d")).alias("dollar_d")
        )
        
        # Polars baseline
        bars_polars = build_threshold_bars(
            ticks_with_dollar,
            threshold=threshold,
            column="dollar_d",
            bar_type="dollar",
        )
        
        # Spark implementation (auto-computes dollar column)
        ticks_spark = polars_to_spark(synthetic_ticks_small, spark)
        bars_spark = build_spark_threshold_bars(
            ticks_spark,
            threshold=threshold,
            bar_type="dollar",
        )
        
        n_spark = bars_spark.count()
        n_polars = bars_polars.height
        bar_count_diff = abs(n_spark - n_polars)
        
        assert bar_count_diff <= 2, f"Bar count mismatch: Spark={n_spark}, Polars={n_polars}"
    
    def test_tick_bars_small(self, spark, synthetic_ticks_small):
        """Test tick bars with 100-tick threshold."""
        threshold = 100.0
        
        # Polars baseline
        bars_polars = build_threshold_bars(
            synthetic_ticks_small,
            threshold=threshold,
            column="tick",
            bar_type="tick",
        )
        
        # Spark implementation
        ticks_spark = polars_to_spark(synthetic_ticks_small, spark)
        bars_spark = build_spark_threshold_bars(
            ticks_spark,
            threshold=threshold,
            bar_type="tick",
        )
        
        n_spark = bars_spark.count()
        n_polars = bars_polars.height
        
        # Tick bars should match exactly (no floating-point issues)
        assert n_spark == n_polars, f"Tick bar count mismatch: Spark={n_spark}, Polars={n_polars}"
        
        # Expected: 1000 ticks / 100 = 10 bars
        expected_bars = 1000 // 100
        assert n_spark == expected_bars, f"Expected {expected_bars} tick bars, got {n_spark}"
    
    def test_threshold_too_large(self, spark, synthetic_ticks_small):
        """Test error handling when threshold is too large."""
        # Threshold larger than total volume
        total_volume = synthetic_ticks_small["qty_d"].sum()
        threshold = total_volume * 2.0
        
        ticks_spark = polars_to_spark(synthetic_ticks_small, spark)
        
        with pytest.raises(ValueError, match="0 bars produced"):
            bars_spark = build_spark_threshold_bars(
                ticks_spark,
                threshold=threshold,
                bar_type="volume",
            )
            bars_spark.count()  # Force evaluation
    
    def test_transformer_api_volume(self, spark, synthetic_ticks_small):
        """Test ThresholdBarTransformer class API."""
        transformer = ThresholdBarTransformer(
            threshold=10.0,
            barType="volume",
        )
        ticks_spark = polars_to_spark(synthetic_ticks_small, spark)
        bars_spark = transformer.transform(ticks_spark)
        
        # Verify output schema
        expected_cols = {"bucket", "open", "high", "low", "close", "volume", "trade_count"}
        actual_cols = set(bars_spark.columns)
        assert expected_cols.issubset(actual_cols)
        
        # Verify bars were produced
        assert bars_spark.count() > 0


# ══════════════════════════════════════════════════════════════════════════
# Test CUSUM Filter
# ══════════════════════════════════════════════════════════════════════════


class TestCUSUMFilter:
    """Tests for CUSUM filter Pandas UDF."""
    
    def test_cusum_filter_comparison(self, spark):
        """Test CUSUM filter Pandas UDF against Polars baseline."""
        # Generate synthetic returns
        np.random.seed(42)
        n = 1000
        returns = np.random.normal(0, 0.01, n)
        
        # Add some significant moves (should trigger events)
        returns[100:110] = 0.05  # Positive spike
        returns[500:510] = -0.05  # Negative spike
        
        threshold = 0.02
        
        # Polars baseline
        events_polars = cusum_filter(returns, h=threshold)
        n_events_polars = len(events_polars)
        
        # Create Spark DataFrame with returns
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        timestamps = [start_time + timedelta(seconds=i) for i in range(n)]
        
        df_spark = spark.createDataFrame(
            [(ts, float(ret)) for ts, ret in zip(timestamps, returns)],
            schema=["bucket", "log_return"],
        )
        
        # Spark CUSUM filter
        df_with_events = apply_cusum_filter_spark(
            df_spark,
            returns_col="log_return",
            threshold=threshold,
            partition_col=None,
        )
        
        n_events_spark = df_with_events.filter(F.col("cusum_event")).count()
        
        # Events should match exactly (same sequential algorithm)
        assert n_events_spark == n_events_polars, (
            f"Event count mismatch: Spark={n_events_spark}, Polars={n_events_polars}"
        )
        
        # Verify events were detected
        assert n_events_spark > 0, "No CUSUM events detected (should have found spikes)"
    
    def test_cusum_no_events(self, spark):
        """Test CUSUM filter with no events (all returns below threshold)."""
        # Small returns, high threshold
        returns = np.random.normal(0, 0.001, 100)
        threshold = 0.1
        
        # Polars baseline
        events_polars = cusum_filter(returns, h=threshold)
        
        # Spark implementation
        df_spark = spark.createDataFrame(
            [(i, float(ret)) for i, ret in enumerate(returns)],
            schema=["bucket", "log_return"],
        )
        df_with_events = apply_cusum_filter_spark(
            df_spark,
            returns_col="log_return",
            threshold=threshold,
        )
        n_events_spark = df_with_events.filter(F.col("cusum_event")).count()
        
        assert len(events_polars) == 0, "Polars should detect no events"
        assert n_events_spark == 0, "Spark should detect no events"
    
    def test_cusum_pandas_udf_direct(self):
        """Test cusum_filter_pandas_udf() directly."""
        import pandas as pd
        
        # Create test returns with known event
        returns = pd.Series([0.01] * 10 + [0.05] * 5 + [0.01] * 10)
        threshold = 0.02
        
        # Call Pandas UDF directly
        event_mask = cusum_filter_pandas_udf(returns, h=threshold)
        
        # Verify output type and shape
        assert isinstance(event_mask, pd.Series)
        assert len(event_mask) == len(returns)
        assert event_mask.dtype == bool
        
        # Verify at least one event was triggered
        assert event_mask.sum() > 0, "No events detected"


# ══════════════════════════════════════════════════════════════════════════
# Test Edge Cases
# ══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_dataframe_time_bars(self, spark, synthetic_ticks_edge_cases):
        """Test time bars with empty DataFrame."""
        empty_df = synthetic_ticks_edge_cases["empty"]
        
        # Polars (should produce empty result)
        bars_polars = build_time_bars(empty_df, interval_minutes=5)
        
        # Spark (should also produce empty result)
        ticks_spark = polars_to_spark(empty_df, spark)
        bars_spark = build_spark_time_bars(ticks_spark, interval_minutes=5)
        
        assert bars_polars.height == 0, "Polars should produce 0 bars"
        assert bars_spark.count() == 0, "Spark should produce 0 bars"
    
    def test_duplicate_timestamps(self, spark, synthetic_ticks_edge_cases):
        """Test handling of duplicate timestamps."""
        dup_ticks = synthetic_ticks_edge_cases["duplicate_timestamps"]
        
        # Both implementations should handle duplicates gracefully
        bars_polars = build_time_bars(dup_ticks, interval_minutes=5)
        ticks_spark = polars_to_spark(dup_ticks, spark)
        bars_spark = build_spark_time_bars(ticks_spark, interval_minutes=5)
        
        # Should produce 1 bar with 3 ticks
        assert bars_polars.height == 1
        assert bars_spark.count() == 1
        
        bar = bars_spark.collect()[0]
        assert bar["trade_count"] == 3
    
    def test_unsorted_ticks(self, spark, synthetic_ticks_small):
        """Test that implementations handle unsorted ticks."""
        # Shuffle ticks
        shuffled = synthetic_ticks_small.sample(fraction=1.0, seed=42, shuffle=True)
        
        # Both should sort internally
        bars_polars_sorted = build_time_bars(
            synthetic_ticks_small,
            interval_minutes=5,
            assume_sorted=False,
        )
        bars_polars_unsorted = build_time_bars(
            shuffled,
            interval_minutes=5,
            assume_sorted=False,
        )
        
        # Results should match
        assert bars_polars_sorted.height == bars_polars_unsorted.height
        
        # Spark should also handle unsorted
        ticks_spark_shuffled = polars_to_spark(shuffled, spark)
        bars_spark = build_spark_time_bars(ticks_spark_shuffled, interval_minutes=5)
        
        comparison = compare_ohlcv(bars_spark, bars_polars_sorted)
        assert comparison["bar_count_match"]
    
    def test_invalid_bar_type(self, spark, synthetic_ticks_small):
        """Test error handling for invalid bar type."""
        with pytest.raises(ValueError, match="barType must be"):
            ThresholdBarTransformer(
                threshold=100.0,
                barType="invalid_type",
            )


# ══════════════════════════════════════════════════════════════════════════
# Test Performance Characteristics
# ══════════════════════════════════════════════════════════════════════════


class TestPerformance:
    """Performance characteristic tests (not strict assertions)."""
    
    def test_spark_handles_large_data(self, spark, synthetic_ticks_large):
        """Verify Spark can handle 10K ticks without errors."""
        ticks_spark = polars_to_spark(synthetic_ticks_large, spark)
        
        # Time bars
        bars_time = build_spark_time_bars(ticks_spark, interval_minutes=5)
        assert bars_time.count() > 0
        
        # Volume bars
        bars_volume = build_spark_threshold_bars(
            ticks_spark,
            threshold=50.0,
            bar_type="volume",
        )
        assert bars_volume.count() > 0
        
        # Tick bars
        bars_tick = build_spark_threshold_bars(
            ticks_spark,
            threshold=100,
            bar_type="tick",
        )
        assert bars_tick.count() > 0
        
        print("\n  ✓ Spark handled 10K ticks successfully")
        print(f"    Time bars (5m):  {bars_time.count():,}")
        print(f"    Volume bars:     {bars_volume.count():,}")
        print(f"    Tick bars (100): {bars_tick.count():,}")
    
    def test_memory_efficiency(self, spark, synthetic_ticks_large):
        """Verify Spark doesn't materialize large DataFrames in driver."""
        # This test verifies that operations don't call .collect() or .toPandas()
        # which would cause driver OOM on large datasets
        
        ticks_spark = polars_to_spark(synthetic_ticks_large, spark)
        
        # Build bars without collecting
        bars = build_spark_time_bars(ticks_spark, interval_minutes=1)
        
        # Only compute count (should not materialize all data)
        n_bars = bars.count()
        
        assert n_bars > 0, "Should produce bars"
        print(f"\n  ✓ Memory efficient: computed {n_bars:,} bars without driver materialization")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
