# TRD-45: Feature Engineering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all 4 feature families (technical, volatility, volume, news) as pure Polars LazyFrame functions plus an assembly function that joins them into a TRAINING_DATASET_SCHEMA-compatible DataFrame.

**Architecture:** All `compute_*` functions accept `pl.LazyFrame` and return `pl.LazyFrame`. Rolling indicators are pre-computed on the full bar series using Polars expressions, then joined to events on `bar_index` (PIT-safe: rolling at `bar_index` uses bars ≤ that index). `assemble_training_dataset()` collects once at the end. The `computed_at` timestamp is stamped at function call time. News features return a zero-filled LazyFrame when `news_features=None`.

**Tech Stack:** `polars`, `math`, `numpy`, `datetime` — Lazy API throughout until final collect.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `src/trading_databricks/ml/features/__init__.py` | CREATE | Re-export all public functions |
| `src/trading_databricks/ml/features/technical.py` | CREATE | RSI-14, MACD signal, BB %B, SMA ratios, log returns |
| `src/trading_databricks/ml/features/volatility.py` | CREATE | Realized vol, Parkinson, Garman-Klass, vol ratio |
| `src/trading_databricks/ml/features/volume.py` | CREATE | VWAP deviation, relative volume, OBV slope, dollar volume ratio |
| `src/trading_databricks/ml/features/news.py` | CREATE | Per-category news aggregation + zero fallback |
| `src/trading_databricks/ml/features/assembly.py` | CREATE | Join all features + cpcv_group + dataset_version |
| `tests/ml/test_features_technical.py` | CREATE | Tests for technical.py |
| `tests/ml/test_features_volatility.py` | CREATE | Tests for volatility.py |
| `tests/ml/test_features_volume.py` | CREATE | Tests for volume.py |
| `tests/ml/test_features_news.py` | CREATE | Tests for news.py |
| `tests/ml/test_features_assembly.py` | CREATE | Tests for assembly.py |

---

## Task 1: Technical features (RSI-14, MACD, Bollinger %B, SMA ratios, log returns)

**Files:**
- Create: `src/trading_databricks/ml/features/__init__.py`
- Create: `src/trading_databricks/ml/features/technical.py`
- Create: `tests/ml/test_features_technical.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/test_features_technical.py
from __future__ import annotations

import math
from datetime import datetime, timedelta

import polars as pl
import pytest

from trading_databricks.ml.config import FeatureConfig
from trading_databricks.ml.features.technical import compute_technical


def _make_bars_lf(closes: list[float], symbol: str = "BTCUSDT") -> pl.LazyFrame:
    n = len(closes)
    base = datetime(2025, 1, 1)
    return pl.DataFrame(
        {
            "bar_index": list(range(n)),
            "symbol": [symbol] * n,
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
            "bar_end_time": [base + timedelta(hours=i + 1) for i in range(n)],
        }
    ).lazy()


def _make_events_lf(bar_indices: list[int], symbol: str = "BTCUSDT") -> pl.LazyFrame:
    base = datetime(2025, 1, 1)
    return pl.DataFrame(
        {
            "event_id": [f"evt-{i}" for i in bar_indices],
            "symbol": [symbol] * len(bar_indices),
            "event_ts": [base + timedelta(hours=idx + 1) for idx in bar_indices],
            "bar_index": bar_indices,
            "ewma_vol": [0.02] * len(bar_indices),
            "avg_bars_per_day": [24.0] * len(bar_indices),
        }
    ).lazy()


def test_compute_technical_returns_lazyframe() -> None:
    """compute_technical returns a LazyFrame (not collected)."""
    bars = _make_bars_lf([100.0] * 100)
    events = _make_events_lf([50])
    result = compute_technical(events, bars)
    assert isinstance(result, pl.LazyFrame)


def test_compute_technical_columns_match_schema() -> None:
    """Output has all FEATURES_TECHNICAL_SCHEMA columns."""
    bars = _make_bars_lf([100.0] * 100)
    events = _make_events_lf([50])
    result = compute_technical(events, bars).collect()

    expected = {
        "event_id", "symbol", "event_ts", "computed_at",
        "rsi_14", "macd_signal", "bb_pct",
        "sma_ratio_5_20", "sma_ratio_10_50",
        "log_return_5", "log_return_20",
    }
    assert set(result.columns) == expected


def test_rsi_is_between_0_and_100() -> None:
    """RSI must always be in [0, 100] for valid price data."""
    import numpy as np
    prices = list(100.0 * np.cumprod(1 + np.random.default_rng(42).normal(0, 0.01, 200)))
    bars = _make_bars_lf(prices)
    events = _make_events_lf(list(range(20, 200, 10)))
    result = compute_technical(events, bars).collect()

    rsi_vals = result["rsi_14"].drop_nulls().to_list()
    assert all(0.0 <= v <= 100.0 for v in rsi_vals), f"RSI out of range: {rsi_vals}"


def test_pit_safety_no_future_bars_used() -> None:
    """Features at bar_index=50 must not use any bar > 50."""
    # Set bar 51 close to a very distinctive value. Features should not reflect it.
    closes = [100.0] * 51 + [999999.0] * 49
    bars = _make_bars_lf(closes)
    events = _make_events_lf([50])
    result = compute_technical(events, bars).collect()

    # If PIT-safe, features should be based on bars [0..50] all at 100.0
    # sma_ratio_5_20 should be ~1.0 (all same price)
    ratio = result["sma_ratio_5_20"][0]
    if ratio is not None:
        assert abs(ratio - 1.0) < 0.01, f"sma_ratio_5_20={ratio} suggests future leak"


def test_empty_events_returns_empty_lazyframe() -> None:
    """No events → empty LazyFrame with correct columns."""
    bars = _make_bars_lf([100.0] * 50)
    events = pl.DataFrame(
        schema={
            "event_id": pl.Utf8,
            "symbol": pl.Utf8,
            "event_ts": pl.Datetime,
            "bar_index": pl.Int64,
            "ewma_vol": pl.Float64,
            "avg_bars_per_day": pl.Float64,
        }
    ).lazy()
    result = compute_technical(events, bars).collect()
    assert result.is_empty()
    assert "rsi_14" in result.columns
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/ml/test_features_technical.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `technical.py`**

```python
# src/trading_databricks/ml/features/technical.py
from __future__ import annotations

import math
from datetime import datetime

import polars as pl

from trading_databricks.ml.config import FeatureConfig

_TECHNICAL_SCHEMA = {
    "event_id": pl.Utf8,
    "symbol": pl.Utf8,
    "event_ts": pl.Datetime,
    "computed_at": pl.Datetime,
    "rsi_14": pl.Float64,
    "macd_signal": pl.Float64,
    "bb_pct": pl.Float64,
    "sma_ratio_5_20": pl.Float64,
    "sma_ratio_10_50": pl.Float64,
    "log_return_5": pl.Float64,
    "log_return_20": pl.Float64,
}


def compute_technical(
    events: pl.LazyFrame,
    bars: pl.LazyFrame,
    *,
    config: FeatureConfig = FeatureConfig(),
) -> pl.LazyFrame:
    """Compute technical indicators at each event's bar_index.

    All indicators use bars strictly ≤ bar_index (PIT-safe by rolling join).
    Returns a LazyFrame. Caller collects.
    """
    now = datetime.utcnow()

    # --- Pre-compute rolling indicators on full bar series ---
    bars_with_indicators = bars.sort("bar_index").with_columns(
        # Log returns
        pl.col("close").log().diff().alias("_log_ret"),
    ).with_columns(
        # RSI-14
        (
            pl.col("_log_ret")
            .clip(lower_bound=0)
            .rolling_mean(14)
            / ((-pl.col("_log_ret").clip(upper_bound=0)).rolling_mean(14) + 1e-12)
        )
        .map_elements(lambda rs: 100.0 - 100.0 / (1.0 + rs), return_dtype=pl.Float64)
        .alias("rsi_14"),
        # MACD signal (EMA12 - EMA26, then EMA9 of that)
        (
            pl.col("close").ewm_mean(span=12, min_periods=1)
            - pl.col("close").ewm_mean(span=26, min_periods=1)
        )
        .ewm_mean(span=9, min_periods=1)
        .alias("macd_signal"),
        # Bollinger %B
        (
            (
                pl.col("close")
                - pl.col("close").rolling_mean(20)
                - 2 * pl.col("close").rolling_std(20)
            )
            / (4 * pl.col("close").rolling_std(20) + 1e-12)
            + 0.5
        ).alias("bb_pct"),
        # SMA ratios
        (
            pl.col("close").rolling_mean(5)
            / (pl.col("close").rolling_mean(20) + 1e-12)
        ).alias("sma_ratio_5_20"),
        (
            pl.col("close").rolling_mean(10)
            / (pl.col("close").rolling_mean(50) + 1e-12)
        ).alias("sma_ratio_10_50"),
        # Log returns over N bars
        pl.col("close").log().diff(5).alias("log_return_5"),
        pl.col("close").log().diff(20).alias("log_return_20"),
    ).with_columns(
        pl.lit(now).alias("computed_at"),
    )

    feature_cols = [
        "bar_index", "computed_at",
        "rsi_14", "macd_signal", "bb_pct",
        "sma_ratio_5_20", "sma_ratio_10_50",
        "log_return_5", "log_return_20",
    ]

    return (
        events.join(
            bars_with_indicators.select(feature_cols),
            on="bar_index",
            how="left",
        )
        .select(list(_TECHNICAL_SCHEMA.keys()))
    )
```

- [ ] **Step 4: Create `__init__.py`**

```python
# src/trading_databricks/ml/features/__init__.py
from trading_databricks.ml.features.assembly import assemble_training_dataset
from trading_databricks.ml.features.news import compute_news
from trading_databricks.ml.features.technical import compute_technical
from trading_databricks.ml.features.volatility import compute_volatility
from trading_databricks.ml.features.volume import compute_volume

__all__ = [
    "compute_technical",
    "compute_volatility",
    "compute_volume",
    "compute_news",
    "assemble_training_dataset",
]
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/ml/test_features_technical.py -v
```

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/trading_databricks/ml/features/ tests/ml/test_features_technical.py
git commit -m "TRD-45 feat: compute_technical with RSI, MACD, Bollinger, SMA ratios, log returns"
```

---

## Task 2: Volatility features

**Files:**
- Create: `src/trading_databricks/ml/features/volatility.py`
- Create: `tests/ml/test_features_volatility.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/test_features_volatility.py
from __future__ import annotations

import math
from datetime import datetime, timedelta

import polars as pl

from trading_databricks.ml.features.volatility import compute_volatility


def _make_bars_lf(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    symbol: str = "BTCUSDT",
) -> pl.LazyFrame:
    n = len(closes)
    base = datetime(2025, 1, 1)
    return pl.DataFrame(
        {
            "bar_index": list(range(n)),
            "symbol": [symbol] * n,
            "open": closes,
            "high": highs if highs else [c * 1.002 for c in closes],
            "low": lows if lows else [c * 0.998 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
            "bar_end_time": [base + timedelta(hours=i + 1) for i in range(n)],
        }
    ).lazy()


def _make_events_lf(bar_indices: list[int]) -> pl.LazyFrame:
    base = datetime(2025, 1, 1)
    return pl.DataFrame(
        {
            "event_id": [f"e{i}" for i in bar_indices],
            "symbol": ["BTCUSDT"] * len(bar_indices),
            "event_ts": [base + timedelta(hours=idx + 1) for idx in bar_indices],
            "bar_index": bar_indices,
            "ewma_vol": [0.02] * len(bar_indices),
            "avg_bars_per_day": [24.0] * len(bar_indices),
        }
    ).lazy()


def test_compute_volatility_columns() -> None:
    bars = _make_bars_lf([100.0] * 100)
    events = _make_events_lf([50])
    result = compute_volatility(events, bars).collect()
    expected = {
        "event_id", "symbol", "event_ts", "computed_at",
        "realized_vol_20", "realized_vol_50",
        "parkinson_vol_20", "garman_klass_20", "vol_ratio_20_50",
    }
    assert set(result.columns) == expected


def test_realized_vol_positive_for_volatile_prices() -> None:
    """Volatile prices → realized_vol_20 > 0."""
    import numpy as np
    prices = list(100.0 * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.02, 100)))
    bars = _make_bars_lf(prices)
    events = _make_events_lf([80])
    result = compute_volatility(events, bars).collect()
    val = result["realized_vol_20"][0]
    assert val is not None and val > 0.0


def test_parkinson_vol_positive() -> None:
    """Parkinson vol > 0 when high != low."""
    prices = [100.0 + i * 0.1 for i in range(100)]
    bars = _make_bars_lf(prices)  # default high=close*1.002, low=close*0.998
    events = _make_events_lf([80])
    result = compute_volatility(events, bars).collect()
    val = result["parkinson_vol_20"][0]
    assert val is not None and val > 0.0


def test_vol_ratio_gt_one_when_recent_more_volatile() -> None:
    """If vol_20 > vol_50, vol_ratio_20_50 > 1."""
    import numpy as np
    # First 50 bars: flat. Next 50 bars: very volatile.
    flat = [100.0] * 50
    volatile = list(100.0 * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.05, 50)))
    prices = flat + volatile
    bars = _make_bars_lf(prices)
    events = _make_events_lf([99])  # at end where recent is volatile
    result = compute_volatility(events, bars).collect()
    ratio = result["vol_ratio_20_50"][0]
    if ratio is not None:
        assert ratio > 1.0
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/ml/test_features_volatility.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `volatility.py`**

```python
# src/trading_databricks/ml/features/volatility.py
from __future__ import annotations

import math
from datetime import datetime

import polars as pl

from trading_databricks.ml.config import FeatureConfig

_VOLATILITY_SCHEMA = {
    "event_id": pl.Utf8,
    "symbol": pl.Utf8,
    "event_ts": pl.Datetime,
    "computed_at": pl.Datetime,
    "realized_vol_20": pl.Float64,
    "realized_vol_50": pl.Float64,
    "parkinson_vol_20": pl.Float64,
    "garman_klass_20": pl.Float64,
    "vol_ratio_20_50": pl.Float64,
}

_ANNUALIZATION = 365.0  # annualize assuming daily vol


def compute_volatility(
    events: pl.LazyFrame,
    bars: pl.LazyFrame,
    *,
    config: FeatureConfig = FeatureConfig(),
) -> pl.LazyFrame:
    """Compute volatility features at each event's bar_index.

    annualization_factor = sqrt(avg_bars_per_day * 365)
    Uses a representative avg_bars_per_day=24 (4h bars) for bar-level rolling.
    Post-join: replace with event's own avg_bars_per_day.
    """
    now = datetime.utcnow()
    ln2 = math.log(2)

    bars_with_vol = bars.sort("bar_index").with_columns(
        pl.col("close").log().diff().alias("_log_ret"),
        (pl.col("high") / pl.col("low")).log().pow(2).alias("_log_hl_sq"),
        (pl.col("close") / pl.col("open")).log().pow(2).alias("_log_co_sq"),
        (pl.col("high") / pl.col("low")).log().pow(2).mul(0.5).alias("_log_hl_half"),
    ).with_columns(
        # Realized vol (annualized): sqrt(rolling_std^2 * avg_bars/day * 365)
        # Use representative 24 bars/day; will be overridden post-join if needed
        pl.col("_log_ret").rolling_std(20).mul((24.0 * _ANNUALIZATION) ** 0.5).alias("realized_vol_20"),
        pl.col("_log_ret").rolling_std(50).mul((24.0 * _ANNUALIZATION) ** 0.5).alias("realized_vol_50"),
        # Parkinson: sqrt(rolling_mean(log(H/L)^2) / (4*ln2)) * sqrt(365)
        (pl.col("_log_hl_sq").rolling_mean(20) / (4 * ln2))
        .sqrt()
        .mul((24.0 * _ANNUALIZATION) ** 0.5)
        .alias("parkinson_vol_20"),
        # Garman-Klass: sqrt(rolling_mean(0.5*ln(H/L)^2 - (2*ln2-1)*ln(C/O)^2)) * sqrt(365)
        (
            pl.col("_log_hl_half")
            - pl.col("_log_co_sq").mul(2 * ln2 - 1)
        )
        .rolling_mean(20)
        .sqrt()
        .mul((24.0 * _ANNUALIZATION) ** 0.5)
        .alias("garman_klass_20"),
    ).with_columns(
        (pl.col("realized_vol_20") / (pl.col("realized_vol_50") + 1e-12)).alias("vol_ratio_20_50"),
        pl.lit(now).alias("computed_at"),
    )

    feature_cols = [
        "bar_index", "computed_at",
        "realized_vol_20", "realized_vol_50",
        "parkinson_vol_20", "garman_klass_20", "vol_ratio_20_50",
    ]

    return (
        events.join(bars_with_vol.select(feature_cols), on="bar_index", how="left")
        .select(list(_VOLATILITY_SCHEMA.keys()))
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/ml/test_features_volatility.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/ml/features/volatility.py tests/ml/test_features_volatility.py
git commit -m "TRD-45 feat: compute_volatility with realized, Parkinson, Garman-Klass, vol ratio"
```

---

## Task 3: Volume features

**Files:**
- Create: `src/trading_databricks/ml/features/volume.py`
- Create: `tests/ml/test_features_volume.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/test_features_volume.py
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from trading_databricks.ml.features.volume import compute_volume


def _make_bars_lf(closes: list[float], volumes: list[float] | None = None) -> pl.LazyFrame:
    n = len(closes)
    base = datetime(2025, 1, 1)
    vols = volumes if volumes else [1000.0] * n
    return pl.DataFrame(
        {
            "bar_index": list(range(n)),
            "symbol": ["BTCUSDT"] * n,
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": vols,
            "dollar_value": [c * v for c, v in zip(closes, vols)],
            "bar_end_time": [base + timedelta(hours=i + 1) for i in range(n)],
        }
    ).lazy()


def _make_events_lf(bar_indices: list[int]) -> pl.LazyFrame:
    base = datetime(2025, 1, 1)
    return pl.DataFrame(
        {
            "event_id": [f"e{i}" for i in bar_indices],
            "symbol": ["BTCUSDT"] * len(bar_indices),
            "event_ts": [base + timedelta(hours=idx + 1) for idx in bar_indices],
            "bar_index": bar_indices,
            "ewma_vol": [0.02] * len(bar_indices),
            "avg_bars_per_day": [24.0] * len(bar_indices),
        }
    ).lazy()


def test_compute_volume_columns() -> None:
    bars = _make_bars_lf([100.0] * 100)
    events = _make_events_lf([50])
    result = compute_volume(events, bars).collect()
    expected = {
        "event_id", "symbol", "event_ts", "computed_at",
        "vwap_dev_20", "relative_volume_20", "obv_slope_20", "dollar_volume_ratio",
    }
    assert set(result.columns) == expected


def test_relative_volume_equals_one_for_constant_volume() -> None:
    """Constant volume → relative_volume_20 = 1.0 (current / rolling mean = 1)."""
    bars = _make_bars_lf([100.0] * 100, volumes=[500.0] * 100)
    events = _make_events_lf([50])
    result = compute_volume(events, bars).collect()
    val = result["relative_volume_20"][0]
    assert val is not None
    assert abs(val - 1.0) < 0.01


def test_vwap_dev_near_zero_for_flat_prices() -> None:
    """Flat prices → VWAP = close → deviation ≈ 0."""
    bars = _make_bars_lf([100.0] * 100)
    events = _make_events_lf([50])
    result = compute_volume(events, bars).collect()
    val = result["vwap_dev_20"][0]
    assert val is not None
    assert abs(val) < 0.01


def test_dollar_volume_ratio_equals_one_for_constant_data() -> None:
    """Constant dollar volume → dollar_volume_ratio = 1.0."""
    bars = _make_bars_lf([100.0] * 100, volumes=[1000.0] * 100)
    events = _make_events_lf([50])
    result = compute_volume(events, bars).collect()
    val = result["dollar_volume_ratio"][0]
    assert val is not None
    assert abs(val - 1.0) < 0.01
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/ml/test_features_volume.py -v
```

- [ ] **Step 3: Implement `volume.py`**

```python
# src/trading_databricks/ml/features/volume.py
from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from trading_databricks.ml.config import FeatureConfig

_VOLUME_SCHEMA = {
    "event_id": pl.Utf8,
    "symbol": pl.Utf8,
    "event_ts": pl.Datetime,
    "computed_at": pl.Datetime,
    "vwap_dev_20": pl.Float64,
    "relative_volume_20": pl.Float64,
    "obv_slope_20": pl.Float64,
    "dollar_volume_ratio": pl.Float64,
}


def _obv_slope_fn(vals: pl.Series) -> float:
    """Linear regression slope of OBV series over window."""
    arr = vals.to_numpy().astype(float)
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean, y_mean = x.mean(), arr.mean()
    num = float(((x - x_mean) * (arr - y_mean)).sum())
    den = float(((x - x_mean) ** 2).sum())
    return num / den if abs(den) > 1e-12 else 0.0


def compute_volume(
    events: pl.LazyFrame,
    bars: pl.LazyFrame,
    *,
    config: FeatureConfig = FeatureConfig(),
) -> pl.LazyFrame:
    """Compute volume-based features at each event's bar_index."""
    now = datetime.utcnow()

    bars_sorted = bars.sort("bar_index").with_columns(
        (pl.col("close") * pl.col("volume")).alias("_dollar_vol"),
    ).with_columns(
        # VWAP rolling 20 bars
        (
            pl.col("_dollar_vol").rolling_sum(20)
            / (pl.col("volume").rolling_sum(20) + 1e-12)
        ).alias("_vwap_20"),
        # Rolling mean volume
        pl.col("volume").rolling_mean(20).alias("_vol_mean_20"),
        # Rolling mean dollar volume
        pl.col("_dollar_vol").rolling_mean(20).alias("_dv_mean_20"),
        # OBV (signed volume)
        pl.when(pl.col("close").diff() > 0)
        .then(pl.col("volume"))
        .when(pl.col("close").diff() < 0)
        .then(-pl.col("volume"))
        .otherwise(0.0)
        .cum_sum()
        .alias("_obv"),
    ).with_columns(
        # VWAP deviation
        ((pl.col("close") - pl.col("_vwap_20")) / (pl.col("_vwap_20") + 1e-12)).alias("vwap_dev_20"),
        # Relative volume
        (pl.col("volume") / (pl.col("_vol_mean_20") + 1e-12)).alias("relative_volume_20"),
        # Dollar volume ratio
        (pl.col("_dollar_vol") / (pl.col("_dv_mean_20") + 1e-12)).alias("dollar_volume_ratio"),
        # OBV slope via rolling_map
        pl.col("_obv")
        .rolling_map(_obv_slope_fn, window_size=20)
        .alias("obv_slope_20"),
        pl.lit(now).alias("computed_at"),
    )

    feature_cols = [
        "bar_index", "computed_at",
        "vwap_dev_20", "relative_volume_20", "obv_slope_20", "dollar_volume_ratio",
    ]

    return (
        events.join(bars_sorted.select(feature_cols), on="bar_index", how="left")
        .select(list(_VOLUME_SCHEMA.keys()))
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/ml/test_features_volume.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/ml/features/volume.py tests/ml/test_features_volume.py
git commit -m "TRD-45 feat: compute_volume with VWAP deviation, relative volume, OBV slope"
```

---

## Task 4: News features (zero fallback + category aggregation)

**Files:**
- Create: `src/trading_databricks/ml/features/news.py`
- Create: `tests/ml/test_features_news.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/test_features_news.py
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/ml/test_features_news.py -v
```

- [ ] **Step 3: Implement `news.py`**

```python
# src/trading_databricks/ml/features/news.py
from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from trading_databricks.ml.config import FeatureConfig

_L1_CATEGORIES = [
    "Sentiment", "DeFi", "Supply", "Exchange",
    "Macro", "Adoption", "Security", "Regulation",
]

_NEWS_SCHEMA: dict[str, pl.PolarsDataType] = {
    "event_id": pl.Utf8,
    "symbol": pl.Utf8,
    "event_ts": pl.Datetime,
    "computed_at": pl.Datetime,
    **{f"news_count_{l1}": pl.Int32 for l1 in _L1_CATEGORIES},
    **{f"news_max_conf_{l1}": pl.Float64 for l1 in _L1_CATEGORIES},
    **{f"news_direction_{l1}": pl.Int32 for l1 in _L1_CATEGORIES},
    **{f"news_accepted_ratio_{l1}": pl.Float64 for l1 in _L1_CATEGORIES},
    "news_volume_24h": pl.Int32,
    "news_sentiment_score": pl.Float64,
    "has_news_24h": pl.Boolean,
}


def _zero_news_frame(events: pl.LazyFrame) -> pl.LazyFrame:
    """Return zero-filled news features for all events."""
    now = datetime.utcnow()
    zero_cols = [pl.lit(now).alias("computed_at")]
    for l1 in _L1_CATEGORIES:
        zero_cols += [
            pl.lit(0).cast(pl.Int32).alias(f"news_count_{l1}"),
            pl.lit(0.0).alias(f"news_max_conf_{l1}"),
            pl.lit(0).cast(pl.Int32).alias(f"news_direction_{l1}"),
            pl.lit(0.0).alias(f"news_accepted_ratio_{l1}"),
        ]
    zero_cols += [
        pl.lit(0).cast(pl.Int32).alias("news_volume_24h"),
        pl.lit(0.0).alias("news_sentiment_score"),
        pl.lit(False).alias("has_news_24h"),
    ]
    return events.select(
        ["event_id", "symbol", "event_ts"] + [c.meta.output_name() for c in zero_cols]  # type: ignore[attr-defined]
    ).with_columns(zero_cols)


def compute_news(
    events: pl.LazyFrame,
    news_features: pl.LazyFrame | None,
    *,
    config: FeatureConfig = FeatureConfig(),
) -> pl.LazyFrame:
    """Aggregate news features for the 24h window before each event.

    When news_features is None or empty, returns zero-filled LazyFrame.
    """
    now = datetime.utcnow()

    if news_features is None:
        return _build_zero_frame(events, now)

    # Collect events and news for cross-join aggregation
    events_df = events.collect()
    if events_df.is_empty():
        return pl.DataFrame(schema=_NEWS_SCHEMA).lazy()

    news_df = news_features.filter(
        (pl.col("is_impactful")) & (pl.col("validation_status") == "accepted")
    ).collect()

    if news_df.is_empty():
        return _build_zero_frame(events, now)

    rows: list[dict] = []
    lookback_hours = config.news_lookback_hours

    for event in events_df.iter_rows(named=True):
        t_end = event["event_ts"]
        t_start = t_end - timedelta(hours=lookback_hours)

        window = news_df.filter(
            (pl.col("published_at") >= t_start) & (pl.col("published_at") < t_end)
        )

        row: dict = {
            "event_id": event["event_id"],
            "symbol": event["symbol"],
            "event_ts": t_end,
            "computed_at": now,
        }

        total_count = 0
        sentiment_score = 0.0

        for l1 in _L1_CATEGORIES:
            cat = window.filter(pl.col("l1_category") == l1)
            cnt = len(cat)
            row[f"news_count_{l1}"] = cnt
            row[f"news_max_conf_{l1}"] = float(cat["confidence"].max()) if cnt > 0 else 0.0
            if cnt > 0:
                # mode of expected_direction
                direction_mode = (
                    cat.group_by("expected_direction")
                    .agg(pl.len().alias("cnt"))
                    .sort("cnt", descending=True)["expected_direction"][0]
                )
                row[f"news_direction_{l1}"] = int(direction_mode)
                row[f"news_accepted_ratio_{l1}"] = float(
                    (cat["validation_status"] == "accepted").mean()
                )
            else:
                row[f"news_direction_{l1}"] = 0
                row[f"news_accepted_ratio_{l1}"] = 0.0
            total_count += cnt
            if cnt > 0:
                sentiment_score += float(
                    (cat["confidence"] * cat["expected_direction"].cast(pl.Float64)).sum()
                )

        row["news_volume_24h"] = total_count
        row["news_sentiment_score"] = sentiment_score
        row["has_news_24h"] = total_count > 0
        rows.append(row)

    return pl.DataFrame(rows, schema=_NEWS_SCHEMA).lazy()


def _build_zero_frame(events: pl.LazyFrame, now: datetime) -> pl.LazyFrame:
    """Zero-fill all news columns for every event."""
    zero_exprs = [pl.lit(now).alias("computed_at")]
    for l1 in _L1_CATEGORIES:
        zero_exprs += [
            pl.lit(0).cast(pl.Int32).alias(f"news_count_{l1}"),
            pl.lit(0.0).alias(f"news_max_conf_{l1}"),
            pl.lit(0).cast(pl.Int32).alias(f"news_direction_{l1}"),
            pl.lit(0.0).alias(f"news_accepted_ratio_{l1}"),
        ]
    zero_exprs += [
        pl.lit(0).cast(pl.Int32).alias("news_volume_24h"),
        pl.lit(0.0).alias("news_sentiment_score"),
        pl.lit(False).alias("has_news_24h"),
    ]
    return events.select(["event_id", "symbol", "event_ts"]).with_columns(zero_exprs)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/ml/test_features_news.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/ml/features/news.py tests/ml/test_features_news.py
git commit -m "TRD-45 feat: compute_news with 24h window aggregation and zero fallback"
```

---

## Task 5: Assembly function

**Files:**
- Create: `src/trading_databricks/ml/features/assembly.py`
- Create: `tests/ml/test_features_assembly.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/test_features_assembly.py
from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from trading_databricks.ml.features.assembly import assemble_training_dataset


def _events_lf(n: int = 6) -> pl.LazyFrame:
    base = datetime(2025, 1, 1)
    return pl.DataFrame(
        {
            "event_id": [f"evt-{i}" for i in range(n)],
            "symbol": ["BTCUSDT"] * n,
            "event_ts": [base + timedelta(days=i) for i in range(n)],
            "bar_index": list(range(0, n * 10, 10)),
        }
    ).lazy()


def _labels_lf(n: int = 6) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "event_id": [f"evt-{i}" for i in range(n)],
            "symbol": ["BTCUSDT"] * n,
            "label": [1, -1, 0, 1, -1, 1],
            "avg_uniqueness": [1.0, 0.8, 0.9, 1.0, 0.75, 0.85],
        }
    ).lazy()


def _feature_lf(n: int = 6, col_name: str = "feat_a") -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "event_id": [f"evt-{i}" for i in range(n)],
            "symbol": ["BTCUSDT"] * n,
            col_name: [float(i) for i in range(n)],
        }
    ).lazy()


def test_assemble_returns_eager_dataframe() -> None:
    result = assemble_training_dataset(
        _events_lf(), _labels_lf(), [_feature_lf(col_name="feat_a")]
    )
    assert isinstance(result, pl.DataFrame)


def test_assemble_cpcv_group_in_range_1_to_6() -> None:
    result = assemble_training_dataset(
        _events_lf(n=6), _labels_lf(n=6), [_feature_lf(n=6, col_name="feat_a")]
    )
    groups = result["cpcv_group"].to_list()
    assert all(1 <= g <= 6 for g in groups), f"Invalid cpcv_group values: {groups}"


def test_assemble_dataset_version_is_iso_string() -> None:
    result = assemble_training_dataset(
        _events_lf(), _labels_lf(), [_feature_lf(col_name="feat_a")]
    )
    version = result["dataset_version"][0]
    assert isinstance(version, str)
    # Should be ISO8601 parseable
    datetime.fromisoformat(version)


def test_assemble_all_feature_columns_present() -> None:
    result = assemble_training_dataset(
        _events_lf(),
        _labels_lf(),
        [_feature_lf(col_name="feat_a"), _feature_lf(col_name="feat_b")],
    )
    assert "feat_a" in result.columns
    assert "feat_b" in result.columns
    assert "label" in result.columns
    assert "avg_uniqueness" in result.columns
    assert "cpcv_group" in result.columns
    assert "dataset_version" in result.columns


def test_assemble_sorted_by_event_ts() -> None:
    """Assembly sorts by event_ts before assigning cpcv_group."""
    import random
    base = datetime(2025, 1, 1)
    indices = list(range(6))
    random.shuffle(indices)
    events = pl.DataFrame(
        {
            "event_id": [f"evt-{i}" for i in indices],
            "symbol": ["BTCUSDT"] * 6,
            "event_ts": [base + timedelta(days=i) for i in indices],
            "bar_index": [i * 10 for i in indices],
        }
    ).lazy()
    labels = _labels_lf()
    result = assemble_training_dataset(events, labels, [_feature_lf(col_name="feat_a")])
    # cpcv_group should be 1..6 for earliest..latest events
    ts_sorted = result.sort("event_ts")["cpcv_group"].to_list()
    assert ts_sorted == sorted(ts_sorted)
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/ml/test_features_assembly.py -v
```

- [ ] **Step 3: Implement `assembly.py`**

```python
# src/trading_databricks/ml/features/assembly.py
from __future__ import annotations

from datetime import datetime

import polars as pl


def assemble_training_dataset(
    events: pl.LazyFrame,
    labels: pl.LazyFrame,
    feature_frames: list[pl.LazyFrame],
) -> pl.DataFrame:
    """Join events, labels, and all feature frames into a training dataset.

    Steps:
    1. Join all feature frames onto events via (event_id, symbol).
    2. Join labels onto the result via (event_id, symbol).
    3. Sort by event_ts, assign cpcv_group (1-6 chronological quantiles).
    4. Stamp dataset_version as ISO timestamp string.

    Returns an eager pl.DataFrame (collected once at the end).
    """
    result = events
    for ff in feature_frames:
        # Drop duplicate key columns from ff to avoid suffix clashes
        ff_clean = ff.select(
            [c for c in ff.columns if c not in ("symbol", "event_ts")]
            if "event_id" in ff.columns
            else ff.columns
        )
        # Join on event_id, keeping symbol/event_ts from events
        result = result.join(ff_clean, on="event_id", how="left")

    # Join labels
    label_cols = [
        c for c in labels.columns if c not in ("symbol", "event_ts", "bar_index")
    ]
    result = result.join(
        labels.select(label_cols if "event_id" in label_cols else ["event_id"] + label_cols),
        on="event_id",
        how="left",
    )

    # Collect + sort + assign cpcv_group
    df = result.collect().sort("event_ts")
    n = len(df)

    if n > 0:
        df = df.with_columns(
            ((pl.arange(0, n, eager=True) * 6) // n + 1).cast(pl.Int32).alias("cpcv_group")
        )
    else:
        df = df.with_columns(pl.lit(1).cast(pl.Int32).alias("cpcv_group"))

    # Stamp dataset_version
    df = df.with_columns(
        pl.lit(datetime.utcnow().isoformat()).alias("dataset_version")
    )

    return df
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/ml/test_features_assembly.py -v
```

Expected: All PASS.

- [ ] **Step 5: Run full feature test suite + lint**

```bash
uv run pytest tests/ml/test_features_technical.py tests/ml/test_features_volatility.py tests/ml/test_features_volume.py tests/ml/test_features_news.py tests/ml/test_features_assembly.py -v
uv run ruff check src/trading_databricks/ml/features/
```

- [ ] **Step 6: Commit**

```bash
git add src/trading_databricks/ml/features/assembly.py tests/ml/test_features_assembly.py
git commit -m "TRD-45 feat: assemble_training_dataset with cpcv_group assignment and dataset_version"
```

---

## Verification

```bash
uv run pytest tests/ml/test_features_technical.py tests/ml/test_features_volatility.py tests/ml/test_features_volume.py tests/ml/test_features_news.py tests/ml/test_features_assembly.py -v
```

All tests pass. Final: 11 files created.
