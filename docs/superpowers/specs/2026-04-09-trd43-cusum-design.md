# TRD-43 Delta Spec: CUSUM Event Detection

> **Linear issue:** [TRD-43](https://linear.app/aaron-solo/issue/TRD-43)
> **Master spec:** `2026-04-07-ml-pipeline-phase1a-design.md` § 4.1
> **This document covers only design decisions and implementation details NOT in the Linear issue.**

---

## Design Decisions

### Polars API: Eager (`pl.DataFrame`)

The CUSUM accumulator (`S+`, `S-`) is inherently sequential — each bar's value depends on the previous. This cannot be vectorized. Use `pl.DataFrame` (not LazyFrame) throughout.

**However**, the pre-processing steps (log returns, EWMA) CAN be vectorized using Polars expressions before entering the Python loop.

### Warm-Up Handling

If the input DataFrame has fewer bars than `config.lookback_days × avg_bars_per_day` (estimated from the first/last bar timestamps), return an **empty DataFrame** with the correct `CUSUM_EVENTS_SCHEMA` column names/types, and log a warning via `logging.getLogger(__name__).warning(...)`.

Rationale: Early pipeline runs (first few days of backfill) should not fail — they just produce zero events until enough data accumulates.

### Threshold Computation: Rolling 30-Day Window

The threshold is NOT static. It must be recomputed for each bar using a **rolling lookback**:

```
For bar at index i:
  lookback_bars = bars in [i - lookback_window, i)
  where lookback_window = avg_bars_per_day × config.lookback_days
  
  ewma_vol = EWMA(log_returns[lookback_bars], lambda=0.94)
  avg_bars_per_day = len(lookback_bars) / calendar_days_in_window
  threshold = config.threshold_multiplier × ewma_vol × sqrt(avg_bars_per_day)
```

The EWMA itself is rolling (exponentially weighted), so it naturally adapts. The `avg_bars_per_day` must also be recomputed per-bar from the lookback window's bar timestamps.

---

## Implementation Approach

### Step 1: Vectorized Pre-Processing

```python
# Vectorize log returns
bars = bars.sort("bar_index")
bars = bars.with_columns(
    pl.col("close").log().diff().alias("log_return"),
)
```

### Step 2: EWMA Volatility (Vectorizable)

Polars does not have a built-in EWMA. Implement via `ewm_mean` on squared log returns:

```python
bars = bars.with_columns(
    pl.col("log_return").pow(2)
      .ewm_mean(alpha=1 - config.ewma_lambda, min_periods=1)
      .sqrt()
      .alias("ewma_vol"),
)
```

### Step 3: avg_bars_per_day (Vectorizable)

Compute from bar timestamps using a rolling window. For each bar at index `i`:

```python
# Convert bar_end_time to ordinal days for calendar span computation
bars = bars.with_columns(
    pl.col("bar_end_time").dt.epoch("d").alias("_day_ordinal"),
)

# For each bar, compute avg_bars_per_day over the lookback window
# lookback_size = estimated bars in lookback_days (start with 75 * lookback_days, refine)
lookback_size = 75 * config.lookback_days  # initial estimate, 75 bars/day typical

bars = bars.with_columns(
    # Count of bars in rolling window (constant = lookback_size, but handles edges)
    pl.lit(lookback_size).alias("_lookback_bar_count"),
    # Calendar days spanned by that window
    (pl.col("_day_ordinal") - pl.col("_day_ordinal").shift(lookback_size))
      .alias("_calendar_days"),
)

bars = bars.with_columns(
    (pl.col("_lookback_bar_count") / pl.col("_calendar_days").clip(lower_bound=1))
      .alias("avg_bars_per_day"),
)

# Compute threshold
bars = bars.with_columns(
    (config.threshold_multiplier * pl.col("ewma_vol") * pl.col("avg_bars_per_day").sqrt())
      .alias("threshold"),
)
```

**Simplification option:** If per-bar recomputation is too expensive, compute `avg_bars_per_day` once from the full dataset (`total_bars / calendar_days`) and use as constant. The rolling approach is more accurate for non-stationary bar production rates.

### Step 4: CUSUM Loop (Sequential Python)

```python
events = []
s_pos, s_neg = 0.0, 0.0
last_event_idx = -config.min_inter_event_bars  # allow first event

for row in bars.iter_rows(named=True):
    if row["log_return"] is None:
        continue
    
    lr = row["log_return"]
    threshold = row["threshold"]  # precomputed in step 2-3
    
    s_pos = max(0.0, s_pos + lr)
    s_neg = min(0.0, s_neg + lr)
    
    spacing_ok = (row["bar_index"] - last_event_idx) >= config.min_inter_event_bars
    
    if s_pos > threshold and spacing_ok:
        events.append({...side: "up", cusum_value: s_pos...})
        s_pos, s_neg = 0.0, 0.0  # reset BOTH
        last_event_idx = row["bar_index"]
    elif abs(s_neg) > threshold and spacing_ok:
        events.append({...side: "down", cusum_value: abs(s_neg)...})
        s_pos, s_neg = 0.0, 0.0  # reset BOTH
        last_event_idx = row["bar_index"]
```

Key: Both accumulators reset to zero on ANY event (symmetric reset).

---

## Edge Cases

| Scenario | Behavior |
|---|---|
| < 30 days of bars | Return empty DF, log warning |
| All prices identical (vol = 0) | threshold = 0, every bar triggers → min-spacing acts as rate limiter. Add guard: if `ewma_vol < 1e-10`, skip bar |
| Single massive price jump | S+ or S- spikes immediately. 5-bar spacing prevents cascade |
| NaN/None in close prices | Skip row in loop (do not update accumulators) |
| Bars not sorted | Sort by `bar_index` at function entry |

---

## Cross-Module Dependencies

| Reads | Provides to |
|---|---|
| `CUSUMConfig` from `ml/config.py` | TRD-44 (labeling): `silver.cusum_events` as input |
| `CUSUM_EVENTS_SCHEMA` from `ml/schemas.py` | TRD-45 (features): events as feature anchor points |
| `CUSUM_EVENT_STATS_SCHEMA` from `ml/schemas.py` | TRD-46 (CPCV): events for sample weight computation |

---

## Test Data Recipes

### Recipe 1: Step Function (Deterministic)

```python
# 200 bars: flat at 100.0 for 100 bars, then jump to 110.0
# Expect: exactly 1 CUSUM event at bar ~101, side="up"
closes = [100.0] * 100 + [110.0] * 100
```

### Recipe 2: Trending (Known Event Count)

```python
# 500 bars: price walks up 0.1% per bar
# With known vol, compute expected threshold and verify event count
import numpy as np
closes = 100.0 * np.cumprod(1 + np.full(500, 0.001))
```

### Recipe 3: Flat Market (Zero Events)

```python
# 300 bars: all close = 50000.0 (zero returns)
# Expect: 0 events
```

### Recipe 4: Flash Crash (Spacing Enforcement)

```python
# 100 bars: normal, then 10 bars of -2% each, then recovery
# Expect: events spaced ≥ 5 bars apart despite rapid moves
```
