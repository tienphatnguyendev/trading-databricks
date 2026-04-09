# TRD-45 Delta Spec: Feature Engineering

> **Linear issue:** [TRD-45](https://linear.app/aaron-solo/issue/TRD-45)
> **Master spec:** `2026-04-07-ml-pipeline-phase1a-design.md` § 4.3
> **This document covers only design decisions and implementation details NOT in the Linear issue.**

---

## Design Decisions

### Polars API: Lazy (`pl.LazyFrame`)

All `compute_*` functions accept and return `pl.LazyFrame`. Feature computation is fully vectorizable — Polars' query optimizer can fuse operations, push down predicates, and parallelize across columns.

**Exception:** `assemble_training_dataset()` calls `.collect()` at the end to materialize the final training matrix as `pl.DataFrame`.

### PIT-Safety: Enforced Internally

Each `compute_*` function computes rolling indicators on the full bar series, then joins to events on `bar_index`. Since rolling indicators at `bar_index` only use bars at or before that index, this is PIT-safe by construction. The caller does NOT need to pre-filter.

**Critical invariant:** No feature computation may reference any bar with `bar_index > event.bar_index`. This is the #1 risk in financial ML.

### News Features: Full Implementation with Zero Fallback

`compute_news()` accepts an optional `news_features: pl.LazyFrame | None` parameter. When `None` or empty, return a zero-filled LazyFrame matching `FEATURES_NEWS_SCHEMA`. This allows the pipeline to run identically with or without MAS data.

---

## Implementation Approach

### Per-Event Feature Computation Pattern

The challenge: each event has its own `bar_index`, so each event's features use a different lookback window. Two approaches:

**Approach: Cross-Join + Filter (Recommended for <1000 events)**

```python
def compute_technical(
    events: pl.LazyFrame,
    bars: pl.LazyFrame,
    *,
    config: FeatureConfig = FeatureConfig(),
) -> pl.LazyFrame:
    # Pre-compute rolling indicators on full bar series
    bars_with_indicators = bars.with_columns(
        # RSI, MACD, Bollinger, SMA ratios, log returns
        # These are rolling computations on the bar series
    )
    
    # For each event, look up the indicator value at bar_index - 1
    # (strictly before event, PIT-safe)
    return events.join(
        bars_with_indicators,
        left_on="bar_index",  # event's bar
        right_on="bar_index",
        suffix="_bar",
    ).select(
        # event key columns + feature columns
    )
```

**Key insight:** RSI, MACD, Bollinger, SMAs, etc. are rolling computations on the bar series. Compute them once for ALL bars, then look up the value at each event's `bar_index` via join.

**PIT-safety clarification:** The CUSUM event triggers AT `bar_index`, meaning the bar's close price is known at `event_ts`. Rolling indicators at `bar_index` use bars `[bar_index - window, bar_index]` — all in the past. Joining features on `bar_index` directly is PIT-safe. No need to offset to `bar_index - 1`.

**`computed_at` field:** All feature schemas include a `computed_at: TIMESTAMP` column. Set this to `datetime.utcnow()` at the time the feature computation runs (not `event_ts`). This enables feature freshness monitoring.

### Specific Feature Formulas

#### RSI-14

```python
delta = pl.col("close").diff()
gain = delta.clip(lower_bound=0).rolling_mean(14)
loss = (-delta.clip(upper_bound=0)).rolling_mean(14)
rsi = 100 - (100 / (1 + gain / loss))
```

#### MACD Signal

```python
ema_12 = pl.col("close").ewm_mean(span=12)
ema_26 = pl.col("close").ewm_mean(span=26)
macd_line = ema_12 - ema_26
macd_signal = macd_line.ewm_mean(span=9)
```

#### Bollinger %B

```python
sma_20 = pl.col("close").rolling_mean(20)
std_20 = pl.col("close").rolling_std(20)
bb_upper = sma_20 + 2 * std_20
bb_lower = sma_20 - 2 * std_20
bb_pct = (pl.col("close") - bb_lower) / (bb_upper - bb_lower)
```

#### Volatility Annualization

```python
# annualization_factor = sqrt(avg_bars_per_day * 365)
# This comes from the event row, not hardcoded
realized_vol_20 = pl.col("log_return").rolling_std(20) * annualization_factor
```

**Note:** `avg_bars_per_day` varies per event (it's a column from CUSUM events). For bar-level rolling computations, use a representative value (median across all events, or the event's own value post-join).

#### Parkinson Volatility (20 bars)

```python
# Uses high/low, not close
# Parkinson = sqrt(1/(4*N*ln2) * sum(ln(high/low)^2))
log_hl = (pl.col("high") / pl.col("low")).log().pow(2)
parkinson = (log_hl.rolling_mean(20) / (4 * math.log(2))).sqrt() * annualization_factor
```

#### Garman-Klass Volatility (20 bars)

```python
# GK = sqrt(1/N * sum(0.5*ln(H/L)^2 - (2*ln2-1)*ln(C/O)^2))
log_hl_sq = 0.5 * (pl.col("high") / pl.col("low")).log().pow(2)
log_co_sq = (2 * math.log(2) - 1) * (pl.col("close") / pl.col("open")).log().pow(2)
gk = (log_hl_sq - log_co_sq).rolling_mean(20).sqrt() * annualization_factor
```

#### OBV Slope (20 bars)

```python
# OBV = cumulative sum of signed volume
# Slope = linear regression slope of OBV over last 20 bars
obv = pl.when(pl.col("close").diff() > 0).then(pl.col("volume"))
       .when(pl.col("close").diff() < 0).then(-pl.col("volume"))
       .otherwise(0).cum_sum()
# Linear regression slope via (N*sum(x*y) - sum(x)*sum(y)) / (N*sum(x^2) - sum(x)^2)
```

### News Feature Aggregation

```python
def compute_news(
    events: pl.LazyFrame,
    news_features: pl.LazyFrame | None,
    *,
    config: FeatureConfig = FeatureConfig(),
) -> pl.LazyFrame:
    if news_features is None:
        return _zero_news_frame(events)
    
    # For each event at time t, query news in [t - 24h, t)
    # where is_impactful=True and validation_status='accepted'
    
    L1_CATEGORIES = [
        "sentiment", "defi", "supply", "exchange",
        "macro", "adoption", "security", "regulation",
    ]
    
    # Per-category aggregation:
    # news_count_{l1}: count
    # news_max_conf_{l1}: max(confidence)
    # news_direction_{l1}: mode of expected_direction
    # news_accepted_ratio_{l1}: mean(validation_status == 'accepted')
    
    # Global:
    # news_volume_24h: total count
    # news_sentiment_score: confidence-weighted direction sum
    # has_news_24h: count > 0
```

### Assembly Function

```python
def assemble_training_dataset(
    events: pl.LazyFrame,
    labels: pl.LazyFrame,
    feature_frames: list[pl.LazyFrame],
) -> pl.DataFrame:  # NOTE: returns eager DataFrame
    """PIT-safe join + cpcv_group assignment."""
    
    # 1. Join all feature frames on (event_id, symbol)
    result = events
    for ff in feature_frames:
        result = result.join(ff, on=["event_id", "symbol"], how="left")
    
    # 2. Join labels (avg_uniqueness comes from here)
    result = result.join(labels, on=["event_id", "symbol"], how="left")
    
    # 3. Assign cpcv_group (1-6, chronological quantiles)
    result = result.collect().sort("event_ts")
    n = len(result)
    result = result.with_columns(
        ((pl.arange(0, n) * 6) // n + 1).alias("cpcv_group")
    )
    
    # 4. Stamp dataset_version (ISO timestamp)
    result = result.with_columns(
        pl.lit(datetime.utcnow().isoformat()).alias("dataset_version")
    )
    
    return result
```

---

## Edge Cases

| Scenario | Behavior |
|---|---|
| Insufficient bars for longest window (50) | Feature columns = NULL for those events. Not an error — let downstream handle |
| No news data (MAS not ready) | Return zero-filled LazyFrame with correct schema |
| Event at bar_index < 50 | Technical features with window > bar_index return NULL |
| Empty events DataFrame | Return empty LazyFrame with correct schema |
| News with no impactful articles in 24h | All news features = 0 for that event |

---

## Cross-Module Dependencies

| Reads | Provides to |
|---|---|
| `FeatureConfig` from `ml/config.py` | TRD-46 (CPCV): assembled training dataset |
| `FEATURES_*_SCHEMA`, `TRAINING_DATASET_SCHEMA` from `ml/schemas.py` | TRD-48 (training): feature matrix for model training |
| CUSUM events (from TRD-43) | |
| Triple-barrier labels (from TRD-44) | |
| Dollar bars (from DE TRD-42) | |
| News causal features (from MAS TRD-6, optional) | |

---

## Test Data Recipes

### Recipe 1: Known RSI

```python
# 30 bars: 14 bars up +1%, then 14 bars down -1%, then 2 flat
# RSI after 14 up bars should be ~100, after 14 down should be ~0
```

### Recipe 2: PIT-Safety Verification

```python
# Create event at bar 50
# Set bar 50's close to a distinctive value (e.g., 99999)
# Verify that NO feature uses bar 50's close or any bar > 50
# (Features should use bars up to and including bar 50 for rolling,
#  but the rolling window ends AT bar 50, using historical data)
```

### Recipe 3: Zero News

```python
# Events with no news data → all 35 news features = 0
# Verify schema still matches FEATURES_NEWS_SCHEMA
```

### Recipe 4: Assembly Join Correctness

```python
# 3 events, each with features from all families
# Verify: final DataFrame has all columns, cpcv_group in [1,6]
# Verify: dataset_version is an ISO timestamp string
```
