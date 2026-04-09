# TRD-44 Delta Spec: Triple-Barrier Labeling

> **Linear issue:** [TRD-44](https://linear.app/aaron-solo/issue/TRD-44)
> **Master spec:** `2026-04-07-ml-pipeline-phase1a-design.md` § 4.2
> **This document covers only design decisions and implementation details NOT in the Linear issue.**

---

## Design Decisions

### Polars API: Eager (`pl.DataFrame`)

The forward-scanning loop per event is sequential. Use `pl.DataFrame` throughout.

### avg_uniqueness: Owned by This Module

This module computes `avg_uniqueness` as part of the label output. TRD-46's `sample_weights.py` is a thin wrapper that reads the pre-computed column.

**Algorithm (AFML Ch. 4 — Concurrency Matrix):**

```
1. For each event i, define its label span: [bar_index_i, vertical_bar_i]
2. Build a concurrency vector c[b] for each bar b:
     c[b] = count of events whose span contains bar b
3. For each event i:
     avg_uniqueness_i = mean(1 / c[b]) for b in [bar_index_i, vertical_bar_i]
```

This measures how "unique" each event's label window is. Events with highly overlapping windows get lower uniqueness (lower sample weight).

**Implementation note:** Build the concurrency array AFTER all events are labeled, not during the labeling loop. This is a second pass.

### Maturation Filter

Only events where `vertical_bar <= max(bars.bar_index)` are included in the output. Events that haven't matured yet (vertical barrier extends beyond available data) are **excluded entirely** — they are NOT returned with NULL labels.

Rationale: The inference pipeline handles unlabeled events separately. The training pipeline only sees matured, labeled events.

---

## Implementation Approach

### Step 1: Compute Barrier Levels (Vectorizable)

Before the loop, join events with bars to get `close` at `event.bar_index`:

```python
events_with_close = events.join(
    bars.select("bar_index", "close"),
    left_on="bar_index",
    right_on="bar_index",
)

# Compute barriers (vectorized)
events_with_close = events_with_close.with_columns(
    (pl.col("close") * (1 + 2 * pl.col("ewma_vol"))).alias("upper_level"),
    (pl.col("close") * (1 - 1 * pl.col("ewma_vol"))).alias("lower_level"),
    (pl.col("bar_index") + (pl.col("avg_bars_per_day") * config.vertical_barrier_days).cast(pl.Int64)).alias("vertical_bar"),
)
```

### Step 2: Forward-Scan Loop (Sequential)

For each event, scan bars from `bar_index + 1` to `vertical_bar`:

```python
for event in events_with_close.iter_rows(named=True):
    # Filter to matured events only
    if event["vertical_bar"] > max_bar_index:
        continue
    
    future_bars = bars.filter(
        (pl.col("bar_index") > event["bar_index"]) &
        (pl.col("bar_index") <= event["vertical_bar"])
    )
    
    label, barrier_hit, return_pct, matured_at = None, None, None, None
    
    for bar in future_bars.iter_rows(named=True):
        if bar["close"] >= event["upper_level"]:
            label, barrier_hit = 1, "upper"
            break
        elif bar["close"] <= event["lower_level"]:
            label, barrier_hit = -1, "lower"
            break
    
    if label is None:  # vertical barrier hit
        last_bar = future_bars.row(-1, named=True)
        return_pct = (last_bar["close"] - event["close"]) / event["close"]
        sigma = event["ewma_vol"]
        if abs(return_pct) < config.zero_label_threshold * sigma:
            label = 0
        else:
            label = 1 if return_pct > 0 else -1
        barrier_hit = "vertical"
```

**Optimization:** Pre-sort bars by `bar_index` once. For each event, use binary search to find the starting position rather than filtering the full DataFrame each time.

### Step 3: Concurrency Matrix (Second Pass)

```python
# After all events are labeled, compute avg_uniqueness
max_bar = max(event["vertical_bar"] for event in labeled_events)
min_bar = min(event["bar_index"] for event in labeled_events)
concurrency = np.zeros(max_bar - min_bar + 1)

for event in labeled_events:
    start = event["bar_index"] - min_bar
    end = event["vertical_bar"] - min_bar
    concurrency[start:end + 1] += 1

for event in labeled_events:
    start = event["bar_index"] - min_bar
    end = event["vertical_bar"] - min_bar
    span = concurrency[start:end + 1]
    event["avg_uniqueness"] = float(np.mean(1.0 / span))
```

---

## Edge Cases

| Scenario | Behavior |
|---|---|
| No matured events | Return empty DF with correct schema, log warning |
| Event at last bar (vertical_bar > max_bar) | Excluded (not matured) |
| Upper and lower hit on same bar | Upper checked first (profit-taking priority) |
| All events hit vertical barrier | Valid — label by return sign. Check if `zero_label_threshold` is appropriate |
| Single event (no overlap) | `avg_uniqueness = 1.0` |
| All events fully overlapping | `avg_uniqueness` approaches `1/N` where N = event count |

---

## Cross-Module Dependencies

| Reads | Provides to |
|---|---|
| `LabelConfig` from `ml/config.py` | TRD-45 (features): labels for assembly into training dataset |
| `TRIPLE_BARRIER_LABELS_SCHEMA` from `ml/schemas.py` | TRD-46 (CPCV): `avg_uniqueness` as sample weight |
| CUSUM events (from TRD-43 output) | TRD-48 (training): labels + uniqueness for model training |
| Dollar bars (from DE TRD-42 output) | |

---

## Test Data Recipes

### Recipe 1: Trending Up (Upper Hit)

```python
# Event at bar 0, close=100, ewma_vol=0.02
# Bars 1-10: prices = [101, 102, 103, 104, 105, ...]
# upper = 100 * (1 + 2*0.02) = 104.0
# Expect: label=+1, barrier="upper" at bar ~4
```

### Recipe 2: Trending Down (Lower Hit)

```python
# Event at bar 0, close=100, ewma_vol=0.02
# Bars 1-10: prices = [99, 98.5, 98, 97.5, ...]
# lower = 100 * (1 - 1*0.02) = 98.0
# Expect: label=-1, barrier="lower" at bar ~3
```

### Recipe 3: Flat Market (Vertical Expiry)

```python
# Event at bar 0, close=100, ewma_vol=0.02, avg_bars_per_day=75
# vertical_bar = 0 + 75*5 = 375
# Bars 1-375: close oscillates 99.5-100.5
# Expect: label=0, barrier="vertical" (return within 0.1 * σ)
```

### Recipe 4: Uniqueness Verification

```python
# 3 events with overlapping windows:
#   Event A: bars [0, 50]
#   Event B: bars [25, 75]
#   Event C: bars [60, 110]
# Verify: A.uniqueness < 1.0, overlap region has concurrency=2
```
