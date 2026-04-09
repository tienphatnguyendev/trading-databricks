# TRD-47 Delta Spec: Bet Sizing + Position Compute

> **Linear issue:** [TRD-47](https://linear.app/aaron-solo/issue/TRD-47)
> **Master spec:** `2026-04-07-ml-pipeline-phase1a-design.md` § 5.4
> **This document covers only design decisions and implementation details NOT in the Linear issue.**

---

## Design Decisions

### Polars API: Eager (`pl.DataFrame`)

Bet sizing operates on a small number of predictions (max ~50 events per inference run). Eager mode is appropriate.

### sigma_probability: Explicit Float Parameter

The `sigma_probability` (standard deviation of meta-label probabilities from OOS calibration) is stored as an MLflow artifact at training time. The caller (inference pipeline, TRD-49) loads it and passes as a float. This module does NOT access MLflow.

### Threshold Filtering: Before Computation

Filter predictions to only those passing both thresholds BEFORE computing bet sizes. This avoids computing sizes for signals that will be discarded anyway:

```python
filtered = predictions.filter(
    (pl.col("meta_confidence") > config.meta_threshold) &
    (pl.col("primary_confidence") > config.primary_threshold)
)
```

### scipy Dependency

`scipy.stats.norm.cdf` is the only external dependency beyond Polars. Used for the Φ(z) in the AFML formula.

---

## Implementation Details

### Bet Sizing Formula (AFML Ch. 10)

```python
import polars as pl
from scipy.stats import norm
from trading_databricks.ml.config import BetSizingConfig

def compute_bet_sizes(
    predictions: pl.DataFrame,
    *,
    sigma_probability: float,
    config: BetSizingConfig = BetSizingConfig(),
) -> pl.DataFrame:
    """Compute raw and discretized bet sizes.
    
    Steps:
    1. Filter by thresholds (meta > 0.50, primary > 0.55)
    2. z_score = (meta_confidence - 0.5) / sigma_probability
    3. raw_bet_size = 2 * Phi(z_score) - 1
    4. Discretize into 5 tiers
    """
    # Step 1: Filter
    filtered = predictions.filter(
        (pl.col("meta_confidence") > config.meta_threshold) &
        (pl.col("primary_confidence") > config.primary_threshold)
    )
    
    if filtered.is_empty():
        return _empty_result(predictions.schema)
    
    # Step 2-3: Raw bet size (need Python UDF for scipy)
    z_scores = (filtered["meta_confidence"].to_numpy() - 0.5) / sigma_probability
    raw_sizes = 2.0 * norm.cdf(z_scores) - 1.0
    
    filtered = filtered.with_columns(
        pl.Series("raw_bet_size", raw_sizes),
    )
    
    # Step 4: Discretize
    filtered = filtered.with_columns(
        _discretize(pl.col("raw_bet_size"), config.tiers).alias("discretized_weight"),
    )
    
    return filtered
```

### Discretization Logic

```python
def _discretize(raw_col: pl.Expr, tiers: tuple[tuple[float, float], ...]) -> pl.Expr:
    """Map raw bet size to discretized weight using config tiers.
    
    Tiers: ((upper_bound, weight), ...)
    Example: ((0.10, 0.00), (0.25, 0.15), (0.50, 0.35), (0.75, 0.60), (1.01, 0.85))
    """
    expr = pl.lit(tiers[-1][1])  # default to highest tier
    
    for upper_bound, weight in reversed(tiers):
        expr = pl.when(raw_col < upper_bound).then(pl.lit(weight)).otherwise(expr)
    
    return expr
```

### Position Computation

```python
def compute_positions(
    signals: pl.DataFrame,
) -> pl.DataFrame:
    """Compute current portfolio state from latest signals.
    
    BTC-only Phase 1A rules:
    - Single position per symbol
    - Latest signal wins (most recent event_ts)
    - If no passing signals: side="flat", weight=0.0
    """
    if signals.is_empty():
        return pl.DataFrame({
            "symbol": ["BTCUSDT"],
            "side": ["flat"],
            "weight": [0.0],
            "confidence": [0.0],
            "signal_id": [""],
            "updated_at": [datetime.utcnow()],
        })
    
    latest = signals.sort("event_ts", descending=True).head(1)
    
    return latest.select(
        pl.col("symbol"),
        pl.when(pl.col("primary_side") == 1)
          .then(pl.lit("long"))
          .otherwise(pl.lit("flat"))  # Phase 1A: long-only
          .alias("side"),
        pl.col("discretized_weight").alias("weight"),
        pl.col("meta_confidence").alias("confidence"),
        pl.col("signal_id"),
        pl.lit(datetime.utcnow()).alias("updated_at"),
    )
```

**Phase 1A constraint:** Long-only. If `primary_side == -1`, the position is `"flat"` (we don't short). The `discretized_weight` still reflects conviction, but the side is clamped.

---

## Edge Cases

| Scenario | Behavior |
|---|---|
| No predictions pass both thresholds | Return empty DataFrame (bet_sizing), position = flat |
| sigma_probability = 0 | Guard: if `sigma_probability < 1e-10`, raise `ValueError`. Zero sigma means degenerate calibration |
| meta_confidence exactly = 0.50 | z_score = 0, raw_bet_size = 0.0, discretized = 0.00 (no position). Threshold is strictly `>`, not `>=` |
| raw_bet_size > 1.0 | Possible if z_score is very large. Clip to `[0.0, 1.0]` before discretization |
| primary_side = -1 (short signal) | Phase 1A: position = "flat". Signal is still written to `silver.signals` for audit trail |
| Multiple signals for same symbol | Latest `event_ts` wins for position. All signals written to `silver.signals` |

---

## Cross-Module Dependencies

| Reads | Provides to |
|---|---|
| `BetSizingConfig` from `ml/config.py` | TRD-49 (inference): bet-sized signals for writing |
| `SIGNALS_SCHEMA` from `ml/schemas.py` | TRD-49 (inference): position state |
| `POSITIONS_SCHEMA` from `ml/schemas.py` | |
| Model predictions (from TRD-48/49 at inference time) | |
| `sigma_probability` (from MLflow artifact, loaded by TRD-49) | |

---

## Test Data Recipes

### Recipe 1: Known z-score → Known Output

```python
# meta_confidence=0.80, sigma_probability=0.10
# z = (0.80 - 0.50) / 0.10 = 3.0
# raw = 2 * Phi(3.0) - 1 = 2 * 0.9987 - 1 = 0.9974
# discretized = 0.85 (conviction tier: [0.75, 1.00])
```

### Recipe 2: Below Threshold Filtering

```python
# meta_confidence=0.45 → filtered out (< 0.50)
# primary_confidence=0.50 → filtered out (< 0.55)
# Both below → definitely filtered out
# Verify: empty result
```

### Recipe 3: Discretization Boundary

```python
# Test each tier boundary:
# raw=0.09 → 0.00, raw=0.10 → 0.15, raw=0.24 → 0.15
# raw=0.25 → 0.35, raw=0.49 → 0.35, raw=0.50 → 0.60
# raw=0.74 → 0.60, raw=0.75 → 0.85, raw=0.99 → 0.85
```

### Recipe 4: Position with No Passing Signals

```python
# All predictions have meta_confidence < 0.50
# Verify: position = ("BTCUSDT", "flat", 0.0, 0.0, "", now)
```

### Recipe 5: Long-Only Constraint

```python
# primary_side=-1 (short), meta_confidence=0.90, primary_confidence=0.80
# Signal is computed (raw_bet_size > 0) but position.side = "flat"
```
