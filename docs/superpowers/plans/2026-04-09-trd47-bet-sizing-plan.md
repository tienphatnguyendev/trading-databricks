# TRD-47: Bet Sizing + Position Compute — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `compute_bet_sizes()` (threshold filter → z-score → AFML Ch. 10 Φ formula → discretization) and `compute_positions()` (latest signal wins, long-only Phase 1A) as pure Polars functions.

**Architecture:** Eager Polars (`pl.DataFrame`) throughout. Threshold filter runs first (discard before computing). Z-scores computed via numpy + `scipy.stats.norm.cdf`. Discretization uses a Polars `when/then` chain over config tiers. `compute_positions()` takes the sized signals DataFrame and returns a single-row position per symbol. `sigma_probability` is passed as a float by the caller (loaded from MLflow artifact by TRD-49) — this module never accesses MLflow.

**Tech Stack:** `polars`, `scipy.stats.norm`, `numpy`, `datetime`.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `src/trading_databricks/ml/inference/__init__.py` | CREATE | Re-export `compute_bet_sizes`, `compute_positions` |
| `src/trading_databricks/ml/inference/bet_sizing.py` | CREATE | `compute_bet_sizes()` |
| `src/trading_databricks/ml/inference/position.py` | CREATE | `compute_positions()` |
| `tests/ml/test_bet_sizing.py` | CREATE | Tests for bet_sizing.py |
| `tests/ml/test_position.py` | CREATE | Tests for position.py |

---

## Task 1: Threshold filtering + z-score + raw bet size

**Files:**
- Create: `src/trading_databricks/ml/inference/__init__.py`
- Create: `src/trading_databricks/ml/inference/bet_sizing.py`
- Create: `tests/ml/test_bet_sizing.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/test_bet_sizing.py
from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from trading_databricks.ml.config import BetSizingConfig
from trading_databricks.ml.inference.bet_sizing import compute_bet_sizes


def _make_predictions(
    meta_confidences: list[float],
    primary_confidences: list[float],
    primary_sides: list[int] | None = None,
) -> pl.DataFrame:
    n = len(meta_confidences)
    sides = primary_sides if primary_sides else [1] * n
    return pl.DataFrame(
        {
            "signal_id": [f"sig-{i}" for i in range(n)],
            "event_id": [f"evt-{i}" for i in range(n)],
            "symbol": ["BTCUSDT"] * n,
            "event_ts": [datetime(2025, 1, 1)] * n,
            "primary_side": sides,
            "primary_confidence": primary_confidences,
            "meta_label": [1] * n,
            "meta_confidence": meta_confidences,
            "primary_model_version": [1] * n,
            "meta_model_version": [1] * n,
        }
    )


def test_known_z_score_produces_expected_output() -> None:
    """meta=0.80, sigma=0.10 → z=3.0, raw=0.9974, discretized=0.85."""
    preds = _make_predictions([0.80], [0.60])
    result = compute_bet_sizes(preds, sigma_probability=0.10)

    assert len(result) == 1
    raw = result["raw_bet_size"][0]
    disc = result["discretized_weight"][0]
    assert abs(raw - 0.9974) < 0.001, f"raw_bet_size={raw}"
    assert abs(disc - 0.85) < 1e-9, f"discretized_weight={disc}"


def test_below_meta_threshold_filtered_out() -> None:
    """meta_confidence=0.45 (< 0.50) → filtered out → empty result."""
    preds = _make_predictions([0.45], [0.60])
    result = compute_bet_sizes(preds, sigma_probability=0.10)
    assert result.is_empty()


def test_below_primary_threshold_filtered_out() -> None:
    """primary_confidence=0.50 (< 0.55) → filtered out → empty result."""
    preds = _make_predictions([0.75], [0.50])
    result = compute_bet_sizes(preds, sigma_probability=0.10)
    assert result.is_empty()


def test_both_thresholds_must_pass() -> None:
    """Only predictions passing BOTH thresholds are included."""
    preds = _make_predictions(
        meta_confidences=[0.80, 0.45, 0.75],
        primary_confidences=[0.60, 0.70, 0.50],
    )
    result = compute_bet_sizes(preds, sigma_probability=0.10)
    assert len(result) == 1  # only first row passes both
    assert result["signal_id"][0] == "sig-0"


def test_sigma_probability_zero_raises_value_error() -> None:
    """sigma_probability=0.0 → ValueError (degenerate calibration)."""
    preds = _make_predictions([0.80], [0.60])
    with pytest.raises(ValueError, match="sigma_probability"):
        compute_bet_sizes(preds, sigma_probability=0.0)


def test_raw_bet_size_clipped_to_0_1() -> None:
    """Very high z-score → raw_bet_size clipped at 1.0."""
    # meta=0.999, sigma=0.001 → z≈499 → Phi≈1 → raw=1.0
    preds = _make_predictions([0.999], [0.60])
    result = compute_bet_sizes(preds, sigma_probability=0.001)
    assert len(result) == 1
    assert result["raw_bet_size"][0] <= 1.0
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/ml/test_bet_sizing.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `bet_sizing.py`**

```python
# src/trading_databricks/ml/inference/bet_sizing.py
from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
from scipy.stats import norm

from trading_databricks.ml.config import BetSizingConfig

_BET_SIZING_COLUMNS = [
    "signal_id", "event_id", "symbol", "event_ts",
    "primary_side", "primary_confidence",
    "meta_label", "meta_confidence",
    "raw_bet_size", "discretized_weight",
    "primary_model_version", "meta_model_version",
]


def _empty_result() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "signal_id": pl.Utf8,
            "event_id": pl.Utf8,
            "symbol": pl.Utf8,
            "event_ts": pl.Datetime,
            "primary_side": pl.Int32,
            "primary_confidence": pl.Float64,
            "meta_label": pl.Int32,
            "meta_confidence": pl.Float64,
            "raw_bet_size": pl.Float64,
            "discretized_weight": pl.Float64,
            "primary_model_version": pl.Int32,
            "meta_model_version": pl.Int32,
        }
    )


def _discretize_expr(
    raw_col: pl.Expr, tiers: tuple[tuple[float, float], ...]
) -> pl.Expr:
    """Map raw bet size to discretized weight using config tiers.

    Tiers: ((upper_bound, weight), ...) sorted ascending by upper_bound.
    Example default: ((0.10, 0.00), (0.25, 0.15), (0.50, 0.35), (0.75, 0.60), (1.01, 0.85))
    Last tier acts as default (anything >= previous threshold).
    """
    # Start from the highest tier as default
    expr = pl.lit(tiers[-1][1])
    # Iterate in reverse: if raw < upper_bound, use this weight
    for upper_bound, weight in reversed(tiers):
        expr = pl.when(raw_col < upper_bound).then(pl.lit(weight)).otherwise(expr)
    return expr


def compute_bet_sizes(
    predictions: pl.DataFrame,
    *,
    sigma_probability: float,
    config: BetSizingConfig = BetSizingConfig(),
) -> pl.DataFrame:
    """Compute raw and discretized bet sizes for predictions passing both thresholds.

    Steps:
    1. Validate sigma_probability > 0.
    2. Filter by meta_threshold (strictly >) and primary_threshold (strictly >).
    3. z_score = (meta_confidence - 0.5) / sigma_probability
    4. raw_bet_size = clip(2 * Phi(z_score) - 1, 0, 1)
    5. Discretize raw_bet_size into 5 tiers from config.

    Args:
        predictions: DataFrame with columns: signal_id, event_id, symbol, event_ts,
                     primary_side, primary_confidence, meta_label, meta_confidence,
                     primary_model_version, meta_model_version.
        sigma_probability: Standard deviation of meta-label probabilities from OOS
                           calibration. Loaded from MLflow artifact by TRD-49. Must > 0.
        config: BetSizingConfig with thresholds and discretization tiers.

    Returns:
        pl.DataFrame with raw_bet_size and discretized_weight columns added.
        Empty if no predictions pass both thresholds.
    """
    if sigma_probability < 1e-10:
        raise ValueError(
            f"sigma_probability={sigma_probability} is too small (< 1e-10). "
            "Zero sigma indicates degenerate calibration."
        )

    # Step 1: Filter by both thresholds (strictly greater than)
    filtered = predictions.filter(
        (pl.col("meta_confidence") > config.meta_threshold)
        & (pl.col("primary_confidence") > config.primary_threshold)
    )

    if filtered.is_empty():
        return _empty_result()

    # Step 2: Compute z-scores and raw bet sizes via numpy/scipy
    z_scores = (filtered["meta_confidence"].to_numpy() - 0.5) / sigma_probability
    raw_sizes = np.clip(2.0 * norm.cdf(z_scores) - 1.0, 0.0, 1.0)

    # Step 3: Attach raw bet size and discretize
    result = filtered.with_columns(
        pl.Series("raw_bet_size", raw_sizes, dtype=pl.Float64),
    ).with_columns(
        _discretize_expr(pl.col("raw_bet_size"), config.tiers).alias("discretized_weight"),
    )

    return result.select(_BET_SIZING_COLUMNS)
```

- [ ] **Step 4: Create `__init__.py`**

```python
# src/trading_databricks/ml/inference/__init__.py
from trading_databricks.ml.inference.bet_sizing import compute_bet_sizes
from trading_databricks.ml.inference.position import compute_positions

__all__ = ["compute_bet_sizes", "compute_positions"]
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/ml/test_bet_sizing.py -v
```

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/trading_databricks/ml/inference/ tests/ml/test_bet_sizing.py
git commit -m "TRD-47 feat: compute_bet_sizes with threshold filter, z-score, discretization"
```

---

## Task 2: Discretization tier boundaries

**Files:**
- Modify: `tests/ml/test_bet_sizing.py`

- [ ] **Step 1: Write tier boundary tests**

Add to `tests/ml/test_bet_sizing.py`:

```python
def test_discretization_tier_boundaries() -> None:
    """Verify discretization at each tier boundary.

    Default tiers: ((0.10, 0.00), (0.25, 0.15), (0.50, 0.35), (0.75, 0.60), (1.01, 0.85))
    raw < 0.10 → 0.00
    0.10 <= raw < 0.25 → 0.15
    0.25 <= raw < 0.50 → 0.35
    0.50 <= raw < 0.75 → 0.60
    0.75 <= raw < 1.01 → 0.85
    """
    config = BetSizingConfig()
    # We'll directly test the output for different sigma values
    # that produce known z-scores and raw bet sizes

    # To get raw=0.09 < 0.10 → 0.00
    # raw = 2*Phi(z)-1 = 0.09 → Phi(z) = 0.545 → z ≈ 0.114
    # z = (meta - 0.5) / sigma = 0.114 → meta = 0.5 + 0.114 * sigma
    # Use sigma=0.10: meta = 0.5 + 0.0114 = 0.5114 (but 0.5114 < 0.5 + threshold
    # just above meta_threshold=0.50), primary=0.60

    test_cases = [
        # (meta_conf, primary_conf, sigma, expected_discretized)
        (0.5057, 0.60, 0.10, 0.00),   # z≈0.057, raw≈0.046 → tier [0,0.10) → 0.00
        (0.56,   0.60, 0.10, 0.15),   # z=0.6, raw=2*Phi(0.6)-1=2*0.726-1=0.452... wait
    ]
    # Compute expected analytically
    from scipy.stats import norm as _norm

    # tier breakpoints are raw values: 0.10, 0.25, 0.50, 0.75
    # raw = 2*Phi(z)-1; to get raw=0.05: z = Phi^-1(0.525) ≈ 0.063
    # z=(meta-0.5)/sigma → meta = 0.5 + z*sigma
    sigma = 0.10
    tier_tests = [
        (0.05, 0.00),   # raw<0.10 → 0.00
        (0.18, 0.15),   # 0.10 <= raw < 0.25 → 0.15
        (0.40, 0.35),   # 0.25 <= raw < 0.50 → 0.35
        (0.60, 0.60),   # 0.50 <= raw < 0.75 → 0.60
        (0.80, 0.85),   # 0.75 <= raw → 0.85
    ]

    for raw_target, expected_disc in tier_tests:
        # Invert raw = 2*Phi(z)-1 to get z
        z = float(_norm.ppf((raw_target + 1.0) / 2.0))
        meta = 0.5 + z * sigma
        meta = max(meta, 0.51)  # ensure it passes meta_threshold > 0.50

        preds = _make_predictions([meta], [0.60])
        result = compute_bet_sizes(preds, sigma_probability=sigma, config=config)

        if result.is_empty():
            continue  # meta barely above threshold; skip this tier

        disc = result["discretized_weight"][0]
        raw_got = result["raw_bet_size"][0]
        assert abs(disc - expected_disc) < 1e-9, (
            f"raw_target={raw_target:.2f}, raw_got={raw_got:.4f}, "
            f"expected_disc={expected_disc}, got={disc}"
        )
```

- [ ] **Step 2: Run**

```bash
uv run pytest tests/ml/test_bet_sizing.py::test_discretization_tier_boundaries -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/ml/test_bet_sizing.py
git commit -m "TRD-47 test: discretization tier boundary verification"
```

---

## Task 3: `compute_positions`

**Files:**
- Create: `src/trading_databricks/ml/inference/position.py`
- Create: `tests/ml/test_position.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/test_position.py
from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from trading_databricks.ml.inference.position import compute_positions


def _make_signals(
    meta_confidences: list[float],
    primary_sides: list[int],
    discretized_weights: list[float],
    event_ts_offsets_h: list[int] | None = None,
) -> pl.DataFrame:
    n = len(meta_confidences)
    base = datetime(2025, 1, 1)
    offsets = event_ts_offsets_h if event_ts_offsets_h else list(range(n))
    return pl.DataFrame(
        {
            "signal_id": [f"sig-{i}" for i in range(n)],
            "event_id": [f"evt-{i}" for i in range(n)],
            "symbol": ["BTCUSDT"] * n,
            "event_ts": [base + timedelta(hours=h) for h in offsets],
            "primary_side": primary_sides,
            "primary_confidence": [0.65] * n,
            "meta_label": [1] * n,
            "meta_confidence": meta_confidences,
            "raw_bet_size": [0.80] * n,
            "discretized_weight": discretized_weights,
            "primary_model_version": [1] * n,
            "meta_model_version": [1] * n,
        }
    )


def test_flat_position_when_no_signals() -> None:
    """Empty signals DataFrame → position side='flat', weight=0.0."""
    empty_signals = pl.DataFrame(
        schema={
            "signal_id": pl.Utf8,
            "event_id": pl.Utf8,
            "symbol": pl.Utf8,
            "event_ts": pl.Datetime,
            "primary_side": pl.Int32,
            "primary_confidence": pl.Float64,
            "meta_label": pl.Int32,
            "meta_confidence": pl.Float64,
            "raw_bet_size": pl.Float64,
            "discretized_weight": pl.Float64,
            "primary_model_version": pl.Int32,
            "meta_model_version": pl.Int32,
        }
    )
    result = compute_positions(empty_signals)

    assert len(result) == 1
    assert result["symbol"][0] == "BTCUSDT"
    assert result["side"][0] == "flat"
    assert result["weight"][0] == 0.0
    assert result["confidence"][0] == 0.0


def test_long_position_for_primary_side_1() -> None:
    """primary_side=1 → side='long', weight=discretized_weight."""
    signals = _make_signals([0.80], [1], [0.85])
    result = compute_positions(signals)

    assert result["side"][0] == "long"
    assert abs(result["weight"][0] - 0.85) < 1e-9
    assert abs(result["confidence"][0] - 0.80) < 1e-9


def test_long_only_constraint_short_signal_becomes_flat() -> None:
    """Phase 1A: primary_side=-1 → position side='flat' (no short selling)."""
    signals = _make_signals([0.85], [-1], [0.85])
    result = compute_positions(signals)

    assert result["side"][0] == "flat"


def test_latest_signal_wins() -> None:
    """Multiple signals: latest event_ts determines the position."""
    signals = _make_signals(
        meta_confidences=[0.60, 0.90],
        primary_sides=[1, 1],
        discretized_weights=[0.35, 0.85],
        event_ts_offsets_h=[0, 5],  # second signal is 5h later
    )
    result = compute_positions(signals)

    # Latest signal: meta_confidence=0.90, discretized_weight=0.85
    assert abs(result["confidence"][0] - 0.90) < 1e-9
    assert abs(result["weight"][0] - 0.85) < 1e-9
    assert result["signal_id"][0] == "sig-1"


def test_output_columns_match_positions_schema() -> None:
    """Output has POSITIONS_SCHEMA columns: symbol, side, weight, confidence, signal_id, updated_at."""
    signals = _make_signals([0.80], [1], [0.85])
    result = compute_positions(signals)

    expected = {"symbol", "side", "weight", "confidence", "signal_id", "updated_at"}
    assert set(result.columns) == expected
    assert result.dtypes[result.columns.index("weight")] == pl.Float64
    assert result.dtypes[result.columns.index("updated_at")] == pl.Datetime
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/ml/test_position.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `position.py`**

```python
# src/trading_databricks/ml/inference/position.py
from __future__ import annotations

from datetime import datetime

import polars as pl


def compute_positions(signals: pl.DataFrame) -> pl.DataFrame:
    """Compute current portfolio position from bet-sized signals.

    Phase 1A rules:
    - BTC-only, single position per symbol.
    - Latest signal wins (most recent event_ts).
    - primary_side=-1 → 'flat' (long-only Phase 1A constraint).
    - No passing signals → 'flat' with weight=0.0.

    Args:
        signals: Output of compute_bet_sizes() with raw_bet_size and
                 discretized_weight columns.

    Returns:
        pl.DataFrame with 1 row per symbol matching POSITIONS_SCHEMA:
        symbol, side, weight, confidence, signal_id, updated_at.
    """
    now = datetime.utcnow()

    _flat_position = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "side": ["flat"],
            "weight": [0.0],
            "confidence": [0.0],
            "signal_id": [""],
            "updated_at": [now],
        }
    )

    if signals.is_empty():
        return _flat_position

    # Latest signal by event_ts
    latest = signals.sort("event_ts", descending=True).head(1)
    row = latest.row(0, named=True)

    primary_side = int(row["primary_side"])
    side = "long" if primary_side == 1 else "flat"

    return pl.DataFrame(
        {
            "symbol": [row["symbol"]],
            "side": [side],
            "weight": [float(row["discretized_weight"])],
            "confidence": [float(row["meta_confidence"])],
            "signal_id": [str(row["signal_id"])],
            "updated_at": [now],
        }
    )
```

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/ml/test_bet_sizing.py tests/ml/test_position.py -v
```

Expected: All PASS.

- [ ] **Step 5: Run lint**

```bash
uv run ruff check src/trading_databricks/ml/inference/ tests/ml/test_bet_sizing.py tests/ml/test_position.py
```

Fix any issues.

- [ ] **Step 6: Commit**

```bash
git add src/trading_databricks/ml/inference/position.py tests/ml/test_position.py
git commit -m "TRD-47 feat: compute_positions with latest-signal-wins and long-only Phase 1A constraint"
```

---

## Task 4: Full run + final verification

- [ ] **Step 1: Run full bet sizing + position test suite**

```bash
uv run pytest tests/ml/test_bet_sizing.py tests/ml/test_position.py -v
```

Expected: All tests PASS.

- [ ] **Step 2: Run the full ML test suite to check no regressions**

```bash
uv run pytest tests/ml/ -v
```

Expected: All tests PASS.

- [ ] **Step 3: Final commit if any cleanup was needed**

```bash
git add -u
git commit -m "TRD-47 test: full suite verification, no regressions"
```

---

## Verification

```bash
uv run pytest tests/ml/test_bet_sizing.py tests/ml/test_position.py -v
```

All tests pass. Final: 5 files created.
