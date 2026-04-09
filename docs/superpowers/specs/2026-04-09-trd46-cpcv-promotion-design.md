# TRD-46 Delta Spec: CPCV + Sample Weights + Promotion Gates

> **Linear issue:** [TRD-46](https://linear.app/aaron-solo/issue/TRD-46)
> **Master spec:** `2026-04-07-ml-pipeline-phase1a-design.md` §§ 5.6, 5.7
> **Validation reference:** `docs/advanced-ml-modelling/validation.md`
> **This document covers only design decisions and implementation details NOT in the Linear issue.**

---

## Design Decisions

### Pure Python/numpy — No Polars, No Spark

All functions in this module use numpy arrays and Python scalars. The CPCV splitter must be sklearn-compatible. Promotion gates are pure math on aggregated metrics.

### CPCVSplitter: Constructor Injection for Purge Spans

The sklearn `.split(X, y, groups)` interface cannot carry label span information. Solution: inject span arrays at construction time.

```python
class CPCVSplitter:
    def __init__(
        self,
        config: CPCVConfig = CPCVConfig(),
        *,
        t_start: np.ndarray,  # bar_index where each event starts
        t_end: np.ndarray,    # vertical_bar where each event's label expires
    ):
        self.config = config
        self.t_start = t_start
        self.t_end = t_end
```

`.split()` signature remains standard: `split(X, y=None, groups=None)`. This means `CPCVSplitter` works with `RandomizedSearchCV` and `cross_val_score` directly.

### sample_weights.py: Thin Wrapper

TRD-44 computes `avg_uniqueness` as part of labeling. This module provides a convenience function that extracts it:

```python
def compute_sample_weights(
    labels: pl.DataFrame,
) -> np.ndarray:
    """Extract avg_uniqueness from labeled events as numpy array.
    
    For sklearn sample_weight parameter compatibility.
    """
    return labels.get_column("avg_uniqueness").to_numpy()
```

The `n_bars` parameter from the Linear issue is NOT needed since TRD-44 already computed uniqueness. Simplify the signature.

---

## CPCV Implementation Details

### Group Formation

```python
n_samples = len(X)
group_size = n_samples // self.config.n_groups
groups = np.zeros(n_samples, dtype=int)
for g in range(self.config.n_groups):
    start = g * group_size
    end = (g + 1) * group_size if g < self.config.n_groups - 1 else n_samples
    groups[start:end] = g
```

Groups are chronologically ordered. Data MUST be sorted by `event_ts` before passing to the splitter.

**Index alignment requirement:** The `t_start` and `t_end` arrays passed to `__init__()` MUST be aligned with the rows of `X` passed to `.split()`. The caller (TRD-48 training pipeline) is responsible for extracting `t_start = labels["bar_index"].to_numpy()` and `t_end = labels["vertical_bar"].to_numpy()` in the same row order as the feature matrix `X`. Sort the assembled training dataset by `event_ts` ONCE before extracting both arrays and the feature matrix.

### Split Generation

```python
from itertools import combinations

test_combos = list(combinations(range(self.config.n_groups), self.config.k_test))
# len(test_combos) = C(6,2) = 15
```

For each combo, test indices = samples in test groups, train indices = samples in remaining groups MINUS purged and embargoed.

### Purge Algorithm

For each test observation `j` in the test set:
- Its label spans bars `[t_start[j], t_end[j]]`
- Any training observation `i` whose label span `[t_start[i], t_end[i]]` overlaps with the test span must be removed

```python
def _purge(self, train_idx, test_idx):
    """Remove training samples whose labels overlap with test samples."""
    test_starts = self.t_start[test_idx]
    test_ends = self.t_end[test_idx]
    test_min_start = test_starts.min()
    test_max_end = test_ends.max()
    
    # Fast pre-filter: only check training samples near test range
    candidates = train_idx[
        (self.t_end[train_idx] >= test_min_start) &
        (self.t_start[train_idx] <= test_max_end)
    ]
    
    purged = set()
    for i in candidates:
        for j in test_idx:
            if self.t_start[i] <= self.t_end[j] and self.t_end[i] >= self.t_start[j]:
                purged.add(i)
                break
    
    return np.array([i for i in train_idx if i not in purged])
```

### Embargo

After purging, remove an additional `embargo_pct × n_samples` observations immediately after each test group boundary:

```python
def _embargo(self, train_idx, test_idx):
    """Remove embargo buffer after test group boundaries."""
    embargo_size = int(self.config.embargo_pct * len(self.t_start))
    test_max_idx = test_idx.max()
    
    embargo_range = set(range(test_max_idx + 1, test_max_idx + 1 + embargo_size))
    return np.array([i for i in train_idx if i not in embargo_range])
```

### Backtest Path Construction

5 non-overlapping paths from 15 splits. Each path covers the full time span.

```python
def get_backtest_paths(self) -> list[list[tuple[int, ...]]]:
    """Return 5 paths, each a list of (test_group_combo) that covers all N groups."""
    # phi[6,2] = 5 paths
    # Each path = sequence of non-overlapping test combos covering groups 0-5
    # Path 1: (0,1), (2,3), (4,5) — but these must form a valid covering
    # 
    # Algorithm: enumerate all combinations of splits where test groups
    # don't overlap and together cover all N groups
    ...
```

The path construction is the most complex part. Reference: AFML Ch. 12, specifically the φ[N,k] formula.

---

## Promotion Gate Formulas

### PSR (Probabilistic Sharpe Ratio)

```python
def compute_psr(
    sharpe: float,
    returns: np.ndarray,
    benchmark_sharpe: float = 0.0,
) -> float:
    """AFML Ch. 14 — Probabilistic Sharpe Ratio."""
    T = len(returns)
    gamma_3 = scipy.stats.skew(returns)      # skewness
    gamma_4 = scipy.stats.kurtosis(returns, fisher=False)  # excess kurtosis + 3
    
    sigma_sr = np.sqrt(
        (1 - gamma_3 * sharpe + (gamma_4 - 1) / 4 * sharpe**2) / (T - 1)
    )
    
    if sigma_sr < 1e-10:
        return 1.0 if sharpe > benchmark_sharpe else 0.0
    
    return float(scipy.stats.norm.cdf((sharpe - benchmark_sharpe) / sigma_sr))
```

**Note on kurtosis:** Use `fisher=False` for raw kurtosis (normal = 3). The formula uses `γ₄ - 1` which assumes raw kurtosis. Crypto BTC dollar bar returns typically have `γ₄ ∈ [8, 18]`.

### DSR (Deflated Sharpe Ratio)

```python
def compute_dsr(
    sharpe: float,
    returns: np.ndarray,
    n_trials: int,
) -> float:
    """AFML Ch. 14 — Deflated Sharpe Ratio."""
    # Expected maximum Sharpe under null hypothesis
    euler_mascheroni = 0.5772156649
    
    if n_trials <= 1:
        e_max_sr = 0.0
    else:
        e_max_sr = (
            (1 - euler_mascheroni) * scipy.stats.norm.ppf(1 - 1 / n_trials)
            + euler_mascheroni * scipy.stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )
    
    # Use E[max(SR)] as benchmark in PSR formula
    return compute_psr(sharpe, returns, benchmark_sharpe=e_max_sr)
```

### Strategy-Failure Probability

```python
def compute_strategy_failure_prob(
    sharpe: float,
    returns: np.ndarray,
) -> float:
    """AFML Ch. 15 — probability strategy is actually unprofitable."""
    # P(failure) = 1 - PSR(SR*=0)
    return 1.0 - compute_psr(sharpe, returns, benchmark_sharpe=0.0)
```

### Execution-Cost Robustness

```python
def check_cost_robustness(
    path_returns: np.ndarray,
    estimated_cost_per_trade: float = 0.001,  # Binance 0.1%
) -> bool:
    """Check if strategy is profitable at 2x estimated costs."""
    cost_adjusted = path_returns - 2 * estimated_cost_per_trade
    return float(np.mean(cost_adjusted)) > 0
```

### Full Gate Evaluation

```python
def evaluate_promotion_gates(
    cpcv_path_results: list[dict],  # per-path: returns, sharpe, max_dd, tuw
    meta_label_metrics: dict,       # precision, recall, f1, logloss
    trial_count: int,               # from silver.trial_registry
) -> pl.DataFrame:
    """Evaluate all 10 gates. Returns PROMOTION_EVIDENCE_SCHEMA."""
    
    # Aggregate across 5 paths
    all_returns = np.concatenate([p["returns"] for p in cpcv_path_results])
    mean_sharpe = np.mean([p["sharpe"] for p in cpcv_path_results])
    max_dd = max(p["max_drawdown"] for p in cpcv_path_results)
    max_tuw = max(p["time_under_water"] for p in cpcv_path_results)
    
    gates = {
        "psr": compute_psr(mean_sharpe, all_returns) > 0.95,
        "dsr": compute_dsr(mean_sharpe, all_returns, trial_count) > 0.90,
        "max_drawdown": max_dd < 0.25,
        "time_under_water": max_tuw < 90,
        "cost_robust": check_cost_robustness(all_returns),
        "strategy_failure_prob": compute_strategy_failure_prob(mean_sharpe, all_returns) < 0.10,
        "meta_precision": meta_label_metrics["precision"] > 0.60,
        "meta_recall": meta_label_metrics["recall"] > 0.50,
        "meta_f1": meta_label_metrics["f1"] > 0.55,
        "meta_logloss": meta_label_metrics["logloss"] < 0.65,
    }
    ...
```

---

## Edge Cases

| Scenario | Behavior |
|---|---|
| < 6 events (can't form 6 groups) | Raise `ValueError` — CPCV requires minimum N×k observations |
| n_trials = 0 (first model ever) | DSR benchmark = 0 (equivalent to PSR). Log warning: "first trial, DSR = PSR" |
| All paths have negative Sharpe | All gates fail. `failure_reasons` includes "psr", "dsr" |
| Path returns all zero | Sharpe = 0, PSR = 0.5. Fails PSR > 0.95 gate |
| Purge removes all training data | Skip that split, log warning. Still produce remaining valid splits |

---

## Cross-Module Dependencies

| Reads | Provides to |
|---|---|
| `CPCVConfig` from `ml/config.py` | TRD-48 (training): `CPCVSplitter` for cross-validation |
| `PROMOTION_EVIDENCE_SCHEMA` from `ml/schemas.py` | TRD-48 (training): `compute_sample_weights()` for sklearn |
| `TRIAL_REGISTRY_SCHEMA` from `ml/schemas.py` | TRD-48 (training): `evaluate_promotion_gates()` post-training |
| Labels with `avg_uniqueness` (from TRD-44) | |

---

## Test Data Recipes

### Recipe 1: CPCV Split Verification

```python
# 120 samples, 6 groups of 20
# Verify: 15 unique splits, each test set = 2 groups (40 samples)
# Verify: 5 non-overlapping backtest paths
X = np.random.randn(120, 16)
t_start = np.arange(120)
t_end = t_start + 10  # each label spans 10 bars
```

### Recipe 2: Purge Correctness

```python
# Create overlapping labels at group boundary
# t_start[19] = 19, t_end[19] = 29 (spans into group 1)
# When group 1 is in test set, sample 19 must be purged from training
```

### Recipe 3: PSR with Known Distribution

```python
# Normal returns with known Sharpe: mean=0.02, std=0.1 → SR=0.2
# T=100 → σ_SR calculable, PSR should match analytic value
```

### Recipe 4: Promotion All-Pass vs One-Fail

```python
# Construct metrics that pass all 10 gates → all_gates_pass=True
# Then flip one gate (e.g., max_dd=0.30) → all_gates_pass=False, failure_reasons="max_drawdown"
```
