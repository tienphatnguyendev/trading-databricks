"""Bar and label quality metrics (AFML-inspired)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats
from statsmodels.stats.diagnostic import acorr_ljungbox


# ═══════════════════════════════════════════════════════════════
# Bar quality
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BarQuality:
    """Quality metrics for a bar-sampling configuration."""
    name: str
    n_bars: int
    jb_stat: float          # Jarque-Bera statistic (lower = more Gaussian)
    jb_pval: float
    lb_stat: float          # Ljung-Box Q-stat at lag 10 (lower = less ACF)
    lb_pval: float
    acf_1: float            # Autocorrelation at lag 1
    ret_mean: float
    ret_std: float
    cv: float               # Coefficient of variation of |returns|

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, n={self.n_bars:,}, "
            f"JB={self.jb_stat:.1f}, LB_Q10={self.lb_stat:.1f}, "
            f"ACF1={self.acf_1:.4f})"
        )


def compute_bar_quality(returns: np.ndarray, name: str) -> BarQuality:
    """Compute IID-ness quality metrics for a return series."""
    returns = returns[np.isfinite(returns)]
    n = len(returns)
    if n < 30:
        raise ValueError(f"{name}: only {n} returns — need ≥30")

    jb_stat, jb_pval = sp_stats.jarque_bera(returns)

    lb_result = acorr_ljungbox(returns, lags=[10], return_df=True)
    lb_stat = float(lb_result["lb_stat"].iloc[0])
    lb_pval = float(lb_result["lb_pvalue"].iloc[0])

    acf_1 = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])

    abs_ret = np.abs(returns)
    cv = float(abs_ret.std() / abs_ret.mean()) if abs_ret.mean() > 0 else float("inf")

    return BarQuality(
        name=name, n_bars=n,
        jb_stat=float(jb_stat), jb_pval=float(jb_pval),
        lb_stat=lb_stat, lb_pval=lb_pval, acf_1=acf_1,
        ret_mean=float(returns.mean()), ret_std=float(returns.std()),
        cv=cv,
    )


# ═══════════════════════════════════════════════════════════════
# Label quality
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LabelQuality:
    """Quality metrics for a triple-barrier labeling configuration."""
    name: str
    n_samples: int
    entropy: float              # Label entropy (bits); max = 1.0 for binary
    avg_uniqueness: float       # AFML Ch. 4 — mean(1/concurrency); want → 1.0
    acf_1: float                # Label ACF at lag 1; want → 0.0
    acf_5: float                # Label ACF at lag 5; want → 0.0
    median_touch_bars: float    # Median bars until touch; want > 5
    vertical_pct: float         # Fraction hitting vertical barrier; want < 0.30
    class_balance: float        # min(p, 1-p); want ∈ [0.40, 0.60]
    score: float = 0.0          # Composite: uniqueness × entropy × (1 - |ACF1|)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, n={self.n_samples:,}, "
            f"entropy={self.entropy:.3f}, uniq={self.avg_uniqueness:.3f}, "
            f"ACF1={self.acf_1:.3f}, score={self.score:.4f})"
        )


def label_entropy(labels: np.ndarray) -> float:
    """Shannon entropy in bits for a discrete label array."""
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def label_avg_uniqueness(
    event_starts: np.ndarray,
    event_ends: np.ndarray,
) -> float:
    """Average uniqueness of labels (AFML Ch. 4.5).

    Parameters
    ----------
    event_starts : bar indices where each label begins
    event_ends   : bar indices where each label's barrier is touched

    Returns
    -------
    Mean uniqueness ∈ (0, 1]. 1.0 = fully independent labels.
    """
    n_events = len(event_starts)
    if n_events == 0:
        return 0.0
    max_bar = int(event_ends.max()) + 1

    # Concurrency: how many labels are active at each bar
    concurrency = np.zeros(max_bar, dtype=np.int32)
    for s, e in zip(event_starts, event_ends):
        concurrency[s : e + 1] += 1

    # Per-sample uniqueness = mean(1 / concurrency) over the sample's span
    uniqueness = np.empty(n_events, dtype=np.float64)
    for i, (s, e) in enumerate(zip(event_starts, event_ends)):
        span = concurrency[s : e + 1]
        uniqueness[i] = float(np.mean(1.0 / span))

    return float(np.mean(uniqueness))


def compute_label_quality(
    labels: np.ndarray,
    touch_bars: np.ndarray,
    touch_types: np.ndarray,
    event_starts: np.ndarray | None = None,
    event_ends: np.ndarray | None = None,
    name: str = "",
) -> LabelQuality:
    """Compute all label-quality metrics for a TB configuration."""
    n = len(labels)
    ent = label_entropy(labels)

    if event_starts is not None and event_ends is not None:
        avg_uniq = label_avg_uniqueness(event_starts, event_ends)
    else:
        avg_uniq = float("nan")

    lbl_num = labels.astype(np.float64)
    acf1 = float(np.corrcoef(lbl_num[:-1], lbl_num[1:])[0, 1]) if n > 1 else 0.0
    acf5 = float(np.corrcoef(lbl_num[:-5], lbl_num[5:])[0, 1]) if n > 5 else 0.0

    med_touch = float(np.median(touch_bars))
    vert_pct = float(np.mean(touch_types == 2))
    _, counts = np.unique(labels, return_counts=True)
    balance = float(counts.min() / counts.sum()) if len(counts) > 1 else 0.0

    score = avg_uniq * ent * (1.0 - abs(acf1))

    return LabelQuality(
        name=name, n_samples=n, entropy=ent,
        avg_uniqueness=avg_uniq, acf_1=acf1, acf_5=acf5,
        median_touch_bars=med_touch, vertical_pct=vert_pct,
        class_balance=balance, score=score,
    )
