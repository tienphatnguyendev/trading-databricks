"""Triple-barrier parameter grid sweep for label optimization.

Provides configuration-driven sweep over PT/SL/hold/vol_span parameter space
with quality gating and efficient array pruning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .labeling import apply_triple_barrier
from .quality import LabelQuality, compute_label_quality


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SweepConfig:
    """Configuration for triple-barrier parameter sweep.
    
    Parameters
    ----------
    pt_mults : tuple of profit-taking multipliers (e.g., (0.5, 1.0, 2.0))
    sl_mults : tuple of stop-loss multipliers (e.g., (0.5, 1.0, 2.0))
    max_holds : tuple of max holding periods in bars (e.g., (30, 60, 120))
    vol_spans : tuple of volatility EWMA spans (e.g., (20, 50, 100))
    min_samples : minimum labels required to pass quality gate
    min_balance : minimum class balance (minority class fraction)
    min_uniqueness : minimum average uniqueness (AFML Ch. 4)
    """
    pt_mults: tuple[float, ...]
    sl_mults: tuple[float, ...]
    max_holds: tuple[int, ...]
    vol_spans: tuple[int, ...]
    min_samples: int = 100
    min_balance: float = 0.20
    min_uniqueness: float = 0.10
    
    @property
    def total_configs(self) -> int:
        """Total number of configurations in the grid."""
        return (len(self.pt_mults) * len(self.sl_mults) * 
                len(self.max_holds) * len(self.vol_spans))


@dataclass(frozen=True)
class SweepResult:
    """Result for a single triple-barrier configuration.
    
    Attributes
    ----------
    config_name : unique identifier (e.g., "pt1.0_sl1.0_h60_v50")
    pt_mult : profit-taking multiplier
    sl_mult : stop-loss multiplier
    max_hold : maximum holding period (bars)
    vol_span : volatility EWMA span
    quality : computed LabelQuality metrics
    rank : optional rank (assigned after sorting)
    """
    config_name: str
    pt_mult: float
    sl_mult: float
    max_hold: int
    vol_span: int
    quality: LabelQuality
    rank: int = 0
    
    def to_dict(self) -> dict:
        """Convert to flat dictionary for DataFrame construction."""
        return {
            "config_name": self.config_name,
            "pt_mult": self.pt_mult,
            "sl_mult": self.sl_mult,
            "max_hold": self.max_hold,
            "vol_span": self.vol_span,
            "rank": self.rank,
            # Unpack quality metrics
            "n_samples": self.quality.n_samples,
            "entropy": self.quality.entropy,
            "avg_uniqueness": self.quality.avg_uniqueness,
            "acf_1": self.quality.acf_1,
            "abs_acf_1": abs(self.quality.acf_1),
            "acf_5": self.quality.acf_5,
            "median_touch_bars": self.quality.median_touch_bars,
            "vertical_pct": self.quality.vertical_pct,
            "class_balance": self.quality.class_balance,
            "score": self.quality.score,
        }


class SweepReporter(Protocol):
    """Protocol for progress reporting during sweep."""
    
    def report_progress(
        self,
        vol_span: int,
        n_evaluated: int,
        n_skipped: int,
        elapsed: float,
    ) -> None:
        """Report progress after completing a vol_span batch.
        
        Parameters
        ----------
        vol_span : volatility span just completed
        n_evaluated : cumulative configs evaluated
        n_skipped : cumulative configs skipped
        elapsed : elapsed time in seconds
        """
        ...


@dataclass(frozen=True)
class SweepStats:
    """Summary statistics from a parameter sweep."""
    n_evaluated: int
    n_skip_samples: int
    n_skip_balance: int
    n_skip_uniq: int
    n_skip_nan: int
    total_time: float
    
    @property
    def n_skipped(self) -> int:
        return (self.n_skip_samples + self.n_skip_balance + 
                self.n_skip_uniq + self.n_skip_nan)


# ═══════════════════════════════════════════════════════════════
# Sweep Engine
# ═══════════════════════════════════════════════════════════════

def run_triple_barrier_sweep(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    vol_arrays: dict[int, np.ndarray],
    config: SweepConfig,
    reporter: SweepReporter | None = None,
) -> tuple[list[SweepResult], dict[str, dict], SweepStats]:
    """Run vectorized triple-barrier parameter sweep with quality gating.
    
    Sweeps the full grid of PT × SL × hold × vol_span configurations,
    applying quality gates at each step:
      1. n_samples ≥ min_samples
      2. class_balance ≥ min_balance (cheap pre-check)
      3. avg_uniqueness ≥ min_uniqueness (full quality computation)
      4. score is not NaN
    
    Only configurations passing all gates are returned.
    
    Algorithm complexity: O(n_configs × max_hold × n_bars) where the
    inner loop uses vectorized wavefront scanning via apply_triple_barrier.
    
    Parameters
    ----------
    close : 1-D array of bar close prices (n_bars,)
    high : 1-D array of bar high prices (n_bars,)
    low : 1-D array of bar low prices (n_bars,)
    vol_arrays : dict mapping vol_span → volatility array (n_bars,)
    config : sweep configuration (grid + quality thresholds)
    reporter : optional progress reporter (called after each vol_span)
    
    Returns
    -------
    results : list of SweepResult, ranked by score (descending)
    label_arrays : dict mapping config_name → raw label arrays
                   (labels, touch_bars, touch_types, event_starts, event_ends)
    stats : sweep statistics (timings, skip counts)
    
    Examples
    --------
    >>> cfg = SweepConfig(
    ...     pt_mults=(0.5, 1.0, 2.0),
    ...     sl_mults=(0.5, 1.0, 2.0),
    ...     max_holds=(30, 60, 120),
    ...     vol_spans=(20, 50),
    ... )
    >>> vol_arrays = {20: df["vol_20"].to_numpy(), 50: df["vol_50"].to_numpy()}
    >>> results, arrays, stats = run_triple_barrier_sweep(
    ...     close, high, low, vol_arrays, cfg
    ... )
    >>> print(f"Evaluated: {stats.n_evaluated}, Best score: {results[0].quality.score:.4f}")
    """
    results: list[SweepResult] = []
    label_arrays: dict[str, dict] = {}
    
    # Counters
    n_evaluated = 0
    n_skip_samples = 0
    n_skip_balance = 0
    n_skip_uniq = 0
    n_skip_nan = 0
    
    t0 = time.perf_counter()
    
    # Outer loop: vol_span (extract vol array once per span)
    for vol_span in config.vol_spans:
        if vol_span not in vol_arrays:
            raise ValueError(
                f"vol_span={vol_span} not in vol_arrays. "
                f"Available: {sorted(vol_arrays.keys())}"
            )
        
        _vol = vol_arrays[vol_span]
        
        # Inner loops: PT × SL × hold
        for pt_mult in config.pt_mults:
            for sl_mult in config.sl_mults:
                for max_hold in config.max_holds:
                    cfg_name = (
                        f"pt{pt_mult}_sl{sl_mult}"
                        f"_h{max_hold}_v{vol_span}"
                    )
                    
                    # ── Apply triple barrier ────────────────────────
                    labels, touch_bars, touch_types, starts, ends = \
                        apply_triple_barrier(
                            close, high, low, _vol,
                            pt_mult, sl_mult, max_hold,
                        )
                    
                    n = len(labels)
                    
                    # ── Quality gate 1: sample count ────────────────
                    if n < config.min_samples:
                        n_skip_samples += 1
                        continue
                    
                    # ── Quality gate 2: class balance (cheap) ───────
                    _, counts = np.unique(labels, return_counts=True)
                    bal = (float(counts.min() / counts.sum())
                           if len(counts) > 1 else 0.0)
                    
                    if bal < config.min_balance:
                        n_skip_balance += 1
                        continue
                    
                    # ── Quality gate 3: full quality + uniqueness ───
                    lq = compute_label_quality(
                        labels, touch_bars, touch_types,
                        starts, ends, name=cfg_name,
                    )
                    
                    if lq.avg_uniqueness < config.min_uniqueness:
                        n_skip_uniq += 1
                        continue
                    
                    # ── Quality gate 4: valid score ─────────────────
                    if np.isnan(lq.score):
                        n_skip_nan += 1
                        continue
                    
                    # ── Accept configuration ────────────────────────
                    results.append(
                        SweepResult(
                            config_name=cfg_name,
                            pt_mult=pt_mult,
                            sl_mult=sl_mult,
                            max_hold=max_hold,
                            vol_span=vol_span,
                            quality=lq,
                        )
                    )
                    
                    # Store raw arrays (will be pruned later)
                    label_arrays[cfg_name] = {
                        "labels": labels,
                        "touch_bars": touch_bars,
                        "touch_types": touch_types,
                        "event_starts": starts,
                        "event_ends": ends,
                    }
                    
                    n_evaluated += 1
        
        # ── Report progress after each vol_span ─────────────────────────
        if reporter is not None:
            elapsed = time.perf_counter() - t0
            n_skipped = (n_skip_samples + n_skip_balance + 
                        n_skip_uniq + n_skip_nan)
            reporter.report_progress(vol_span, n_evaluated, n_skipped, elapsed)
    
    # ── Rank by score (descending) ──────────────────────────────────────
    results.sort(key=lambda r: r.quality.score, reverse=True)
    
    # Assign ranks (1-indexed)
    results = [
        SweepResult(
            config_name=r.config_name,
            pt_mult=r.pt_mult,
            sl_mult=r.sl_mult,
            max_hold=r.max_hold,
            vol_span=r.vol_span,
            quality=r.quality,
            rank=i + 1,
        )
        for i, r in enumerate(results)
    ]
    
    # ── Build statistics ────────────────────────────────────────────────
    total_time = time.perf_counter() - t0
    stats = SweepStats(
        n_evaluated=n_evaluated,
        n_skip_samples=n_skip_samples,
        n_skip_balance=n_skip_balance,
        n_skip_uniq=n_skip_uniq,
        n_skip_nan=n_skip_nan,
        total_time=total_time,
    )
    
    return results, label_arrays, stats


def prune_label_arrays(
    sweep_results: list[SweepResult],
    label_arrays: dict[str, dict],
    keep_top_n: int = 20,
) -> dict[str, dict]:
    """Keep only top-N configs' label arrays to save memory.
    
    Parameters
    ----------
    sweep_results : ranked list of SweepResult (best first)
    label_arrays : dict mapping config_name → raw arrays
    keep_top_n : number of top configs to retain
    
    Returns
    -------
    pruned_arrays : dict with only top-N configs
    
    Examples
    --------
    >>> # After sweep, keep only top 20 for visualization
    >>> pruned = prune_label_arrays(results, arrays, keep_top_n=20)
    >>> print(f"Kept {len(pruned)} / {len(arrays)} configs")
    """
    if keep_top_n <= 0:
        raise ValueError(f"keep_top_n must be > 0, got {keep_top_n}")
    
    top_names = {r.config_name for r in sweep_results[:keep_top_n]}
    
    return {
        name: arrays
        for name, arrays in label_arrays.items()
        if name in top_names
    }
