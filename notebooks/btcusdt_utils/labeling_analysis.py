"""Analysis and visualization helpers for triple-barrier labeling sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl

from btcusdt_utils.labeling_sweep import SweepResult


# ═══════════════════════════════════════════════════════════════
# Gap analysis data structures
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GapAnalysisTarget:
    """Quality target for gap analysis."""
    metric: str
    target_desc: str
    threshold_fn: Callable[[float], bool]


@dataclass(frozen=True)
class GapAnalysisResult:
    """Result of comparing a metric against its target."""
    metric: str
    target: str
    actual: float
    passed: bool

    def __repr__(self) -> str:
        status = "✓" if self.passed else "✗"
        return (
            f"{status} {self.metric:<20} target={self.target:<12} "
            f"actual={self.actual:.4f}"
        )


# ═══════════════════════════════════════════════════════════════
# DataFrame construction
# ═══════════════════════════════════════════════════════════════

def build_sweep_dataframe(
    results: list[SweepResult],
    *,
    add_rank: bool = False,
) -> pl.DataFrame:
    """Convert list of SweepResult to Polars DataFrame.

    Parameters
    ----------
    results : list of SweepResult (may already contain rank field)
    *
    add_rank : if True, add/overwrite "rank" column (1 = best)

    Returns
    -------
    Polars DataFrame with config params + all quality metrics.
    """
    if not results:
        return pl.DataFrame()

    rows = [r.to_dict() for r in results]
    df = pl.DataFrame(rows)

    if add_rank:
        # Overwrite rank column
        df = df.with_columns(
            pl.int_range(1, len(df) + 1, dtype=pl.Int32).alias("rank")
        )

    return df


# ═══════════════════════════════════════════════════════════════
# Heatmap data preparation
# ═══════════════════════════════════════════════════════════════

def build_heatmap_data(
    df: pl.DataFrame,
    pt_col: str = "pt_mult",
    sl_col: str = "sl_mult",
    value_col: str = "score",
    *,
    annotation_fmt: str = ".3f",
) -> tuple[list[float], list[float], list[list[float]], list[list[str]]]:
    """Build PT×SL heatmap matrices from sweep DataFrame.

    Returns a 4-tuple ready for Plotly Heatmap:
      (x_vals, y_vals, z_matrix, annotation_text)

    Parameters
    ----------
    df : Polars DataFrame with PT, SL, and value columns
    pt_col : column name for PT multiplier (x-axis)
    sl_col : column name for SL multiplier (y-axis)
    value_col : column name for heatmap values (z)
    *
    annotation_fmt : format string for annotation text (e.g., ".3f")

    Returns
    -------
    x_vals : sorted unique PT values
    y_vals : sorted unique SL values
    z_matrix : 2D list[list[float]], shape (len(y_vals), len(x_vals))
    annot_text : 2D list[list[str]], same shape, formatted values
    """
    if df.height == 0:
        return [], [], [], []

    # Get unique sorted values
    x_vals = sorted(df[pt_col].unique().to_list())
    y_vals = sorted(df[sl_col].unique().to_list())

    # Build lookup: (pt, sl) -> value
    lookup = {
        (row[pt_col], row[sl_col]): row[value_col]
        for row in df.iter_rows(named=True)
    }

    # Build z_matrix and annotations (row-major: y × x)
    z_matrix: list[list[float]] = []
    annot_text: list[list[str]] = []

    for sl in y_vals:
        z_row: list[float] = []
        annot_row: list[str] = []
        for pt in x_vals:
            val = lookup.get((pt, sl), float("nan"))
            z_row.append(val)
            if np.isnan(val):
                annot_row.append("—")
            else:
                annot_row.append(f"{val:{annotation_fmt}}")
        z_matrix.append(z_row)
        annot_text.append(annot_row)

    return x_vals, y_vals, z_matrix, annot_text


# ═══════════════════════════════════════════════════════════════
# Gap analysis
# ═══════════════════════════════════════════════════════════════

def analyze_gaps(
    best_result: SweepResult,
    targets: list[GapAnalysisTarget],
) -> list[GapAnalysisResult]:
    """Compare best config's metrics against quality targets.

    Parameters
    ----------
    best_result : the top-ranked SweepResult
    targets : list of GapAnalysisTarget defining thresholds

    Returns
    -------
    List of GapAnalysisResult, one per target.
    """
    q = best_result.quality
    metric_map = {
        "entropy": q.entropy,
        "avg_uniqueness": q.avg_uniqueness,
        "acf_1": q.acf_1,
        "acf_5": q.acf_5,
        "median_touch_bars": q.median_touch_bars,
        "vertical_pct": q.vertical_pct,
        "class_balance": q.class_balance,
        "score": q.score,
        "n_samples": float(q.n_samples),
    }

    results: list[GapAnalysisResult] = []
    for target in targets:
        actual = metric_map.get(target.metric, float("nan"))
        passed = target.threshold_fn(actual) if not np.isnan(actual) else False
        results.append(
            GapAnalysisResult(
                metric=target.metric,
                target=target.target_desc,
                actual=actual,
                passed=passed,
            )
        )

    return results


# ═══════════════════════════════════════════════════════════════
# Filtering helpers
# ═══════════════════════════════════════════════════════════════

def filter_top_n(
    results: list[SweepResult],
    n: int,
) -> list[SweepResult]:
    """Return top N results by score (assumes pre-sorted)."""
    return results[:n]


def filter_by_vol_span(
    df: pl.DataFrame,
    vol_span: int,
) -> pl.DataFrame:
    """Filter DataFrame to configs with specific vol_span."""
    return df.filter(pl.col("vol_span") == vol_span)


def get_top_n_configs(
    df: pl.DataFrame,
    n: int,
    sort_by: str = "score",
) -> pl.DataFrame:
    """Get top N configurations sorted by a metric."""
    return df.sort(sort_by, descending=True).head(n)
