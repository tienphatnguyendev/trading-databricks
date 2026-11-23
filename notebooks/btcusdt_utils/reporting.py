"""Console reporting utilities for AFML pipeline.

Provides consistent ASCII table formatting and section headers.
All functions print directly to stdout (no return values).
"""

from __future__ import annotations

import polars as pl
from btcusdt_utils.labeling_analysis import GapAnalysisResult
from btcusdt_utils.labeling_sweep import SweepResult


# ═══════════════════════════════════════════════════════════════════════════
# Section Headers
# ═══════════════════════════════════════════════════════════════════════════

def print_section_header(title: str, width: int = 70, *, char: str = "=") -> None:
    """Print a consistent section header.
    
    Args:
        title: Section title
        width: Header width in characters (default: 70)
        char: Border character (default: "=")
    
    Example:
        >>> print_section_header("PHASE 2B: TRIPLE BARRIER SWEEP")
        ======================================================================
        PHASE 2B: TRIPLE BARRIER SWEEP
        ======================================================================
    """
    border = char * width
    print(border)
    print(title)
    print(border)


def print_subsection(title: str, prefix: str = "▸") -> None:
    """Print a subsection marker.
    
    Args:
        title: Subsection title
        prefix: Prefix symbol (default: "▸")
    
    Example:
        >>> print_subsection("Step 1: Load data")
        ▸ Step 1: Load data
    """
    print(f"\n{prefix} {title}")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 Tables
# ═══════════════════════════════════════════════════════════════════════════

def print_top_configs_table(
    results: list[SweepResult],
    top_n: int = 10,
    *,
    columns: list[str] | None = None,
) -> None:
    """Print ASCII table of top triple-barrier configurations.
    
    Args:
        results: List of SweepResult (ranked, best first)
        top_n: Number of top configs to display
        columns: Optional column subset (default: all key columns)
    
    Example:
        >>> print_top_configs_table(sweep_results, top_n=5)
          Rank  Config             Score  Entropy  Uniq    ACF1  Med Touch  Vert%
          ────  ─────────────────  ─────  ───────  ──────  ────  ─────────  ─────
             1  PT3.0_SL3.0_h480   0.466    0.990   0.476  0.02          8   2.1%
             2  PT2.0_SL3.0_h480   0.465    0.991   0.475  0.02          8   2.0%
             ...
    """
    default_cols = ["rank", "config_name", "score", "entropy", "avg_uniqueness", 
                    "acf_1", "median_touch_bars", "vertical_pct"]
    cols_to_show = columns or default_cols
    
    print(f"\n  {'Rank':>6}  {'Config':<20}  {'Score':>7}  {'Entropy':>8}  "
          f"{'Uniq':>6}  {'ACF1':>6}  {'MedTouch':>9}  {'Vert%':>6}")
    print(f"  {'─'*6}  {'─'*20}  {'─'*7}  {'─'*8}  "
          f"{'─'*6}  {'─'*6}  {'─'*9}  {'─'*6}")
    
    for rank, result in enumerate(results[:top_n], 1):
        q = result.quality
        print(f"  {rank:>6}  {result.config_name:<20}  {q.score:>7.4f}  {q.entropy:>8.3f}  "
              f"{q.avg_uniqueness:>6.3f}  {q.acf_1:>6.3f}  {q.median_touch_bars:>9.1f}  "
              f"{q.vertical_pct:>5.1%}")


def print_gap_analysis_table(gaps: list[GapAnalysisResult]) -> None:
    """Print gap analysis results with ✓/✗ status.
    
    Args:
        gaps: List of GapAnalysisResult
    
    Example:
        >>> print_gap_analysis_table(gap_results)
          Metric               Target         Actual  Status
          ───────────────────  ─────────────  ──────  ──────
          Entropy              > 0.95 bits     0.990  ✓
          Avg Uniqueness       > 0.50          0.476  ✗
          ...
    """
    print(f"\n  {'Metric':<20}  {'Target':>13}  {'Actual':>7}  {'Status':>7}")
    print(f"  {'─'*20}  {'─'*13}  {'─'*7}  {'─'*7}")
    
    for gap in gaps:
        status = "✓" if gap.passed else "✗"
        print(f"  {gap.metric:<20}  {gap.target:>13}  {gap.actual:>7.3f}  {status:>7}")


def print_label_distribution(
    df: pl.DataFrame,
    *,
    label_col: str = "label",
    type_col: str = "touch_type",
) -> None:
    """Print label and touch-type distribution tables.
    
    Args:
        df: Polars DataFrame with labeled data
        label_col: Name of label column (default: "label")
        type_col: Name of touch_type column (default: "touch_type")
    
    Example:
        >>> print_label_distribution(df_labeled_best)
        
        Label distribution:
          Label  Count  Percent
          ─────  ─────  ───────
              0  3,789    49.9%
              1  3,795    50.1%
        
        Touch type distribution:
          Type          Count  Percent
          ────────────  ─────  ───────
          0 (TP)        2,145    28.3%
          1 (SL)        5,279    69.6%
          2 (Vertical)    160     2.1%
    """
    # Label distribution
    label_counts = df.group_by(label_col).agg(pl.len().alias("count")).sort(label_col)
    total = label_counts["count"].sum()
    
    print(f"\nLabel distribution:")
    print(f"  {'Label':>6}  {'Count':>7}  {'Percent':>8}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*8}")
    for row in label_counts.iter_rows(named=True):
        lbl = row[label_col]
        cnt = row["count"]
        pct = cnt / total
        print(f"  {lbl:>6}  {cnt:>7,}  {pct:>7.1%}")
    
    # Touch type distribution
    type_counts = df.group_by(type_col).agg(pl.len().alias("count")).sort(type_col)
    
    type_labels = {0: "TP", 1: "SL", 2: "Vertical"}
    print(f"\nTouch type distribution:")
    print(f"  {'Type':<14}  {'Count':>7}  {'Percent':>8}")
    print(f"  {'─'*14}  {'─'*7}  {'─'*8}")
    for row in type_counts.iter_rows(named=True):
        tt = row[type_col]
        cnt = row["count"]
        pct = cnt / total
        label = type_labels.get(tt, str(tt))
        print(f"  {tt} ({label}){' '*(9-len(label))}  {cnt:>7,}  {pct:>7.1%}")


# ═══════════════════════════════════════════════════════════════════════════
# Feature Scoring Tables
# ═══════════════════════════════════════════════════════════════════════════

def print_feature_scores(
    feature_scores: dict[str, float],
    noise_floor: float,
    top_n: int = 20,
) -> None:
    """Print top-N feature scores with noise floor reference.
    
    Args:
        feature_scores: Dict mapping feature name -> score
        noise_floor: Noise floor threshold
        top_n: Number of top features to display
    
    Example:
        >>> print_feature_scores(mi_scores, noise_floor=0.012, top_n=10)
          Rank  Feature                    Score  Above Floor
          ────  ─────────────────────────  ─────  ───────────
             1  kyle_lambda_100          0.0245  ✓
             2  adf_stat_50              0.0213  ✓
             ...
    """
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n  {'Rank':>6}  {'Feature':<30}  {'Score':>7}  {'Above Floor':>12}")
    print(f"  {'─'*6}  {'─'*30}  {'─'*7}  {'─'*12}")
    
    for rank, (feat_name, score) in enumerate(sorted_features[:top_n], 1):
        above_floor = "✓" if score > noise_floor else "✗"
        print(f"  {rank:>6}  {feat_name:<30}  {score:>7.4f}  {above_floor:>12}")
    
    n_above = sum(1 for s in feature_scores.values() if s > noise_floor)
    print(f"\n  Noise floor: {noise_floor:.4f}")
    print(f"  Features above floor: {n_above}/{len(feature_scores)}")


# ═══════════════════════════════════════════════════════════════════════════
# CV Results Tables
# ═══════════════════════════════════════════════════════════════════════════

def print_cv_results(
    results: list[dict],
    *,
    title: str = "Cross-Validation Results",
) -> None:
    """Print per-fold CV results in ASCII table.
    
    Args:
        results: List of per-fold metric dicts (keys: split, auc, acc, etc.)
        title: Table title
    
    Example:
        >>> print_cv_results(wf_results, title="Walk-Forward CV")
        Walk-Forward CV:
          Split  Train    Test   AUC    Acc   Prec
          ─────  ──────  ─────  ─────  ────  ────
              0  5,000    1,500  0.502  0.51  0.52
              1  6,500    1,500  0.498  0.50  0.51
              ...
    """
    if not results:
        print(f"\n{title}: No results.")
        return
    
    print(f"\n{title}:")
    print(f"  {'Split':>6}  {'Train':>7}  {'Test':>5}  {'AUC':>7}  {'Acc':>7}  {'Prec':>6}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*6}")
    
    for r in results:
        split = r.get("split", r.get("fold", "?"))
        train_size = r.get("train_size", r.get("n_train", 0))
        test_size = r.get("test_size", r.get("n_test", 0))
        auc = r.get("auc", 0.0)
        acc = r.get("acc", r.get("accuracy", 0.0))
        prec = r.get("precision", 0.0)
        
        print(f"  {split:>6}  {train_size:>7,}  {test_size:>5,}  "
              f"{auc:>7.4f}  {acc:>7.4f}  {prec:>6.3f}")
    
    # Summary statistics
    aucs = [r.get("auc", 0.0) for r in results]
    if aucs:
        import numpy as np
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))
        print(f"\n  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
