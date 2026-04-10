# ML QA Notebook 02: Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exercise the four feature families on a shared `(events, bars)` pair and surface null-ratio, range, and correlation diagnostics.

**Architecture:** A plain `.py` script runnable locally via Databricks Connect. Uses shared QA utilities. Includes a zero-fill news path and a mock non-zero news path.

**Tech Stack:** Python 3.12, Polars, MLflow, Plotly, PySpark (Databricks Connect)

---

### Task 1: Create Notebook 02

**Files:**
- Create: `notebooks/qa/ml/02_features.py`

- [ ] **Step 1: Write the notebook script**

```python
"""
ml/02_features.py — QA notebook for feature engineering.

Run: uv run python notebooks/qa/ml/02_features.py
"""
import polars as pl
from datetime import datetime, timedelta
from notebooks.qa.utils import (
    mlflow_component_run, load_raw_dollar_bars, log_diag, log_plotly,
    null_ratios, build_heatmap
)
from trading_databricks.ml.cusum import detect_cusum_events
from trading_databricks.ml.features.technical import compute_technical
from trading_databricks.ml.features.volatility import compute_volatility
from trading_databricks.ml.features.volume import compute_volume
from trading_databricks.ml.features.news import compute_news
from trading_databricks.ml.config import CUSUMConfig, FeatureConfig

cusum_config = CUSUMConfig(lookback_days=1)
feature_config = FeatureConfig()

with mlflow_component_run(
    layer="ml", component="features",
    params={
        "feature.technical_windows": feature_config.technical_windows,
        "feature.volatility_windows": feature_config.volatility_windows,
        "feature.volume_windows": feature_config.volume_windows,
        "feature.news_lookback_hours": feature_config.news_lookback_hours,
    }
):
    # Section 1 — Load input
    bars = load_raw_dollar_bars(symbol="BTCUSDT", limit=50_000)
    events = detect_cusum_events(bars, config=cusum_config)
    
    if events.is_empty():
        raise RuntimeError("No events generated. Check bar data and CUSUM config.")

    # Section 2 — Exercise component
    # --- Technical ---
    tech_feats = compute_technical(events.lazy(), bars.lazy(), config=feature_config).collect()
    log_diag("technical_null_ratios", null_ratios(tech_feats))

    # --- Volatility ---
    vol_feats = compute_volatility(events.lazy(), bars.lazy(), config=feature_config).collect()
    log_diag("volatility_null_ratios", null_ratios(vol_feats))

    # --- Volume ---
    volu_feats = compute_volume(events.lazy(), bars.lazy(), config=feature_config).collect()
    log_diag("volume_null_ratios", null_ratios(volu_feats))

    # --- News (zero-fill) ---
    news_zero = compute_news(events.lazy(), None, config=feature_config).collect()
    log_diag("news_zero_fill_report", null_ratios(news_zero))

    # --- News (mock non-zero) ---
    mock_news = pl.DataFrame({
        "published_at": [events["event_ts"].min() - timedelta(hours=1)],
        "is_impactful": [True],
        "validation_status": ["accepted"],
        "l1_category": ["Sentiment"],
        "confidence": [0.9],
        "expected_direction": [1]
    }).lazy()
    news_mock = compute_news(events.lazy(), mock_news, config=feature_config).collect()

    # Section 3 & 4 — Visuals and Correlations
    all_features = tech_feats.join(vol_feats, on="event_id").join(volu_feats, on="event_id")
    # Ensure we drop string columns for correlation
    numeric_features = all_features.select([col for col in all_features.columns if all_features[col].dtype.is_numeric()])
    log_plotly("feature_correlation_heatmap", build_heatmap(numeric_features, corr=True))

    # Section 5 — Sanity assertions
    assert tech_feats.height == events.height
    assert (tech_feats["rsi_14"].drop_nulls().is_between(0, 100)).all()
    assert (vol_feats["parkinson_vol_20"].drop_nulls() >= 0).all()
    assert (vol_feats["garman_klass_20"].drop_nulls() >= 0).all()
    assert (volu_feats["relative_volume_20"].drop_nulls() >= 0).all()
    assert not news_zero["has_news_24h"].any(), "zero-fill path should never flag news"
    assert news_mock["news_count_Sentiment"].sum() > 0, "mock path must generate counts"
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/qa/ml/02_features.py
git commit -m "test(ml): add 02_features QA notebook"
```