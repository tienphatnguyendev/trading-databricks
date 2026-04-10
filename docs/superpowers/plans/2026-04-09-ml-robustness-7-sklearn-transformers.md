# ML Robustness Plan 7/8: sklearn Transformer Wrappers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap existing pure ML functions in `scikit-learn` compatible Transformer/Estimator classes to ensure train/inference consistency and pipeline serialization.

**Architecture:** Create a `PolarsTransformerMixin` base class that bridges `scikit-learn`'s NumPy expectations with our `polars` implementation. Build specific transformers for CUSUM event detection, triple-barrier labeling, feature engineering, and a feature union that joins results on `event_id`. Add a final bridge to strip metadata and output pure NumPy arrays for the classifier.

**Tech Stack:** scikit-learn, Polars, NumPy, pytest.

---

### Task 1: Create the PolarsTransformerMixin Base Class

**Files:**
- Create: `src/trading_databricks/ml/transformers/__init__.py`
- Create: `src/trading_databricks/ml/transformers/base.py`
- Create: `tests/ml/test_transformers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/test_transformers.py
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from trading_databricks.ml.transformers.base import PolarsTransformerMixin

def test_base_transformer_fit_stores_columns() -> None:
    class DummyTransformer(PolarsTransformerMixin):
        def _fit_polars(self, df: pl.DataFrame, y=None, **fit_params):
            return self
        def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
            return df
    
    t = DummyTransformer()
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    t.fit(df)
    assert t.feature_names_in_ == ["a", "b"]

def test_base_transformer_numpy_roundtrip() -> None:
    class DummyTransformer(PolarsTransformerMixin):
        def _fit_polars(self, df: pl.DataFrame, y=None, **fit_params):
            return self
        def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns((pl.col("f0") * 2).alias("f0"))
            
    t = DummyTransformer()
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    t.fit(arr)
    assert t.feature_names_in_ == ["f0", "f1"]
    
    result = t.transform(arr)
    assert isinstance(result, pl.DataFrame)
    assert result.select("f0").to_series().to_list() == [2.0, 6.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_transformers.py::test_base_transformer_fit_stores_columns -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# src/trading_databricks/ml/transformers/__init__.py
from __future__ import annotations
```

```python
# src/trading_databricks/ml/transformers/base.py
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

class PolarsTransformerMixin(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        if isinstance(X, pl.DataFrame):
            self.feature_names_in_ = X.columns
        elif isinstance(X, np.ndarray):
            self.feature_names_in_ = [f"f{i}" for i in range(X.shape[1])]
        else:
            raise TypeError("Input must be a Polars DataFrame or NumPy array")
        return self._fit_polars(self._to_polars(X), y, **fit_params)

    def transform(self, X):
        return self._transform_polars(self._to_polars(X))

    def _to_polars(self, X) -> pl.DataFrame:
        if isinstance(X, pl.DataFrame):
            return X
        if not hasattr(self, "feature_names_in_"):
            raise ValueError("Estimator not fitted yet.")
        return pl.DataFrame(X, schema=self.feature_names_in_)

    def _fit_polars(self, df: pl.DataFrame, y=None, **fit_params):
        return self

    def _transform_polars(self, df: pl.DataFrame):
        return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_transformers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/ml/transformers/__init__.py src/trading_databricks/ml/transformers/base.py tests/ml/test_transformers.py
git commit -m "feat: add PolarsTransformerMixin base class for sklearn bridging"
```

---

### Task 2: Create CUSUM Event Detector Transformer

**Files:**
- Create: `src/trading_databricks/ml/transformers/cusum_transformer.py`
- Modify: `tests/ml/test_transformers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/ml/test_transformers.py`:
```python
from datetime import datetime
from trading_databricks.ml.config import CUSUMConfig
from trading_databricks.ml.cusum import detect_cusum_events

def test_cusum_transformer() -> None:
    from trading_databricks.ml.transformers.cusum_transformer import CUSUMEventDetector
    bars = pl.DataFrame({
        "bar_index": [1, 2, 3],
        "symbol": ["BTCUSDT"] * 3,
        "close": [100.0, 105.0, 110.0],
        "bar_end_time": [datetime(2026, 1, 1), datetime(2026, 1, 2), datetime(2026, 1, 3)]
    })
    
    config = CUSUMConfig(lookback_days=0)
    expected = detect_cusum_events(bars, config=config)
    
    detector = CUSUMEventDetector(config=config)
    result = detector.fit_transform(bars)
    
    assert result.frame_equal(expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_transformers.py::test_cusum_transformer -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# src/trading_databricks/ml/transformers/cusum_transformer.py
from __future__ import annotations

import polars as pl

from trading_databricks.ml.config import CUSUMConfig
from trading_databricks.ml.cusum import detect_cusum_events
from trading_databricks.ml.transformers.base import PolarsTransformerMixin

class CUSUMEventDetector(PolarsTransformerMixin):
    def __init__(self, config: CUSUMConfig | None = None):
        self.config = config or CUSUMConfig()

    def _fit_polars(self, df: pl.DataFrame, y=None, **fit_params):
        return self

    def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        return detect_cusum_events(df, config=self.config)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_transformers.py::test_cusum_transformer -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/ml/transformers/cusum_transformer.py tests/ml/test_transformers.py
git commit -m "feat: add CUSUMEventDetector sklearn wrapper"
```

---

### Task 3: Create Triple Barrier Labeler Transformer

**Files:**
- Create: `src/trading_databricks/ml/transformers/label_transformer.py`
- Modify: `tests/ml/test_transformers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/ml/test_transformers.py`:
```python
from trading_databricks.ml.config import LabelConfig
from trading_databricks.ml.labeling import compute_triple_barrier_labels

def test_label_transformer() -> None:
    from trading_databricks.ml.transformers.label_transformer import TripleBarrierLabeler
    events = pl.DataFrame({
        "event_id": ["e1"], "symbol": ["BTCUSDT"], "event_ts": [datetime(2026, 1, 1)],
        "bar_index": [1], "log_return": [0.01], "cusum_side": ["up"], "cusum_value": [0.05],
        "threshold_used": [0.02], "ewma_vol": [0.01], "avg_bars_per_day": [10.0]
    })
    bars = pl.DataFrame({
        "bar_index": [1, 2], "symbol": ["BTCUSDT", "BTCUSDT"],
        "close": [100.0, 110.0], "bar_end_time": [datetime(2026, 1, 1), datetime(2026, 1, 2)]
    })
    
    config = LabelConfig(vertical_barrier_days=1)
    expected = compute_triple_barrier_labels(events, bars, config=config)
    
    labeler = TripleBarrierLabeler(config=config)
    result = labeler.fit(events, bars=bars).transform(events)
    
    assert result.frame_equal(expected)

def test_label_transformer_missing_bars() -> None:
    from trading_databricks.ml.transformers.label_transformer import TripleBarrierLabeler
    events = pl.DataFrame({"a": [1]})
    labeler = TripleBarrierLabeler()
    with pytest.raises(ValueError, match="requires 'bars' passed to fit"):
        labeler.fit(events)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_transformers.py::test_label_transformer -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# src/trading_databricks/ml/transformers/label_transformer.py
from __future__ import annotations

import polars as pl

from trading_databricks.ml.config import LabelConfig
from trading_databricks.ml.labeling import compute_triple_barrier_labels
from trading_databricks.ml.transformers.base import PolarsTransformerMixin

class TripleBarrierLabeler(PolarsTransformerMixin):
    def __init__(self, config: LabelConfig | None = None):
        self.config = config or LabelConfig()

    def _fit_polars(self, df: pl.DataFrame, y=None, **fit_params):
        if "bars" not in fit_params:
            raise ValueError("TripleBarrierLabeler requires 'bars' passed to fit().")
        self.bars_ = fit_params["bars"]
        return self

    def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        if not hasattr(self, "bars_"):
            raise ValueError("Estimator not fitted yet.")
        return compute_triple_barrier_labels(df, self.bars_, config=self.config)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_transformers.py -k "test_label" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/ml/transformers/label_transformer.py tests/ml/test_transformers.py
git commit -m "feat: add TripleBarrierLabeler sklearn wrapper"
```

---

### Task 4: Stub Feature Computations

The ML feature pure functions don't exist yet, so we need stubs in `features.py` before we can build their wrappers.

**Files:**
- Modify: `src/trading_databricks/ml/features.py`
- Modify: `tests/ml/test_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/test_features.py
from __future__ import annotations

import polars as pl
from trading_databricks.ml.features import compute_technical, compute_volatility, compute_volume, compute_news

def test_feature_stubs() -> None:
    events = pl.DataFrame({"event_id": ["1"]})
    bars = pl.DataFrame({"bar_index": [1]})
    news_features = pl.DataFrame({"story_id": ["s1"]})
    
    assert "tech_feat" in compute_technical(events, bars).columns
    assert "vol_feat" in compute_volatility(events, bars).columns
    assert "volume_feat" in compute_volume(events, bars).columns
    assert "news_feat" in compute_news(events, news_features).columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_features.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write minimal implementation**

```python
# src/trading_databricks/ml/features.py
"""Pure feature engineering transforms for the ML domain."""
from __future__ import annotations

import polars as pl
from trading_databricks.ml.config import FeatureConfig

def compute_technical(events: pl.DataFrame, bars: pl.DataFrame, *, config: FeatureConfig | None = None) -> pl.DataFrame:
    if events.is_empty():
        return events.with_columns(pl.lit(0.0).alias("tech_feat"))
    return events.with_columns(pl.lit(1.0).alias("tech_feat"))

def compute_volatility(events: pl.DataFrame, bars: pl.DataFrame, *, config: FeatureConfig | None = None) -> pl.DataFrame:
    if events.is_empty():
        return events.with_columns(pl.lit(0.0).alias("vol_feat"))
    return events.with_columns(pl.lit(1.0).alias("vol_feat"))

def compute_volume(events: pl.DataFrame, bars: pl.DataFrame, *, config: FeatureConfig | None = None) -> pl.DataFrame:
    if events.is_empty():
        return events.with_columns(pl.lit(0.0).alias("volume_feat"))
    return events.with_columns(pl.lit(1.0).alias("volume_feat"))

def compute_news(events: pl.DataFrame, news_features: pl.DataFrame, *, config: FeatureConfig | None = None) -> pl.DataFrame:
    if events.is_empty():
        return events.with_columns(pl.lit(0.0).alias("news_feat"))
    return events.with_columns(pl.lit(1.0).alias("news_feat"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/ml/features.py tests/ml/test_features.py
git commit -m "feat: add feature computation stubs for transformer wrappers"
```

---

### Task 5: Create Feature Transformers

**Files:**
- Create: `src/trading_databricks/ml/transformers/feature_transformers.py`
- Modify: `tests/ml/test_transformers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/ml/test_transformers.py`:
```python
def test_feature_transformers() -> None:
    from trading_databricks.ml.transformers.feature_transformers import (
        TechnicalFeatureTransformer, VolatilityFeatureTransformer,
        VolumeFeatureTransformer, NewsFeatureTransformer
    )
    events = pl.DataFrame({"event_id": ["e1"]})
    bars = pl.DataFrame({"bar_index": [1]})
    news_features = pl.DataFrame({"story_id": ["s1"]})
    
    t1 = TechnicalFeatureTransformer()
    assert "tech_feat" in t1.fit(events, bars=bars).transform(events).columns
    
    t2 = VolatilityFeatureTransformer()
    assert "vol_feat" in t2.fit(events, bars=bars).transform(events).columns
    
    t3 = VolumeFeatureTransformer()
    assert "volume_feat" in t3.fit(events, bars=bars).transform(events).columns
    
    t4 = NewsFeatureTransformer()
    assert "news_feat" in t4.fit(events, news_features=news_features).transform(events).columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_transformers.py::test_feature_transformers -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# src/trading_databricks/ml/transformers/feature_transformers.py
from __future__ import annotations

import polars as pl

from trading_databricks.ml.config import FeatureConfig
from trading_databricks.ml.features import compute_news, compute_technical, compute_volatility, compute_volume
from trading_databricks.ml.transformers.base import PolarsTransformerMixin

class TechnicalFeatureTransformer(PolarsTransformerMixin):
    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
    
    def _fit_polars(self, df: pl.DataFrame, y=None, **fit_params):
        if "bars" not in fit_params:
            raise ValueError("TechnicalFeatureTransformer requires 'bars' passed to fit().")
        self.bars_ = fit_params["bars"]
        return self
    
    def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        if not hasattr(self, "bars_"):
            raise ValueError("Estimator not fitted yet.")
        return compute_technical(df, self.bars_, config=self.config)

class VolatilityFeatureTransformer(PolarsTransformerMixin):
    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
    
    def _fit_polars(self, df: pl.DataFrame, y=None, **fit_params):
        if "bars" not in fit_params:
            raise ValueError("VolatilityFeatureTransformer requires 'bars' passed to fit().")
        self.bars_ = fit_params["bars"]
        return self
    
    def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        if not hasattr(self, "bars_"):
            raise ValueError("Estimator not fitted yet.")
        return compute_volatility(df, self.bars_, config=self.config)

class VolumeFeatureTransformer(PolarsTransformerMixin):
    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
    
    def _fit_polars(self, df: pl.DataFrame, y=None, **fit_params):
        if "bars" not in fit_params:
            raise ValueError("VolumeFeatureTransformer requires 'bars' passed to fit().")
        self.bars_ = fit_params["bars"]
        return self
    
    def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        if not hasattr(self, "bars_"):
            raise ValueError("Estimator not fitted yet.")
        return compute_volume(df, self.bars_, config=self.config)

class NewsFeatureTransformer(PolarsTransformerMixin):
    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
    
    def _fit_polars(self, df: pl.DataFrame, y=None, **fit_params):
        if "news_features" not in fit_params:
            raise ValueError("NewsFeatureTransformer requires 'news_features' passed to fit().")
        self.news_features_ = fit_params["news_features"]
        return self
    
    def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        if not hasattr(self, "news_features_"):
            raise ValueError("Estimator not fitted yet.")
        return compute_news(df, self.news_features_, config=self.config)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_transformers.py::test_feature_transformers -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/ml/transformers/feature_transformers.py tests/ml/test_transformers.py
git commit -m "feat: add sklearn wrappers for feature computation"
```

---

### Task 6: Create PolarsFeatureUnion

**Files:**
- Create: `src/trading_databricks/ml/transformers/feature_union.py`
- Modify: `tests/ml/test_transformers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/ml/test_transformers.py`:
```python
def test_feature_union() -> None:
    from trading_databricks.ml.transformers.feature_transformers import TechnicalFeatureTransformer, VolatilityFeatureTransformer
    from trading_databricks.ml.transformers.feature_union import PolarsFeatureUnion
    
    events = pl.DataFrame({"event_id": ["e1"], "symbol": ["BTCUSDT"], "event_ts": [datetime(2026, 1, 1)]})
    bars = pl.DataFrame({"bar_index": [1]})
    
    union = PolarsFeatureUnion([
        ("tech", TechnicalFeatureTransformer()),
        ("vol", VolatilityFeatureTransformer())
    ])
    
    result = union.fit(events, bars=bars).transform(events)
    
    assert "tech_feat" in result.columns
    assert "vol_feat" in result.columns
    assert "symbol" in result.columns
    assert result.shape[1] == 5 # event_id, symbol, event_ts, tech_feat, vol_feat
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_transformers.py::test_feature_union -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# src/trading_databricks/ml/transformers/feature_union.py
from __future__ import annotations

import polars as pl

from trading_databricks.ml.transformers.base import PolarsTransformerMixin

class PolarsFeatureUnion(PolarsTransformerMixin):
    def __init__(self, transformers: list[tuple[str, PolarsTransformerMixin]]):
        self.transformers = transformers
    
    def _fit_polars(self, df: pl.DataFrame, y=None, **fit_params):
        for name, t in self.transformers:
            t.fit(df, y, **fit_params)
        return self
    
    def _transform_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        for name, t in self.transformers:
            features = t.transform(df)
            join_cols = [c for c in features.columns if c not in ("symbol", "event_ts") and c != "event_id"]
            result = result.join(
                features.select(["event_id"] + join_cols), on="event_id", how="left"
            )
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_transformers.py::test_feature_union -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/ml/transformers/feature_union.py tests/ml/test_transformers.py
git commit -m "feat: add PolarsFeatureUnion to join features on event_id"
```

---

### Task 7: Create PolarsToNumpy Bridge

**Files:**
- Create: `src/trading_databricks/ml/transformers/numpy_bridge.py`
- Modify: `tests/ml/test_transformers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/ml/test_transformers.py`:
```python
def test_numpy_bridge() -> None:
    from trading_databricks.ml.transformers.numpy_bridge import PolarsToNumpy
    events = pl.DataFrame({
        "event_id": ["e1"], "symbol": ["BTC"], "event_ts": [datetime(2026, 1, 1)],
        "bar_index": [1], "tech_feat": [1.0], "vol_feat": [2.0]
    })
    
    bridge = PolarsToNumpy()
    bridge.fit(events)
    
    assert bridge.output_features_ == ["tech_feat", "vol_feat"]
    
    result = bridge.transform(events)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 2)
    assert result[0, 0] == 1.0
    assert result[0, 1] == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_transformers.py::test_numpy_bridge -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# src/trading_databricks/ml/transformers/numpy_bridge.py
from __future__ import annotations

import numpy as np
import polars as pl

from trading_databricks.ml.transformers.base import PolarsTransformerMixin

class PolarsToNumpy(PolarsTransformerMixin):
    _METADATA_COLS = {"event_id", "symbol", "event_ts", "computed_at", "bar_index",
                      "bar_end_time", "cpcv_group", "dataset_version"}
    
    def _fit_polars(self, df: pl.DataFrame, y=None, **fit_params):
        self.output_features_ = [c for c in df.columns if c not in self._METADATA_COLS]
        return self
    
    def _transform_polars(self, df: pl.DataFrame):
        if not hasattr(self, "output_features_"):
            raise ValueError("Estimator not fitted yet.")
        return df.select(self.output_features_).to_numpy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_transformers.py::test_numpy_bridge -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading_databricks/ml/transformers/numpy_bridge.py tests/ml/test_transformers.py
git commit -m "feat: add PolarsToNumpy bridge for final pipeline step"
```

---

### Task 8: Package Init & Final Verification

**Files:**
- Modify: `src/trading_databricks/ml/transformers/__init__.py`

- [ ] **Step 1: Write re-exports**

```python
# src/trading_databricks/ml/transformers/__init__.py
from __future__ import annotations

from trading_databricks.ml.transformers.base import PolarsTransformerMixin
from trading_databricks.ml.transformers.cusum_transformer import CUSUMEventDetector
from trading_databricks.ml.transformers.feature_transformers import (
    NewsFeatureTransformer,
    TechnicalFeatureTransformer,
    VolatilityFeatureTransformer,
    VolumeFeatureTransformer,
)
from trading_databricks.ml.transformers.feature_union import PolarsFeatureUnion
from trading_databricks.ml.transformers.label_transformer import TripleBarrierLabeler
from trading_databricks.ml.transformers.numpy_bridge import PolarsToNumpy

__all__ = [
    "PolarsTransformerMixin",
    "CUSUMEventDetector",
    "TripleBarrierLabeler",
    "TechnicalFeatureTransformer",
    "VolatilityFeatureTransformer",
    "VolumeFeatureTransformer",
    "NewsFeatureTransformer",
    "PolarsFeatureUnion",
    "PolarsToNumpy",
]
```

- [ ] **Step 2: Verify linting**

Run: `uv run ruff check src/trading_databricks/ml/transformers/`
Expected: PASS

- [ ] **Step 3: Verify all tests pass**

Run: `uv run pytest tests/ml/ -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/trading_databricks/ml/transformers/__init__.py
git commit -m "feat: export all transformers from package root"
```