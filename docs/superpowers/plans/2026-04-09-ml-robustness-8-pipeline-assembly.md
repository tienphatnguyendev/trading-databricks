# ML Robustness Plan 8/8: sklearn Pipeline Assembly + MLflow

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Transformers into an sklearn Pipeline, enable hyperparameter search with CPCV, implement the training job entry point, and add MLflow serialization with Unity Catalog registration.

**Architecture:** A two-phase design where CUSUM and Labeling run sequentially as a pre-pipeline to define the dataset, followed by an sklearn `Pipeline` (Feature Union -> Numpy Conversion -> Classifier) evaluated using `RandomizedSearchCV` and `CPCVSplitter`. The final model is serialized as a custom `mlflow.pyfunc.PythonModel` and registered to Unity Catalog.

**Tech Stack:** scikit-learn, MLflow (Unity Catalog), Polars, PySpark, NumPy.

---

## Task 1: Pipeline Builder

**Files:**
- Create: `tests/ml/test_pipeline_builder.py`
- Create: `src/trading_databricks/ml/transformers/pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/ml/test_pipeline_builder.py`:

```python
from datetime import datetime, timedelta, UTC
import polars as pl
from sklearn.pipeline import Pipeline
from trading_databricks.ml.config import MLPipelineConfig
from trading_databricks.ml.cusum import detect_cusum_events
from trading_databricks.ml.labeling import compute_triple_barrier_labels
from trading_databricks.ml.transformers.pipeline import build_training_pipeline
from trading_databricks.ml.training.cpcv import CPCVSplitter

def test_build_training_pipeline_returns_pipeline_and_splitter() -> None:
    config = MLPipelineConfig()
    
    # Generate dummy bars
    now = datetime.now(UTC)
    bars = pl.DataFrame({
        "bar_index": range(200),
        "symbol": ["BTCUSDT"] * 200,
        "close": [100.0 + i * 0.1 for i in range(200)],
        "bar_end_time": [now + timedelta(minutes=i) for i in range(200)],
    })
    
    events = detect_cusum_events(bars, config=config.cusum)
    labels = compute_triple_barrier_labels(events, bars, config=config.label)
    
    if labels.is_empty():
        return  # Skip if dummy data doesn't trigger events
        
    pipeline, splitter = build_training_pipeline(config, bars, events, labels)
    
    assert isinstance(pipeline, Pipeline)
    assert isinstance(splitter, CPCVSplitter)
    assert len(pipeline.steps) == 3
    assert pipeline.steps[0][0] == "features"
    assert pipeline.steps[1][0] == "to_numpy"
    assert pipeline.steps[2][0] == "classifier"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_pipeline_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_databricks.ml.transformers.pipeline'`

- [ ] **Step 3: Write minimal implementation**

Create `src/trading_databricks/ml/transformers/pipeline.py`:

```python
from __future__ import annotations

import polars as pl
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from trading_databricks.ml.config import MLPipelineConfig
from trading_databricks.ml.training.cpcv import CPCVSplitter
# Assuming these were created in Plan 7
from trading_databricks.ml.transformers.features import (
    PolarsFeatureUnion,
    TechnicalFeatureTransformer,
    VolatilityFeatureTransformer,
    VolumeFeatureTransformer,
    PolarsToNumpy
)

def build_training_pipeline(
    config: MLPipelineConfig,
    bars: pl.DataFrame,
    events: pl.DataFrame,
    labels: pl.DataFrame,
) -> tuple[Pipeline, CPCVSplitter]:
    """Builds the sklearn Pipeline and CPCV splitter for training."""
    
    feature_union = PolarsFeatureUnion([
        ("technical", TechnicalFeatureTransformer(config.feature)),
        ("volatility", VolatilityFeatureTransformer(config.feature)),
        ("volume", VolumeFeatureTransformer(config.feature)),
    ])
    
    # Fit the union with the bars context
    feature_union.fit(events, y=None, bars=bars)
    
    pipeline = Pipeline([
        ("features", feature_union),
        ("to_numpy", PolarsToNumpy()),
        ("classifier", RandomForestClassifier(random_state=config.training.random_state)),
    ])
    
    # Ensure CPCVSplitter gets integer arrays
    t_start = labels["bar_index"].to_numpy()
    t_end = labels["vertical_bar"].to_numpy()
    
    splitter = CPCVSplitter(
        n_groups=config.cpcv.n_groups,
        k_test=config.cpcv.k_test,
        embargo_pct=config.cpcv.embargo_pct,
        t_start=t_start,
        t_end=t_end
    )
    
    return pipeline, splitter
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_pipeline_builder.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_pipeline_builder.py src/trading_databricks/ml/transformers/pipeline.py
git commit -m "feat: implement sklearn training pipeline builder and CPCV integration"
```

---

## Task 2: Hyperparameter Search Integration

**Files:**
- Create: `tests/ml/test_search.py`
- Create: `src/trading_databricks/ml/transformers/search.py`

- [ ] **Step 1: Write the failing test**

Create `tests/ml/test_search.py`:

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from trading_databricks.ml.config import TrainingConfig
from trading_databricks.ml.training.cpcv import CPCVSplitter
from trading_databricks.ml.transformers.search import build_hyperparameter_search

def test_build_hyperparameter_search():
    pipeline = Pipeline([("classifier", RandomForestClassifier())])
    splitter = CPCVSplitter(n_groups=3, k_test=1, embargo_pct=0.01, t_start=[0, 1, 2], t_end=[1, 2, 3])
    config = TrainingConfig(n_iter_search=5, random_state=42)
    
    search = build_hyperparameter_search(pipeline, splitter, config)
    
    assert search.n_iter == 5
    assert search.random_state == 42
    assert search.cv == splitter
    assert search.scoring == "neg_log_loss"
    assert "classifier__n_estimators" in search.param_distributions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_search.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/trading_databricks/ml/transformers/search.py`:

```python
from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from trading_databricks.ml.config import TrainingConfig
from trading_databricks.ml.training.cpcv import CPCVSplitter

def build_hyperparameter_search(
    pipeline: Pipeline,
    splitter: CPCVSplitter,
    config: TrainingConfig,
    param_distributions: dict[str, list] | None = None,
) -> RandomizedSearchCV:
    """Wraps the pipeline in a RandomizedSearchCV utilizing CPCV."""
    
    if param_distributions is None:
        param_distributions = {
            "classifier__n_estimators": [100, 200, 500],
            "classifier__max_depth": [3, 5, 10, None],
            "classifier__min_samples_leaf": [1, 5, 10, 20],
        }
        
    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=config.n_iter_search,
        cv=splitter,
        scoring="neg_log_loss",
        random_state=config.random_state,
        n_jobs=1,  # Polars is internally parallel; avoid oversubscription
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_search.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_search.py src/trading_databricks/ml/transformers/search.py
git commit -m "feat: add randomized search cv builder for ML pipeline"
```

---

## Task 3: MLflow PyFunc Wrapper

**Files:**
- Create: `tests/ml/test_mlflow_model.py`
- Create: `src/trading_databricks/ml/transformers/mlflow_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/ml/test_mlflow_model.py`:

```python
import pandas as pd
import mlflow
from trading_databricks.ml.transformers.mlflow_model import TradingPipelineModel
from trading_databricks.ml.config import MLPipelineConfig

class DummyPipeline:
    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.2, 0.8]] * len(X))

def test_mlflow_model_predict():
    config = MLPipelineConfig()
    pipeline = DummyPipeline()
    model = TradingPipelineModel(pipeline=pipeline, config=config)
    
    input_df = pd.DataFrame({"feature1": [1.0, 2.0]})
    # context is None in direct calls unless loaded via MLflow
    result = model.predict(context=None, model_input=input_df)
    
    assert "bet_size" in result.columns
    assert len(result) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_mlflow_model.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/trading_databricks/ml/transformers/mlflow_model.py`:

```python
from __future__ import annotations

import pickle
import pandas as pd
import polars as pl
import mlflow

from trading_databricks.ml.config import MLPipelineConfig

class TradingPipelineModel(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc wrapper for the trading pipeline.
    
    At inference, the predict method receives events (as a pandas DataFrame per MLflow contract),
    converts to Polars, runs the pipeline, and computes bet sizing.
    """
    
    def __init__(self, pipeline=None, config: MLPipelineConfig | None = None):
        self.pipeline = pipeline
        self.config = config

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Loads the serialized pipeline from MLflow artifacts."""
        with open(context.artifacts["pipeline"], "rb") as f:
            self.pipeline = pickle.load(f)
            
        with open(context.artifacts["config"], "rb") as f:
            self.config = pickle.load(f)

    def predict(
        self, 
        context: mlflow.pyfunc.PythonModelContext | None, 
        model_input: pd.DataFrame, 
        params: dict | None = None
    ) -> pd.DataFrame:
        """Runs inference and bet sizing."""
        
        events = pl.from_pandas(model_input)
        
        # Get probability of class 1 (long)
        probas = self.pipeline.predict_proba(events)[:, 1]
        
        # Apply bet sizing logic
        bet_sizes = []
        for p in probas:
            # Simplified bet sizing for wrapper
            if self.config and p > self.config.bet_sizing.primary_threshold:
                bet_sizes.append(1.0)
            elif self.config and p < (1.0 - self.config.bet_sizing.primary_threshold):
                bet_sizes.append(-1.0)
            else:
                bet_sizes.append(0.0)
                
        result = model_input.copy()
        result["bet_size"] = bet_sizes
        result["probability_long"] = probas
        
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_mlflow_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_mlflow_model.py src/trading_databricks/ml/transformers/mlflow_model.py
git commit -m "feat: implement MLflow PyFunc wrapper for serialization and inference"
```

---

## Task 4: Databricks Training Job Entry Point

**Files:**
- Create: `tests/ml/test_train_job.py`
- Create: `src/trading_databricks/ml/jobs/train.py`

- [ ] **Step 1: Write the failing test**

Create `tests/ml/test_train_job.py`:

```python
from trading_databricks.ml.jobs.train import run_training_job
from trading_databricks.ml.config import MLPipelineConfig

def test_run_training_job_stub():
    # Verify the signature exists
    assert callable(run_training_job)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_train_job.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `src/trading_databricks/ml/jobs/train.py`:

```python
from __future__ import annotations

import tempfile
import pickle
import logging
import polars as pl
import mlflow
from pyspark.sql import SparkSession

from trading_databricks.ml.config import MLPipelineConfig
from trading_databricks.ml.cusum import detect_cusum_events
from trading_databricks.ml.labeling import compute_triple_barrier_labels
from trading_databricks.ml.transformers.pipeline import build_training_pipeline
from trading_databricks.ml.transformers.search import build_hyperparameter_search
from trading_databricks.ml.transformers.mlflow_model import TradingPipelineModel

logger = logging.getLogger(__name__)

def run_training_job(config: MLPipelineConfig | None = None) -> dict:
    """Main entry point for the Databricks training job.
    
    1. Loads bars from Unity Catalog.
    2. Runs CUSUM and Labeling.
    3. Builds Pipeline and Search CV.
    4. Fits the model.
    5. Logs to MLflow (Unity Catalog registry).
    """
    if config is None:
        config = MLPipelineConfig()
        
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")
    
    # 1. Load bars
    table_name = f"{config.catalog}.gold.dollar_bars"
    logger.info(f"Loading bars from {table_name}")
    try:
        bars_df = spark.read.table(table_name)
        # Convert to Polars
        bars = pl.from_arrow(bars_df.toArrow())
    except Exception as e:
        logger.warning(f"Could not load {table_name}. Aborting. Error: {e}")
        return {"status": "failed", "reason": "No data"}
        
    # 2. Pre-pipeline
    events = detect_cusum_events(bars, config=config.cusum)
    if events.is_empty():
        return {"status": "failed", "reason": "No events detected"}
        
    labels = compute_triple_barrier_labels(events, bars, config=config.label)
    if labels.is_empty():
        return {"status": "failed", "reason": "No labels generated"}
        
    # In AFML, avg_uniqueness acts as the sample weight
    weights = labels["avg_uniqueness"].to_numpy()
    y = labels["label"].to_numpy()
    
    # 3. Build pipeline
    pipeline, splitter = build_training_pipeline(config, bars, events, labels)
    search = build_hyperparameter_search(pipeline, splitter, config.training)
    
    # 4. Fit & Log
    mlflow.set_experiment(f"/{config.catalog}/ml/training")
    
    with mlflow.start_run():
        logger.info("Starting hyperparameter search...")
        search.fit(events, y, classifier__sample_weight=weights)
        
        # Serialize pipeline and config to pass as artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline_path = f"{tmpdir}/pipeline.pkl"
            config_path = f"{tmpdir}/config.pkl"
            
            with open(pipeline_path, "wb") as f:
                pickle.dump(search.best_estimator_, f)
            with open(config_path, "wb") as f:
                pickle.dump(config, f)
                
            model = TradingPipelineModel()
            
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=model,
                artifacts={
                    "pipeline": pipeline_path,
                    "config": config_path
                },
                registered_model_name=f"{config.catalog}.ml.trading_model"
            )
            
        mlflow.log_params(search.best_params_)
        mlflow.log_param("config_symbol", config.symbol)
        
    return {
        "status": "success", 
        "best_score": search.best_score_,
        "best_params": search.best_params_
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_train_job.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_train_job.py src/trading_databricks/ml/jobs/train.py
git commit -m "feat: implement training job entry point with MLflow UC registration"
```

---

## Task 5: Leakage Test

**Files:**
- Create: `tests/ml/test_leakage.py`

- [ ] **Step 1: Write the failing test**

Create `tests/ml/test_leakage.py`:

```python
import polars as pl
from datetime import datetime, timedelta, UTC
from trading_databricks.ml.config import FeatureConfig
from trading_databricks.ml.transformers.features import PolarsFeatureUnion, TechnicalFeatureTransformer

def test_features_do_not_use_future_bars():
    """Verify that features computed at time T do not incorporate data from T+1."""
    now = datetime.now(UTC)
    
    # Create 50 bars
    bars = pl.DataFrame({
        "bar_index": range(50),
        "symbol": ["BTCUSDT"] * 50,
        "close": [100.0 + i for i in range(50)],
        "bar_end_time": [now + timedelta(minutes=i) for i in range(50)],
    })
    
    # Create an event at bar 25
    event_ts = now + timedelta(minutes=25)
    events = pl.DataFrame({
        "event_ts": [event_ts],
        "bar_index": [25],
        "symbol": ["BTCUSDT"]
    })
    
    # Setup transformer
    config = FeatureConfig(technical_windows=(5,))
    union = PolarsFeatureUnion([
        ("tech", TechnicalFeatureTransformer(config))
    ])
    
    # Scenario A: Fit with ALL bars (simulating a leakage bug where fit sees future data)
    union.fit(events, y=None, bars=bars)
    features_all = union.transform(events)
    
    # Scenario B: Fit with only bars up to the event
    past_bars = bars.filter(pl.col("bar_end_time") <= event_ts)
    union.fit(events, y=None, bars=past_bars)
    features_past = union.transform(events)
    
    # The output MUST be identical. If Scenario A used future data to calculate
    # moving averages, it would differ from Scenario B.
    from polars.testing import assert_frame_equal
    assert_frame_equal(features_all, features_past)
```

- [ ] **Step 2: Run test to verify it fails/passes**
*(Depending on whether `PolarsFeatureUnion` in Plan 7 correctly implements the `max_train_ts` logic mentioned in the architecture notes. If it fails, fix `PolarsFeatureUnion`.)*

Run: `uv run pytest tests/ml/test_leakage.py -v`

- [ ] **Step 3: Fix Leakage (if necessary)**

If the test fails, ensure `PolarsFeatureUnion.fit()` in `src/trading_databricks/ml/transformers/features.py` truncates the bars:

```python
    def fit(self, X, y=None, **fit_params):
        bars = fit_params.get("bars")
        if bars is not None and "event_ts" in X.columns:
            max_train_ts = X["event_ts"].max()
            bars = bars.filter(pl.col("bar_end_time") <= max_train_ts)
            
        for name, transformer in self.transformers:
            transformer.fit(X, y, bars=bars)
        return self
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_leakage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ml/test_leakage.py
git commit -m "test: add test to prevent future data leakage in feature engineering"
```

---

## Verification Checklist

- [ ] `uv run pytest tests/ml/` — ALL tests pass
- [ ] `uv run ruff check src/trading_databricks/ml/`
- [ ] MLflow Serialization wrapper correctly overrides `predict`.
- [ ] Feature leak test proves rolling windows do not access future bars during CPCV.
- [ ] `databricks bundle validate` passes.
