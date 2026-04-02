# ML Signal Generation Research Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a Databricks research notebook script to extract 2 million rows of tick data, construct dollar bars entirely in memory using Polars, engineer momentum and volatility features, and train an MLflow-tracked scikit-learn baseline model predicting 5-bar forward returns.

**Architecture:** A single, isolated research script that queries Unity Catalog from Databricks PySpark, converts the payload to Polars for fast grouped aggregation (dollar bars) and window operations (features/target). It utilizes chronological splitting to evaluate a RandomForestRegressor baseline without lookahead bias, tracking all metrics and artifacts in MLflow.

**Tech Stack:** Databricks PySpark, Polars, scikit-learn, MLflow.

---

### Task 1: Setup & PySpark to Polars Extraction

**Files:**
- Create: `.worktrees/aaronngdev/trd-16-build-ml-signal-generation-research-notebook/notebooks/research_ml_signal_gen.py`

- [ ] **Step 1: Write the PySpark to Polars extraction script**

```python
# notebooks/research_ml_signal_gen.py
import polars as pl
from pyspark.sql import SparkSession

def extract_data(spark: SparkSession, limit: int = 2000000) -> pl.DataFrame:
    """Extract tick data from bronze layer and convert to Polars."""
    df = (
        spark.table("trading-signal-gen-dev.bronze.raw_aggtrades_samples")
        .orderBy("timestamp")
        .limit(limit)
    )
    # Convert efficiently using toPandas() which leverages Arrow under the hood
    return pl.from_pandas(df.toPandas())

if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    
    print("Extracting data...")
    # Use a smaller limit for testing the extraction step
    raw_df = extract_data(spark, limit=10000)
    
    print(f"Extracted {raw_df.height} rows.")
    assert raw_df.height > 0, "DataFrame is empty"
    assert "price" in raw_df.columns and "quantity" in raw_df.columns, "Missing required columns"
```

- [ ] **Step 2: Run script to verify extraction**

Run: Use the `mcp_databricks_run_python_file_on_databricks` tool with `file_path="notebooks/research_ml_signal_gen.py"`.
Expected: Output showing "Extracted 10000 rows." without assertion errors.

- [ ] **Step 3: Commit**

```bash
git add notebooks/research_ml_signal_gen.py
git commit -m "feat: add PySpark to Polars tick data extraction"
```

---

### Task 2: Polars Dollar Bar Construction

**Files:**
- Modify: `.worktrees/aaronngdev/trd-16-build-ml-signal-generation-research-notebook/notebooks/research_ml_signal_gen.py`

- [ ] **Step 1: Append dollar bar construction logic**

```python
# Append to notebooks/research_ml_signal_gen.py

def construct_dollar_bars(df: pl.DataFrame, threshold: float = 500_000.0) -> pl.DataFrame:
    """Construct dollar bars using Polars."""
    return (
        df.with_columns([
            (pl.col("price") * pl.col("quantity")).alias("dollar_value")
        ])
        .with_columns([
            pl.col("dollar_value").cum_sum().alias("cum_dollar_value")
        ])
        .with_columns([
            (pl.col("cum_dollar_value") // threshold).cast(pl.Int64).alias("bar_id")
        ])
        .group_by("bar_id")
        .agg([
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("quantity").sum().alias("volume"),
            pl.col("timestamp").last().alias("timestamp")
        ])
        .sort("bar_id")
    )

if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    
    print("Extracting data...")
    raw_df = extract_data(spark, limit=10000)
    
    print("Constructing dollar bars...")
    # Use a smaller threshold for the test limit
    bars_df = construct_dollar_bars(raw_df, threshold=5000.0) 
    
    print(f"Constructed {bars_df.height} bars.")
    assert bars_df.height > 0, "No bars constructed"
    assert all(c in bars_df.columns for c in ["open", "high", "low", "close", "volume"]), "Missing OHLCV columns"
```

*(Note: Replace the previous `if __name__ == "__main__":` block with the updated one above.)*

- [ ] **Step 2: Run script to verify bar construction**

Run: Use the `mcp_databricks_run_python_file_on_databricks` tool with `file_path="notebooks/research_ml_signal_gen.py"`.
Expected: Output showing "Constructed X bars." without assertion errors.

- [ ] **Step 3: Commit**

```bash
git add notebooks/research_ml_signal_gen.py
git commit -m "feat: add Polars dollar bar construction logic"
```

---

### Task 3: Polars Feature Engineering & Target

**Files:**
- Modify: `.worktrees/aaronngdev/trd-16-build-ml-signal-generation-research-notebook/notebooks/research_ml_signal_gen.py`

- [ ] **Step 1: Append feature and target engineering logic**

```python
# Append to notebooks/research_ml_signal_gen.py

def engineer_features_and_target(bars_df: pl.DataFrame) -> pl.DataFrame:
    """Engineer momentum/volatility features and 5-bar forward return target."""
    return (
        bars_df.with_columns([
            # 1-bar return for volatility calc
            (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret_1"),
            # 5-bar momentum (return)
            (pl.col("close") / pl.col("close").shift(5) - 1.0).alias("momentum_5"),
            # Target: 5-bar forward return
            (pl.col("close").shift(-5) / pl.col("close") - 1.0).alias("target_5")
        ])
        .with_columns([
            # 10-bar rolling standard deviation of 1-bar returns
            pl.col("ret_1").rolling_std(window_size=10).alias("volatility_10")
        ])
        .drop_nulls()
    )

if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    
    print("Extracting data...")
    raw_df = extract_data(spark, limit=10000)
    
    print("Constructing dollar bars...")
    bars_df = construct_dollar_bars(raw_df, threshold=5000.0)
    
    print("Engineering features...")
    feat_df = engineer_features_and_target(bars_df)
    
    print(f"Features engineered. Final rows: {feat_df.height}")
    assert "momentum_5" in feat_df.columns, "Missing momentum"
    assert "volatility_10" in feat_df.columns, "Missing volatility"
    assert "target_5" in feat_df.columns, "Missing target"
    assert feat_df.null_count().sum_horizontal()[0] == 0, "Nulls remain in dataset"
```

*(Note: Replace the previous `if __name__ == "__main__":` block with the updated one above.)*

- [ ] **Step 2: Run script to verify features and target**

Run: Use the `mcp_databricks_run_python_file_on_databricks` tool with `file_path="notebooks/research_ml_signal_gen.py"`.
Expected: Output showing "Features engineered. Final rows: X" without assertion errors.

- [ ] **Step 3: Commit**

```bash
git add notebooks/research_ml_signal_gen.py
git commit -m "feat: add feature engineering and 5-bar forward return target"
```

---

### Task 4: MLflow Baseline Training

**Files:**
- Modify: `.worktrees/aaronngdev/trd-16-build-ml-signal-generation-research-notebook/notebooks/research_ml_signal_gen.py`

- [ ] **Step 1: Write the MLflow and scikit-learn training logic**

```python
# OVERWRITE notebooks/research_ml_signal_gen.py completely with this finalized version
import polars as pl
from pyspark.sql import SparkSession
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def extract_data(spark: SparkSession, limit: int = 2000000) -> pl.DataFrame:
    df = (
        spark.table("trading-signal-gen-dev.bronze.raw_aggtrades_samples")
        .orderBy("timestamp")
        .limit(limit)
    )
    return pl.from_pandas(df.toPandas())

def construct_dollar_bars(df: pl.DataFrame, threshold: float = 500_000.0) -> pl.DataFrame:
    return (
        df.with_columns([
            (pl.col("price") * pl.col("quantity")).alias("dollar_value")
        ])
        .with_columns([
            pl.col("dollar_value").cum_sum().alias("cum_dollar_value")
        ])
        .with_columns([
            (pl.col("cum_dollar_value") // threshold).cast(pl.Int64).alias("bar_id")
        ])
        .group_by("bar_id")
        .agg([
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("quantity").sum().alias("volume"),
            pl.col("timestamp").last().alias("timestamp")
        ])
        .sort("bar_id")
    )

def engineer_features_and_target(bars_df: pl.DataFrame) -> pl.DataFrame:
    return (
        bars_df.with_columns([
            (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret_1"),
            (pl.col("close") / pl.col("close").shift(5) - 1.0).alias("momentum_5"),
            (pl.col("close").shift(-5) / pl.col("close") - 1.0).alias("target_5")
        ])
        .with_columns([
            pl.col("ret_1").rolling_std(window_size=10).alias("volatility_10")
        ])
        .drop_nulls()
    )

def train_baseline(df: pl.DataFrame, threshold: float):
    print("Preparing data for ML...")
    # Chronological split (70% train / 30% test) - NO shuffling
    split_idx = int(df.height * 0.7)
    train_df = df.slice(0, split_idx)
    test_df = df.slice(split_idx, df.height - split_idx)
    
    features = ["momentum_5", "volatility_10"]
    X_train = train_df[features].to_numpy()
    y_train = train_df["target_5"].to_numpy()
    
    X_test = test_df[features].to_numpy()
    y_test = test_df["target_5"].to_numpy()
    
    n_estimators = 100
    
    print("Training model...")
    mlflow.set_experiment("/Shared/ml_signal_gen_research")
    with mlflow.start_run():
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("n_estimators", n_estimators)
        
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "random_forest_baseline")
        
        print(f"Training complete. MSE: {mse:.6f}, R2: {r2:.6f}")

if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    
    # Production limit constraints
    LIMIT = 2_000_000
    THRESHOLD = 500_000.0
    
    print("Extracting data...")
    raw_df = extract_data(spark, limit=LIMIT)
    
    print("Constructing dollar bars...")
    bars_df = construct_dollar_bars(raw_df, threshold=THRESHOLD)
    
    print("Engineering features...")
    feat_df = engineer_features_and_target(bars_df)
    
    train_baseline(feat_df, threshold=THRESHOLD)
```

- [ ] **Step 2: Run script to verify MLflow training**

Run: Use the `mcp_databricks_run_python_file_on_databricks` tool with `file_path="notebooks/research_ml_signal_gen.py"`.
Expected: Output showing "Training complete. MSE: ..., R2: ..." with no execution errors, indicating that the full 2M extraction, formatting, and MLflow logging run seamlessly in Databricks.

- [ ] **Step 3: Commit**

```bash
git add notebooks/research_ml_signal_gen.py
git commit -m "feat: train and log RandomForest baseline using MLflow"
```
