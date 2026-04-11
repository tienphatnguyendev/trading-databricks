# Databricks notebook source
# MAGIC %md
# MAGIC # ML Signal Generation Research Notebook
# MAGIC
# MAGIC This notebook extracts tick data from the bronze layer, constructs Dollar Bars
# MAGIC in memory using Polars, engineers momentum and volatility features, and trains
# MAGIC an MLflow-tracked scikit-learn baseline model predicting 5-bar forward returns.
# MAGIC
# MAGIC **Constraints enforced:**
# MAGIC - Reads from `trading-signal-gen-dev.bronze.raw_aggtrades_samples`
# MAGIC - Uses Polars for all bar construction and feature engineering
# MAGIC - Purely tick/bar data (no news integration in MVP)

# COMMAND ----------
# MAGIC %pip install polars pyarrow scikit-learn mlflow

# COMMAND ----------
import mlflow
import mlflow.sklearn
import polars as pl
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Initialize Spark Session for Databricks
spark = SparkSession.builder.getOrCreate()

# Set up MLflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Extract Data to Polars
# MAGIC We query the Unity Catalog Delta table, filter out corrupted timestamps, sort, limit, and convert to Polars via PyArrow.

def extract_data_to_polars(limit: int = 2_000_000) -> pl.DataFrame:
    print(f"Extracting {limit} rows from Delta to Polars...")
    
    # Read from the Dev Bronze table
    df_spark = spark.table("`smart-trading-dev`.`01_bronze`.`raw_aggtrades_samples`")
    
    # Filter corrupted timestamps and parse strings to actual types
    df_spark_clean = (
        df_spark
        .filter(~F.col("timestamp").startswith("+"))
        .withColumn("timestamp", F.col("timestamp").cast("timestamp"))
        .withColumn("price", F.col("price").cast("double"))
        .withColumn("quantity", F.col("quantity").cast("double"))
        .filter(F.col("price").isNotNull() & F.col("quantity").isNotNull() & F.col("timestamp").isNotNull())
        .orderBy("timestamp")
        .limit(limit)
    )
    
    # Convert Spark DataFrame -> PyArrow Table -> Polars DataFrame
    arrow_table = df_spark_clean.toArrow()
    df_pl = pl.from_arrow(arrow_table)
    
    print(f"Extracted {len(df_pl)} rows successfully.")
    return df_pl

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Dollar Bar Construction
# MAGIC We calculate the dollar value of each trade, accumulate it, and split into bars whenever the threshold is crossed.

def build_dollar_bars(df: pl.DataFrame, threshold: float = 500_000.0) -> pl.DataFrame:
    print(f"Building Dollar Bars with threshold: ${threshold:,.2f}...")
    
    # 1. Calculate individual trade dollar value
    df = df.with_columns(
        (pl.col("price") * pl.col("quantity")).alias("dollar_value")
    )
    
    # 2. Calculate cumulative dollar value
    df = df.with_columns(
        pl.col("dollar_value").cum_sum().alias("cumulative_dollars")
    )
    
    # 3. Assign a bar index based on threshold crossings
    df = df.with_columns(
        (pl.col("cumulative_dollars") // threshold).cast(pl.Int64).alias("bar_id")
    )
    
    # 4. Group by bar_id and aggregate OHLCV + Metadata
    bars_df = (
        df.group_by("bar_id", maintain_order=True)
        .agg(
            pl.col("timestamp").first().alias("bar_start_time"),
            pl.col("timestamp").last().alias("bar_end_time"),
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("quantity").sum().alias("volume"),
            pl.col("dollar_value").sum().alias("dollar_volume"),
            pl.len().alias("tick_count")
        )
    )
    
    print(f"Generated {len(bars_df)} Dollar Bars.")
    return bars_df

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Feature Engineering & Target Creation
# MAGIC We calculate baseline features (momentum, volatility) and a forward-looking target.

def engineer_features_and_target(bars_df: pl.DataFrame) -> pl.DataFrame:
    print("Engineering features and target...")
    
    # Ensure bars are sorted chronologically
    df = bars_df.sort("bar_start_time")
    
    # Calculate features and target using Polars window functions
    df_features = df.with_columns(
        # Features
        ((pl.col("close") / pl.col("close").shift(5)) - 1).alias("momentum_5b"),
        ((pl.col("close") / pl.col("close").shift(1)) - 1).rolling_std(window_size=10).alias("volatility_10b"),
        
        # Target: Forward 5-bar return
        ((pl.col("close").shift(-5) / pl.col("close")) - 1).alias("target_5_fwd")
    )
    
    # Drop rows with NaN/Nulls (caused by rolling windows and shifts)
    df_clean = df_features.drop_nulls()
    
    print(f"Final dataset has {len(df_clean)} rows.")
    return df_clean

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. MLflow Baseline Model Training
# MAGIC Train a baseline Random Forest and log it to MLflow. No shuffling allowed!

def train_baseline_model(df: pl.DataFrame, threshold: float):
    print("Training baseline model...")
    
    # Convert Polars to Pandas for Scikit-Learn
    pdf = df.to_pandas()
    
    # Features and Target
    feature_cols = ["open", "high", "low", "close", "volume", "dollar_volume", "tick_count", "momentum_5b", "volatility_10b"]
    target_col = "target_5_fwd"
    
    X = pdf[feature_cols]
    y = pdf[target_col]
    
    # Chronological Split (NO SHUFFLE to prevent lookahead bias)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    with mlflow.start_run(run_name="dollar_bar_rf_baseline"):
        # Model Params
        n_estimators = 50
        max_depth = 5
        
        # Log parameters
        mlflow.log_param("bar_type", "dollar_bar")
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Train
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict & Evaluate
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        # Log metrics
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_r2", r2)
        
        # Log model
        mlflow.sklearn.log_model(model, "rf_baseline_model")
        
        print(f"Training Complete. Test MSE: {mse:.6f}, Test R2: {r2:.4f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Execution Pipeline

if __name__ == "__main__":
    # Hyperparameters
    ROW_LIMIT = 2_000_000
    DOLLAR_THRESHOLD = 500_000.0
    
    # Execute Pipeline
    raw_pl_df = extract_data_to_polars(limit=ROW_LIMIT)
    
    if not raw_pl_df.is_empty():
        bars_df = build_dollar_bars(raw_pl_df, threshold=DOLLAR_THRESHOLD)
        features_df = engineer_features_and_target(bars_df)
        train_baseline_model(features_df, threshold=DOLLAR_THRESHOLD)
    else:
        print("No valid data extracted. Check upstream table or filter logic.")
