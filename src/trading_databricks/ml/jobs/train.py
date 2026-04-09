from __future__ import annotations

import logging
import pickle
import tempfile

import mlflow
import polars as pl
from pyspark.sql import SparkSession

from trading_databricks.ml.config import MLPipelineConfig
from trading_databricks.ml.cusum import detect_cusum_events
from trading_databricks.ml.labeling import compute_triple_barrier_labels
from trading_databricks.ml.transformers.mlflow_model import TradingPipelineModel
from trading_databricks.ml.transformers.pipeline import build_training_pipeline
from trading_databricks.ml.transformers.search import build_hyperparameter_search

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
