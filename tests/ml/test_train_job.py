from trading_databricks.ml.jobs.train import run_training_job
from trading_databricks.ml.config import MLPipelineConfig

def test_run_training_job_stub():
    # Verify the signature exists
    assert callable(run_training_job)
