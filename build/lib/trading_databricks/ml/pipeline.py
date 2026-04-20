"""SDP pipeline definition for the ML domain.

Consumes gold tables and produces ML feature tables via materialized views.
This file contains ONLY @dp.materialized_view decorators.

DABs pipeline resource: resources/ml_pipeline.pipeline.yml

TODO: implement after gold tables are populated and feature set is designed.
"""
from __future__ import annotations

# from pyspark import pipelines as dp
# from trading_databricks.ml.features import build_features
