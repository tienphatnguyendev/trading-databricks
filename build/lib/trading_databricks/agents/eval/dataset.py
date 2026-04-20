"""Evaluation dataset builders for the MAS harness.

Primary corpus stratifies on ``source × token`` with a marginal fallback
when a cell has fewer than ``min_cell`` rows.
"""
from __future__ import annotations

from typing import Literal

import polars as pl

PRIMARY_DATASET_NAME = "smart-trading-dev.04_eval.mas_eval_v1"
CALIBRATION_DATASETS = {
    "cryptolin": "smart-trading-dev.04_eval.mas_eval_direction_cryptolin_v1",
    "fincausal": "smart-trading-dev.04_eval.mas_eval_grounding_fincausal_v1",
    "cnc": "smart-trading-dev.04_eval.mas_eval_grounding_cnc_v1",
    "mechanism_bootstrap": "smart-trading-dev.04_eval.mas_eval_mechanism_bootstrap_v1",
}

Corpus = Literal["primary", "cryptolin", "fincausal", "cnc", "mechanism_bootstrap"]


def _resolve_corpus(dataset_name: str) -> Corpus:
    if dataset_name == PRIMARY_DATASET_NAME:
        return "primary"
    reverse = {v: k for k, v in CALIBRATION_DATASETS.items()}
    if dataset_name in reverse:
        return reverse[dataset_name]
    known = [PRIMARY_DATASET_NAME, *CALIBRATION_DATASETS.values()]
    raise ValueError(f"Unknown dataset: {dataset_name!r}. Expected one of: {known}")


def _stratified_sample(
    df: pl.DataFrame,
    *,
    strata: list[str],
    target_n: int,
    min_cell: int,
    seed: int,
) -> pl.DataFrame:
    """Proportional stratified sample with marginal fallback for sparse cells."""
    cell_counts = df.group_by(strata).len()
    sparse = cell_counts.filter(pl.col("len") < min_cell)
    if sparse.height == 0:
        cells_iter = cell_counts.iter_rows(named=True)
    else:
        # Fallback: drop the higher-cardinality axis, resample marginals
        marginal_axis = strata[0]
        cell_counts = df.group_by([marginal_axis]).len()
        strata = [marginal_axis]
        cells_iter = cell_counts.iter_rows(named=True)

    total = df.height
    out_parts: list[pl.DataFrame] = []
    for row in cells_iter:
        k = max(1, round(target_n * row["len"] / total))
        mask = pl.lit(True)
        for s in strata:
            mask = mask & (pl.col(s) == row[s])
        cell_df = df.filter(mask)
        take = min(k, cell_df.height)
        out_parts.append(cell_df.sample(n=take, seed=seed, shuffle=True))

    out = pl.concat(out_parts)
    if out.height > target_n:
        out = out.sample(n=target_n, seed=seed, shuffle=True)
    return out


def _load_primary_source() -> pl.DataFrame:
    """Pull rows from smart-trading-dev.01_bronze.news_agent_eval_dataset via Spark."""
    from trading_databricks.agents.nodes.few_shot_retriever import get_spark_session

    spark = get_spark_session()
    sdf = spark.table("smart-trading-dev.01_bronze.news_agent_eval_dataset")
    return pl.from_arrow(sdf.collect().to_arrow())


def build_eval_dataset(
    corpus: Corpus = "primary",
    version: str | int = "latest",
    *,
    target_n: int = 200,
    seed: int = 42,
):
    """Return a pandas DataFrame ready for ``mlflow.genai.evaluate(data=...)``.

    Schema: columns ``inputs`` (dict), ``expectations`` (dict), ``tags`` (dict).
    """
    if corpus == "primary":
        df = _load_primary_source()
        sampled = _stratified_sample(
            df,
            strata=["source", "token"],
            target_n=target_n,
            min_cell=5,
            seed=seed,
        )
        records = [
            {
                "inputs": {"headline": r["headline"], "body": r["body"], "symbol": r["token"]},
                "expectations": {},
                "tags": {"source": r["source"], "token": r["token"], "corpus": "primary"},
            }
            for r in sampled.iter_rows(named=True)
        ]
    elif corpus in CALIBRATION_DATASETS:
        # Delegated to per-corpus loader in Task 9
        records = _load_calibration(corpus)
    else:
        raise ValueError(f"unknown corpus: {corpus!r}")

    import pandas as pd

    return pd.DataFrame(records)


_CALIBRATION_EXPECTATIONS = {
    "cryptolin": ("direction_label", "direction"),
    "fincausal": ("causal_span_present", "grounding"),
    "cnc": ("causal_span_present", "grounding"),
    "mechanism_bootstrap": ("mechanism_label", "mechanism"),
}


def _load_calibration(corpus: str) -> list[dict]:
    if corpus not in CALIBRATION_DATASETS:
        raise ValueError(f"unsupported calibration corpus: {corpus!r}")

    from trading_databricks.agents.nodes.few_shot_retriever import get_spark_session

    table = CALIBRATION_DATASETS[corpus]
    label_key, rubric = _CALIBRATION_EXPECTATIONS[corpus]

    spark = get_spark_session()
    sdf = spark.table(table)
    df = pl.from_arrow(sdf.collect().to_arrow())

    records: list[dict] = []
    for r in df.iter_rows(named=True):
        records.append(
            {
                "inputs": {
                    "headline": r.get("headline", ""),
                    "body": r.get("body", ""),
                    "symbol": r.get("token", r.get("symbol", "BTC")),
                },
                "expectations": {label_key: r[label_key]},
                "tags": {
                    "source": r.get("source", corpus),
                    "token": r.get("token", "BTC"),
                    "corpus": corpus,
                    "rubric": rubric,
                },
            }
        )
    return records
