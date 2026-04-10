"""
ml/06_promotion_gates.py — QA notebook for promotion gates evaluation.

Run: uv run python notebooks/qa/ml/06_promotion_gates.py
"""
import mlflow
import numpy as np
import polars as pl

from notebooks.qa.utils import (
    build_equity_curve,
    build_heatmap,
    log_diag,
    log_plotly,
    mlflow_component_run,
)
from trading_databricks.ml.schemas import PROMOTION_EVIDENCE_SCHEMA
from trading_databricks.ml.training.promotion import (
    compute_dsr,
    compute_psr,
    evaluate_promotion_gates,
)


def _generate_scenario(loc: float, scale: float, size: int = 252, n_paths: int = 5):
    rng = np.random.default_rng(seed=int(loc * 100000 + scale * 1000))
    paths = []
    for _ in range(n_paths):
        returns = rng.normal(loc=loc, scale=scale, size=size)
        sharpe = float(np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(252)
        cum = np.cumprod(1 + returns) - 1
        dd = float(np.min(cum - np.maximum.accumulate(cum + 1) + 1))  # simplified max DD
        # Mock time under water
        tuw = 30 if loc > 0 else 120
        paths.append({
            "returns": returns,
            "sharpe": sharpe,
            "max_drawdown": abs(dd),
            "time_under_water": tuw
        })
    return paths

scenarios = {
    "good_strategy": (_generate_scenario(0.0005, 0.015), {"precision": 0.65, "recall": 0.55, "f1": 0.60, "logloss": 0.60}),
    "fragile_strategy": (_generate_scenario(0.0002, 0.02), {"precision": 0.65, "recall": 0.55, "f1": 0.60, "logloss": 0.60}),
    "broken_strategy": (_generate_scenario(-0.0003, 0.025), {"precision": 0.40, "recall": 0.30, "f1": 0.35, "logloss": 0.90})
}

with mlflow_component_run(
    layer="ml", component="promotion_gates",
    params={"n_paths": 5, "n_days": 252}
):
    # Section 2 & 3 & 4 — Exercise component, Diag Tables, and Visuals
    gate_results = []
    for name, (paths, meta) in scenarios.items():
        result = evaluate_promotion_gates(paths, meta, trial_count=10)
        gate_results.append({"scenario": name, **result})

        all_returns = np.concatenate([p["returns"] for p in paths])
        log_plotly(f"equity_{name}", build_equity_curve(all_returns))

        mlflow.log_metrics({
            f"{name}_psr": result["psr"],
            f"{name}_dsr": result["dsr"],
            f"{name}_strategy_failure_prob": result["strategy_failure_prob"],
            f"{name}_max_drawdown": result["max_drawdown"],
            f"{name}_time_under_water": result["time_under_water"],
            f"{name}_all_gates_pass": int(result["all_gates_pass"])
        })

    results_df = pl.DataFrame(gate_results)
    log_diag("gate_results_per_scenario", results_df)

    # PSR DSR grid
    grid_rows = []
    for sh in np.linspace(0, 3, 20):
        for t in [100, 252, 1000]:
            ret = np.random.normal(sh / np.sqrt(t), 0.01, t)
            # Recompute exact observed sharpe for function
            obs_sh = float(np.mean(ret) / np.std(ret, ddof=1)) * np.sqrt(252)
            grid_rows.append({
                "sharpe": obs_sh,
                "n_samples": t,
                "psr": compute_psr(obs_sh, ret),
                "dsr": compute_dsr(obs_sh, ret, n_trials=50)
            })

    grid_df = pl.DataFrame(grid_rows)
    log_diag("psr_dsr_grid", grid_df)

    # 2D Heatmap of PSR Surface — cast sharpe to string so Polars can use as column names
    psr_pivot = (
        grid_df
        .with_columns(pl.col("sharpe").round(3).cast(pl.String).alias("sharpe_key"))
        .pivot(index="n_samples", on="sharpe_key", values="psr")
        .fill_null(0)
    )
    log_plotly("psr_surface", build_heatmap(psr_pivot, corr=False))

    # Section 5 — Sanity assertions
    rng = np.random.default_rng(42)
    assert 0.0 <= compute_psr(1.5, rng.normal(0.01, 0.1, 252)) <= 1.0

    good_result = results_df.filter(pl.col("scenario") == "good_strategy").to_dicts()[0]
    broken_result = results_df.filter(pl.col("scenario") == "broken_strategy").to_dicts()[0]

    assert good_result["all_gates_pass"] is True
    assert broken_result["all_gates_pass"] is False

    req_keys = set(f.name for f in PROMOTION_EVIDENCE_SCHEMA.fields) - {"run_id","model_type","model_version","dataset_version"}
    assert req_keys.issubset(set(good_result.keys()))
