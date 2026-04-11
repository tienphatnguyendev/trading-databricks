# QA Notebook Suite Design for AFML Pipeline

## Objective
Build a comprehensive suite of Databricks notebooks to test, QA, and visualize every component of the ML pipeline located in `@src/trading_databricks/ml/**`. The suite will ensure mathematical correctness, point-in-time safety, and provide a clear comparison of the Advanced Financial Machine Learning (AFML) techniques against a naive baseline.

## Constraints & Context
*   **Environment:** Databricks Workspace (URL: `https://dbc-cbfc9b81-4abc.cloud.databricks.com/editor/notebooks/4210131968845413?o=7474659189103028`).
*   **Data Source:** Must use `smart-trading-dev.others.ml_training_data` and relevant bronze tables (e.g., `smart-trading-dev.01_bronze.raw_aggtrades_samples`).
*   **Frameworks:** Polars for data manipulation, Scikit-learn for modeling, MLflow for tracking, Plotly/Matplotlib for visualizations.
*   **Safety:** Data loading must handle memory constraints (slice by time/symbol before `.toArrow()`).

## Architecture: The 6-Notebook Modular Suite

The QA process is split into 6 modular notebooks to isolate debugging, match the source code structure, and culminate in a performance tear sheet.

### Notebook 1: Dollar Bar Construction QA (`01_dollar_bar_construction_qa.py`)
*   **Purpose:** Validate the aggregation of raw tick/aggtrade data into information-driven Dollar Bars.
*   **Execution:** Load raw ticks, apply dollar volume thresholding, and aggregate OHLCV.
*   **Assertions:**
    *   Accumulated dollar volume per bar $\ge$ threshold.
    *   OHLC logic is correct (High $\ge$ Low).
    *   `bar_index` is strictly monotonically increasing.
*   **Visualizations:** Scatter plot of ticks overlaid with the resulting candlestick chart; histogram of accumulated volume overshoots.

### Notebook 2: CUSUM & Labeling QA (`02_cusum_and_labeling_qa.py`)
*   **Purpose:** Verify the symmetric CUSUM filter and Triple-Barrier labeling method.
*   **Execution:** Run `detect_cusum_events` and `compute_triple_barrier_labels` on validated Dollar Bars.
*   **Assertions:**
    *   No events during the `lookback_days` warm-up period.
    *   Barrier hits mathematically match the dynamic EWMA volatility bounds or vertical expiration.
    *   `avg_uniqueness` is correctly bounded `(0.0, 1.0]`.
*   **Visualizations:** Candlestick chart with dynamic EWMA bands; scatter plot of CUSUM Up/Down triggers; histogram of event uniqueness.

### Notebook 3: Feature Engineering QA (`03_feature_engineering_qa.py`)
*   **Purpose:** Ensure Polars feature transformations (Technical, Volatility, Volume, News) are mathematically sound and strictly point-in-time (PIT).
*   **Execution:** Apply feature transformers to the CUSUM events and Dollar Bars.
*   **Assertions:**
    *   Strict PIT safety: Features for event $T$ only use bar data $\le T$.
    *   Graceful `NaN` handling for early rolling windows.
    *   Volatility measures (Parkinson, Garman-Klass) are non-negative.
*   **Visualizations:** Correlation heatmap of all features to check multicollinearity; time-series plots of price vs. specific indicators (e.g., MACD, Bollinger %B).

### Notebook 4: CPCV & Training Pipeline QA (`04_cpcv_and_training_qa.py`)
*   **Purpose:** Validate Combinatorial Purged Cross-Validation (CPCV) to ensure no data leakage between train/test splits.
*   **Execution:** Initialize `CPCVSplitter` using label bounds (`bar_index` to `vertical_bar`) and generate splits.
*   **Assertions:**
    *   Purge window is respected: `max(train_end_time)` strictly $\le$ `min(test_start_time) - purge_window`.
    *   Embargo is respected after test sets.
*   **Visualizations:** Gantt chart (horizontal bars) illustrating Train (Blue), Test (Orange), Purged (Red), and Embargoed (Grey) periods across time for all generated folds.

### Notebook 5: Inference & Bet Sizing QA (`05_inference_and_bet_sizing_qa.py`)
*   **Purpose:** Verify that meta-label probabilities correctly discretize into bet sizes and positions.
*   **Execution:** Run `compute_bet_sizes` and `compute_positions` using mock or trained probabilities.
*   **Assertions:**
    *   Predictions below `primary_threshold` yield $0$ bet size.
    *   Discretized weights strictly match the configured tiers.
*   **Visualizations:** Step-function plot of `meta_confidence` vs. `discretized_weight`; time-series of position state (`flat` vs `long`) and weight magnitude.

### Notebook 6: Baseline vs. AFML Comparison QA (`06_baseline_vs_afml_comparison_qa.py`)
*   **Purpose:** Prove the alpha-generation capability of the AFML techniques via an ablation study against a naive baseline.
*   **Execution:** Run a simplified pipeline (Time Bars, fixed-horizon labels, standard CV, binary sizing) and compare it against the full AFML pipeline on the same out-of-sample data.
*   **Assertions:**
    *   AFML pipeline achieves a higher Probabilistic Sharpe Ratio (PSR).
    *   AFML pipeline exhibits a lower maximum drawdown.
*   **Visualizations (The Tear Sheet):** Cumulative equity curves (Buy & Hold vs. Naive vs. AFML); underwater plot for drawdowns; scatter plot showing trade clustering during high-information regimes vs. uniform baseline trading.

## Next Steps
1.  User review and approval of this spec.
2.  Invoke `writing-plans` skill to generate the implementation plan for the notebooks.
