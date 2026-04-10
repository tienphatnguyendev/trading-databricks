# Triple Barrier Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a high-fidelity visual representation of triple-barrier labels overlayed on a price chart inside `notebooks/qa/ml/99_e2e_training_job.py`.

**Architecture:** We will create a `build_triple_barrier_chart` utility function in `notebooks/qa/utils.py`. This function will subset a continuous series of CUSUM events (and their corresponding bars), and construct a Plotly `go.Figure` with traces for the closing price, event entries, and the three barrier levels (upper, lower, time). The QA notebook will invoke this utility and log it via `log_plotly`.

**Tech Stack:** Python, Polars, Plotly, MLflow.

---

### Task 1: Create the visualization helper in `notebooks/qa/utils.py`

**Files:**
- Modify: `notebooks/qa/utils.py`

- [ ] **Step 1: Write the implementation**
Insert the new helper function `build_triple_barrier_chart` into `notebooks/qa/utils.py` right before the `NODE_NAMES` constant.

```python
def build_triple_barrier_chart(events: pl.DataFrame, labels: pl.DataFrame, bars: pl.DataFrame, n_samples: int = 5) -> go.Figure:
    """Build a visualization showing price trajectories and triple barrier bounds for a sample of events."""
    # Ensure data is loaded
    events_df = events.collect() if isinstance(events, pl.LazyFrame) else events
    labels_df = labels.collect() if isinstance(labels, pl.LazyFrame) else labels
    bars_df = bars.collect() if isinstance(bars, pl.LazyFrame) else bars

    # Join labels to events to get barrier levels and hit times
    if "event_ts" in labels_df.columns:
        labeled_events = events_df.join(labels_df, on="event_ts", how="inner")
    else:
        # Fallback if label index is bar_index or something else
        labeled_events = events_df.join(labels_df, left_on="bar_index", right_on="bar_index", how="inner")

    # Sample a continuous subset
    if labeled_events.height == 0:
        return go.Figure()
    
    # Take the first n_samples valid labeled events
    sample = labeled_events.head(n_samples).to_pandas()
    
    # Determine the time window for the chart + 10% margin
    min_ts = sample["event_ts"].min()
    max_ts = sample["matured_at"].max()
    
    time_span = max_ts - min_ts
    margin = time_span * 0.1 if time_span.total_seconds() > 0 else pd.Timedelta(hours=1)
    
    start_window = min_ts - margin
    end_window = max_ts + margin

    # Filter bars to this window
    window_bars = bars_df.filter(
        (pl.col("bar_end_time") >= start_window) & 
        (pl.col("bar_end_time") <= end_window)
    ).to_pandas()

    fig = go.Figure()

    # 1. Base trace: Close price
    fig.add_trace(go.Scatter(
        x=window_bars["bar_end_time"],
        y=window_bars["close"],
        mode="lines",
        name="Close Price",
        line=dict(color="#222222") # mc_dark_gray
    ))

    # 2. Entries (Buying points)
    fig.add_trace(go.Scatter(
        x=sample["event_ts"],
        y=sample["close_bar"],
        mode="markers",
        name="CUSUM Event (Entry)",
        marker=dict(color="#005eb8", size=10, symbol="triangle-up") # mc_blue
    ))

    # 3. Draw Barriers and Hit Markers for each event
    for _, row in sample.iterrows():
        entry_time = row["event_ts"]
        entry_price = row["close_bar"]
        mature_time = row["matured_at"]
        hit_type = row["barrier_hit"] # usually 1 (upper), -1 (lower), 0 (time)
        
        # We need the barrier thresholds. Usually config is applied. 
        # For visualization, we can infer upper/lower from the return if barrier_hit != 0
        # However, to be exact we rely on the hit price. 
        # Since we don't have the exact threshold levels in the return DF standardly,
        # we will plot a horizontal line from entry to mature_time for the Hit price if hit.
        
        hit_price = window_bars.loc[window_bars["bar_end_time"] == mature_time, "close"]
        if not hit_price.empty:
            hit_price = hit_price.values[0]
        else:
            hit_price = entry_price # fallback
            
        # Upper/Lower/Time hit marker
        color = "#a2aaad" # Default time barrier
        name = "Time Barrier Hit"
        if hit_type == 1:
            color = "#005eb8" # Upper
            name = "Take Profit Hit"
        elif hit_type == -1:
            color = "#f3c13a" # Lower
            name = "Stop Loss Hit"

        # Draw a dotted line to the hit point
        fig.add_trace(go.Scatter(
            x=[entry_time, mature_time],
            y=[entry_price, hit_price],
            mode="lines",
            line=dict(color=color, width=1, dash="dot"),
            showlegend=False
        ))
        
        # Draw the hit marker
        fig.add_trace(go.Scatter(
            x=[mature_time],
            y=[hit_price],
            mode="markers",
            name=name,
            marker=dict(color=color, size=12, symbol="x"),
            showlegend=False # We'll just rely on colors, legend gets too cluttered if repeated
        ))
        
    # Add dummy traces for legend entries for the hit markers
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="#005eb8", size=12, symbol="x"), name="Take Profit Hit"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="#f3c13a", size=12, symbol="x"), name="Stop Loss Hit"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="#a2aaad", size=12, symbol="x"), name="Time Barrier Hit"))

    fig.update_layout(
        title="Triple Barrier Labeling Trajectories",
        xaxis_title="Time",
        yaxis_title="Price"
    )
    
    return fig
```

- [ ] **Step 2: Commit**
```bash
git add notebooks/qa/utils.py
git commit -m "feat: add build_triple_barrier_chart visualization helper"
```

### Task 2: Integrate visualization into the QA notebook

**Files:**
- Modify: `notebooks/qa/ml/99_e2e_training_job.py`

- [ ] **Step 1: Import the new helper function**
In `notebooks/qa/ml/99_e2e_training_job.py`, add `build_triple_barrier_chart` to the `notebooks.qa.utils` import block.

```python
from notebooks.qa.utils import (
    build_candlestick,
    build_heatmap,
    build_histogram,
    build_label_distribution,
    build_scatter,
    build_triple_barrier_chart, # <-- Added
    load_raw_dollar_bars,
    log_dataset,
    log_diag,
    log_plotly,
    mlflow_component_run,
)
```

- [ ] **Step 2: Invoke the visualization in Step 3**
In `notebooks/qa/ml/99_e2e_training_job.py`, locate the end of "Step 3: Triple Barrier Labeling" (around the line `avg_holding = ...`). Insert the visualization call:

```python
    avg_holding = (labels["matured_at"] - labels["event_ts"]).mean()
    print_observation(f"Average holding period for a labeled trade is {avg_holding}.")

    # Visualize the triple barrier trajectories
    log_plotly("step3_triple_barrier_trajectories", build_triple_barrier_chart(events, labels, bars, n_samples=5))
    print_observation("Logged interactive triple barrier trajectories visualization showing price paths and barrier hits.")

    print_header("Step 4: Pipeline Assembly & Feature Extraction")
```

- [ ] **Step 3: Run notebook to verify it passes**

Run: `PYTHONPATH=. uv run python notebooks/qa/ml/99_e2e_training_job.py`
Expected: Notebook completes successfully, detecting CUSUM events, labeling, and logging the new `step3_triple_barrier_trajectories.html` artifact without errors.

- [ ] **Step 4: Commit**
```bash
git add notebooks/qa/ml/99_e2e_training_job.py
git commit -m "feat: log triple barrier trajectory visualization in E2E QA job"
```
