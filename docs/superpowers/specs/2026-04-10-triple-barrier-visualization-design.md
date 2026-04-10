# Triple Barrier Visualization Design

## Purpose
Add a high-fidelity visual representation of triple-barrier labels to the ML end-to-end QA training job (`notebooks/qa/ml/99_e2e_training_job.py`). This will allow researchers to visually verify that the stateful triple barrier labeling logic is correctly triggering events, establishing boundaries, and capturing the path-dependent outcomes in a real market context.

## Constraints
- Must use the McKinsey color palette:
  - #f5f5f5 (Light Gray) - Background (now applied globally via Plotly theme)
  - #222222 (Dark Gray / Black) - Price History Line & Text (now default via theme)
  - #005eb8 (Blue) - Buying point (Event trigger/entry)
  - #f3c13a (Yellow/Gold) - Stop Trade / Cutloss point (Lower barrier / Time barrier limits)
  - #a2aaad (Medium Gray) - Secondary muted elements / Time barrier
- Must display 5-10 sampled CUSUM events with their trajectories, barrier lines, and hit points overlayed on a single combined plot.
- Must provide a context margin (price history slightly before the first event and after the last event's barrier hit).
- Underlying price data must be displayed as a Close Price Line, not candlesticks.

## Architecture / Implementation Plan
1. **Global Theming**: We have already modified `notebooks/qa/utils.py` to register and set the "mckinsey" Plotly template globally, guaranteeing `#f5f5f5` backgrounds, `#222222` text/axes, and the correct colorway.
2. **Helper Function**: Create a new visualization builder in `notebooks/qa/utils.py` (e.g. `build_triple_barrier_chart(events, labels, bars, n_samples=5)`).
3. **Data Selection**:
   - Randomly or sequentially select `n_samples` from the `labels` DataFrame where the barrier hit successfully (so we can see full trajectories).
   - Find the absolute minimum `event_ts` and maximum `matured_at` from the sampled labels.
   - Add a 10% time margin (or N bars) on both sides of this window.
   - Filter the `bars` DataFrame to just this window.
4. **Plotting Layers**:
   - **Trace 1 (Price)**: Add a line trace for the `close` price across the sampled window (`#222222`).
   - **Trace 2 (Entries)**: Add scatter points at `event_ts` indicating the CUSUM trigger / entry points (`#005eb8`).
   - **Barrier Shapes (Lines)**: For each event, draw horizontal line segments for the upper barrier (`#005eb8`) and lower barrier (`#f3c13a`), and a vertical line segment for the time barrier (`#a2aaad`).
   - **Hit Markers**: Plot a distinct marker at `matured_at` colored by which barrier was hit (Blue for upper, Yellow for lower, Gray for time).
5. **Integration**: Call this new helper in `99_e2e_training_job.py` within "Step 3: Triple Barrier Labeling" and log it to MLflow via `log_plotly()`.
