05 — Validation Methodology
Cross-references: trading-logic.md (models being validated), docs/daily-workflow.md (S3 data model),
src/trading_databricks/ml/jobs/train.py (training entry point), src/trading_databricks/ticks/schemas.py (bar schema)

Philosophy
Most backtested trading strategies are overfit. Standard CV fails for financial time series because:

Temporal dependence: Returns are not i.i.d. Random splits leak future info.

Label overlap: Triple-barrier labels from overlapping events cause leakage between train/test.

Multiple testing: Trying many strategies and reporting the best inflates apparent Sharpe.

Dollar bars compound these challenges: irregular time spacing means purge windows must be denominated in bars, not clock time.

The platform addresses all three using AFML methodology.

Combinatorially Purged Cross-Validation (CPCV) — AFML Ch. 12
Primary validation method. Generates multiple synthetic backtest paths from CV splits while properly purging leakage.

Default configuration:

Parameter	Value	Rationale
N (number of groups)	6	Balances cost with path diversity
k (test groups per split)	2	Gives C(6,2) = 15 unique train/test combinations
Unique backtest paths	phi[6,2] = 5	Each path concatenates OOS predictions from non-overlapping test groups
Purge window	N_bars	Number of dollar bars spanning the vertical barrier window: avg_bars_per_day x 5 (same adaptive computation as vertical barrier in trading-logic.md). Dollar bars are irregularly spaced in time; purging by bar count prevents leakage through overlapping labels.
Embargo	1% of total sample size	Applied after each test group boundary

Implementation details:

Group formation: Dataset split into N = 6 chronologically ordered groups of ~equal size. Never overlap.

Split generation: For each C(6,2) = 15 combinations of k = 2 test groups, remaining 4 groups = training.

Purging: Training observations whose labels overlap with any test observation (within purge window in bar units) removed.

Embargo: Additional 1% buffer removed after each test group boundary.

Path construction: 15 OOS prediction sets arranged into 5 non-overlapping paths covering full time span.

Path evaluation: Each path evaluated as standalone backtest: returns, Sharpe, max drawdown.

Aggregation: Distribution of path-level metrics (especially Sharpe) informs model quality.

Probabilistic Sharpe Ratio (PSR) — AFML Ch. 14
Tests whether observed Sharpe is statistically distinguishable from benchmark.

text
PSR(SR*) = Phi( (SR_hat - SR*) / sigma_SR )

where:
  SR_hat = observed Sharpe ratio
  SR*    = benchmark Sharpe ratio (0 for "is this positive?")
  sigma_SR = sqrt( (1 - gamma_3 * SR_hat + (gamma_4 - 1)/4 * SR_hat^2) / (T-1) )
  gamma_3  = skewness of returns
  gamma_4  = kurtosis of returns
  T        = number of observations
  Phi      = standard normal CDF

Threshold: PSR > 0.95

Crypto return distributions exhibit heavier tails (excess kurtosis typically 5-15 for BTC dollar bar returns) than equities. The gamma_4 term in sigma_SR is therefore more influential — PSR estimates are wider, making it harder to achieve statistical significance. This is a feature, not a bug: it correctly penalizes strategies that appear profitable only due to fat-tail sampling noise.

Deflated Sharpe Ratio (DSR) — AFML Ch. 14
Adjusts PSR for number of independent trials conducted.

Count total independent trials from silver.trial_registry.

Compute expected maximum Sharpe under null (all strategies have zero true Sharpe) given trial count.

Use expected maximum as benchmark SR* in PSR formula.

Threshold: DSR > 0.90

Walk-Forward Analysis
Additional validation perspective beyond CPCV. Simulates retraining/deployment over time.

Parameter	Value
Training window	Expanding (all available history up to split point)
Retraining frequency	Monthly (30 calendar days — crypto is 24/7, no trading-day concept)
Out-of-sample window	30 calendar days
Purge and embargo	Same as CPCV (bar-denominated)

Secondary to CPCV for promotion decisions. Useful diagnostics about model stability.

Feature Importance Analysis (AFML Ch. 8)
Three methods, all within CPCV framework:

MDI (Mean Decrease Impurity): In-sample, tree-based. Fast but biased toward high-cardinality.

MDA (Mean Decrease Accuracy): Out-of-sample, permutation-based. Unbiased but expensive.

SFI (Single Feature Importance): Out-of-sample, each feature individually. Detects combination-only features.

Rules:

Features with negative MDA (averaged across CPCV splits) → candidates for removal.

Intersection of top-20 from MDI and MDA → "robust feature set."

Recomputed with each model retraining.

Strategy-Failure Probability (AFML Ch. 15)
Estimates likelihood a seemingly profitable strategy is actually unprofitable, accounting for estimation error, non-normality, finite samples.

Threshold: Strategy-failure probability < 10%.

Promotion Thresholds
A model is promoted only if it passes ALL gates:

Metric	Threshold	Source/Rationale
PSR	> 0.95	AFML Ch. 14 — probability true Sharpe > 0
DSR	> 0.90	AFML Ch. 14 — after deflation for multiple trials
Max drawdown	< 25%	Crypto is structurally more volatile than equities
Time under water	< 90 dollar bars	Recovery speed in bar units
Execution-cost robustness	Profitable at 2x estimated costs	Includes Binance maker/taker fees (0.1%) + estimated slippage
Strategy-failure probability	< 10%	AFML Ch. 15
Meta-label precision	> 0.60	Quality of take decisions
Meta-label recall	> 0.50	Avoid excessive filtering
Meta-label F1	> 0.55	Balance precision/recall
Meta-label negative log-loss	< 0.65	Calibration quality

Implementation:

Promotion gate = Lakeflow Job task after CPCV completion.

Reads CPCV results, computes all metrics, writes verdict to silver.promotion_evidence.

Any metric fails → model not promoted, failure reason logged.

All gates pass → auto-registered in MLflow Model Registry with stage = "Staging."

Staging → Production requires second pass on staging catalog data.

Compute Budget for Validation

Activity	Environment	Estimated Cost
Research CPCV iterations	Local MacBook or notebooks/ (non-promotion-grade)	$0
Promotion-grade CPCV	Databricks single-node job cluster	$15-20/month
Walk-forward analysis	Databricks (included)	Included
Feature importance	Databricks (included)	Included

Budget rules:

Reserve $15-20/month for promotion-grade CPCV.

Single-node clusters with scikit-learn only.

Limit Databricks CPCV to 1-2 full sweeps/week for final candidates.

Local/notebook CPCV for research iterations (non-promotion-grade).
