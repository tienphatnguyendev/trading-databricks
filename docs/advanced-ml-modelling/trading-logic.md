02 — Trading Logic
Cross-references: validation.md (CPCV, promotion gates), docs/daily-workflow.md (S3 data model), src/trading_databricks/ticks/schemas.py (tick/bar schema contracts), docs/superpowers/specs/multi-agent-system-for-financial-news-analysis-specs.md (MAS causality layer)

Universe
Instruments: BTC/ETH on Binance (BTCUSDT, ETHUSDT)

Direction: Long-only in Phase 1 (short positions deferred to Phase 2)

Instrument type: Spot (Binance aggTrades)

Settlement: Instant (no T+2 — crypto settles immediately)

Market hours: 24/7, no holidays, no session breaks

"Trading day" for volatility purposes: rolling 24-hour window

Data path: kinesis-tick-streamer -> S3 -> Auto Loader -> bronze.ticks_raw -> silver.ticks_cleaned -> gold.dollar_bars

Unity Catalog: {catalog}.01_bronze.*, {catalog}.silver.*, {catalog}.03_gold.* where catalog = smart-trading-dev | staging | prod

CUSUM Event Detection (AFML Ch. 2)
Standard time-based bars oversample quiet periods and undersample volatile ones. The platform uses CUSUM (cumulative sum) event detection on information-driven dollar bars to generate adaptive observation times.

Parameters:

| Parameter | Default Value | Notes |
|-----------|--------------|-------|
| Input series | Log returns of gold.dollar_bars close prices | log(close_t / close_{t-1}), per symbol |
| Volatility estimator | EWMA with lambda = 0.94 | Applied to dollar bar log returns |
| Threshold | 1x daily volatility, scaled by sqrt(avg_bars_per_day) | avg_bars_per_day = total bars / total calendar days in lookback |
| Reset | Symmetric (positive and negative cumulative sums) | S+ and S- reset on event trigger |
| Minimum inter-event spacing | 5 dollar bars | Prevents clustered events in flash crashes |
| Target event frequency | 20-50 events per asset per month | Monitored; threshold recalibrated if outside range |

Threshold scaling rationale: Because dollar bars sample at information-driven rates (not fixed 1-minute intervals), the classic sqrt(390) denominator is replaced by sqrt(avg_bars_per_day). This adapts the CUSUM threshold to the empirical sampling rate of each asset.

Calibration rules:

CUSUM threshold is a meta-hyperparameter — not optimized via grid search.

Validated through CPCV: if performance is sensitive to +/-20% threshold perturbation, the model is suspect.

Sensitivity analysis runs during Phase 1B.

Event frequency monitored daily. Alert if outside [15, 70] per asset/month for 2 consecutive months.

Event stats tracked in: silver.cusum_event_stats — columns: symbol, date, event_count, threshold_used, ewma_vol, avg_bars_per_day

Events tracked in: silver.cusum_events

Triple-Barrier Labeling (AFML Ch. 3)
Each CUSUM event is labeled using triple barriers applied to dollar bar close prices.

| Barrier | Parameter | Default | Notes |
|---------|-----------|---------|-------|
| Upper (profit-taking) | Profit target | 2x daily vol | From dollar bar returns |
| Lower (stop-loss) | Loss limit | 1x daily vol | Asymmetric by design |
| Vertical (time expiry) | Maximum holding period | N dollar bars ~ 5 calendar days | N = avg_bars_per_day x 5, adaptive, recomputed monthly |

Labeling rules:

Label determined by whichever barrier is touched first.

Upper barrier hit -> label = +1 (profitable long)

Lower barrier hit -> label = -1 (loss)

Vertical barrier hit -> label by sign of return: +1 if positive, -1 if negative, 0 if within +/-0.1x daily vol

Barriers use dollar bar close prices only (not intraday high/low) to avoid lookahead within bar.

All barrier levels computed at event time using only PIT-available information.

Labels table: silver.triple_barrier_labels

Feature Engineering (AFML Ch. 5, 8)
Five feature families:

| Family | Examples | Computed From | Storage |
|--------|----------|---------------|---------|
| Microstructure | VPIN, Kyle's lambda, Amihud illiquidity, roll spread | silver.ticks_cleaned (tick-level) | silver.features_microstructure |
| Technical | RSI, MACD, Bollinger bands, moving averages | gold.dollar_bars (OHLCV) | silver.features_technical |
| Volatility | Realized vol, Parkinson, Garman-Klass | gold.dollar_bars | silver.features_volatility |
| Volume | VWAP deviation, relative volume, OBV | gold.dollar_bars | silver.features_volume |
| News/LLM | treatment_class, mechanism, expected_direction, confidence, validation_status | gold.news_causal_features (MAS) | silver.features_news |

Crypto-specific features (Phase 1B): funding rate, open interest, exchange net flows.

Feature engineering rules:

Fractional differentiation (AFML Ch. 5): Applied to non-stationary features. Parameter d chosen per feature: ADF test rejects unit root at p < 0.05, maximizing correlation with original series.

No forward-looking features. Every feature function receives a time boundary (event timestamp) and uses only data strictly before event_ts.

Feature importance assessed via MDI, MDA, SFI — all within CPCV framework (see validation.md).

Feature staleness monitored: if value unchanged in 5+ consecutive events for a symbol, flagged for review.

Assembled features written to: silver.training_datasets with dataset_version column for reproducibility.

Primary Model (AFML Ch. 6)
Predicts direction of next significant price move (triple-barrier label) given features at event time.

Candidates (Phase 1):

| Model | Library | Rationale |
|-------|---------|-----------|
| Elastic Net | scikit-learn | Baseline, interpretable, regularized |
| Random Forest | scikit-learn | Non-linear, handles feature interactions, AFML-recommended |
| Bagged Decision Trees | scikit-learn | Reduces variance, AFML Ch. 6 |

Training rules:

All training uses single-node job clusters (no Spark ML).

Hyperparameter tuning uses randomized search within CPCV — not grid search on single split.

Target: triple-barrier label (+1, -1, 0).

Sample weights: average uniqueness per label (AFML Ch. 4) for overlapping labels.

Class imbalance: sample weights, not oversampling/undersampling.

Meta-Labeling (AFML Ch. 3)
Second-stage model deciding whether to act on primary model's prediction. Binary: 1 = take, 0 = pass.

Specification:

Input: Primary prediction (side) + original features + confidence features (predicted probability, rank in recent history).

Target: Binary — did primary model's side result in profit?

Model: GradientBoostingClassifier or RandomForest (scikit-learn).

Output: Probability of primary model being correct. Used for trade filtering AND bet sizing.

Thresholds for taking a trade:

| Condition | Threshold |
|-----------|-----------|
| Meta-label probability | > 0.50 |
| Primary model confidence | > 0.55 |
| Both conditions met | Required |

Bet Sizing (AFML Ch. 10)
Pipeline:

Raw bet size: size = 2 x Phi(z_score) - 1 where z_score = (probability - 0.5) / sigma_probability. sigma estimated from meta-label OOS calibration curve.

Active-bet averaging: Multiple concurrent signals for same symbol -> average weighted by confidence.

Budget constraint: Total portfolio exposure <= 1.0 (no leverage).

Discretization:

| Tier | Weight Range | Discretized Position |
|------|-------------|---------------------|
| No position | | 0.00 |
| Small | | 0.15 |
| Medium | | 0.35 |
| Large | | 0.60 |
| Conviction | [0.75, 1.00] | 0.85 |

Position count limit: Max 2 concurrent positions (BTC + ETH only). Rank by confidence.

Single-asset concentration: No single position > 70% of portfolio (2 assets, not 30 stocks).

No T+2 settlement check — crypto settles instantly.
