"""Gold feature engineering: silver.news_cleaned → gold.news_features.

Planned features:
- Sentiment score (headline + body): NLP library TBD — brainstorm before implementing
- Named entity extraction (tickers, companies, people): NLP library TBD
- Topic/keyword classification flags

Also creates gold.news_latest view for gen-AI agent queries (last 24h).

Write mode: overwrite.

TODO: brainstorm NLP stack → design doc → implement run_gold_features()
"""
from __future__ import annotations
