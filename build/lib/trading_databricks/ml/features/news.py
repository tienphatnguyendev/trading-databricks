from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from trading_databricks.ml.config import FeatureConfig

_L1_CATEGORIES = [
    "Sentiment",
    "DeFi",
    "Supply",
    "Exchange",
    "Macro",
    "Adoption",
    "Security",
    "Regulation",
]

_NEWS_SCHEMA: dict[str, pl.PolarsDataType] = {
    "event_id": pl.Utf8,
    "symbol": pl.Utf8,
    "event_ts": pl.Datetime,
    "computed_at": pl.Datetime,
    **{f"news_count_{l1}": pl.Int32 for l1 in _L1_CATEGORIES},
    **{f"news_max_conf_{l1}": pl.Float64 for l1 in _L1_CATEGORIES},
    **{f"news_direction_{l1}": pl.Int32 for l1 in _L1_CATEGORIES},
    **{f"news_accepted_ratio_{l1}": pl.Float64 for l1 in _L1_CATEGORIES},
    "news_volume_24h": pl.Int32,
    "news_sentiment_score": pl.Float64,
    "has_news_24h": pl.Boolean,
}


def _build_zero_frame(events: pl.LazyFrame, now: datetime) -> pl.LazyFrame:
    """Zero-fill all news columns for every event."""
    zero_exprs = [pl.lit(now).alias("computed_at")]
    for l1 in _L1_CATEGORIES:
        zero_exprs += [
            pl.lit(0).cast(pl.Int32).alias(f"news_count_{l1}"),
            pl.lit(0.0).alias(f"news_max_conf_{l1}"),
            pl.lit(0).cast(pl.Int32).alias(f"news_direction_{l1}"),
            pl.lit(0.0).alias(f"news_accepted_ratio_{l1}"),
        ]
    zero_exprs += [
        pl.lit(0).cast(pl.Int32).alias("news_volume_24h"),
        pl.lit(0.0).alias("news_sentiment_score"),
        pl.lit(False).alias("has_news_24h"),
    ]
    return events.select(["event_id", "symbol", "event_ts"]).with_columns(zero_exprs)


def compute_news(
    events: pl.LazyFrame,
    news_features: pl.LazyFrame | None,
    *,
    config: FeatureConfig | None = None,
) -> pl.LazyFrame:
    """Aggregate news features for the 24h window before each event.

    When news_features is None or empty, returns zero-filled LazyFrame.
    """
    if config is None:
        config = FeatureConfig()
    now = datetime.now(UTC)

    if news_features is None:
        return _build_zero_frame(events, now)

    # Collect events and news for cross-join aggregation
    events_df = events.collect()
    if events_df.is_empty():
        return pl.DataFrame(schema=_NEWS_SCHEMA).lazy()

    news_df = news_features.filter(pl.col("is_impactful")).collect()

    if news_df.is_empty():
        return _build_zero_frame(events, now)

    rows: list[dict] = []
    lookback_hours = config.news_lookback_hours

    for event in events_df.iter_rows(named=True):
        t_end = event["event_ts"]
        t_start = t_end - timedelta(hours=lookback_hours)

        window = news_df.filter(
            (pl.col("published_at") >= t_start) & (pl.col("published_at") < t_end)
        )

        row: dict = {
            "event_id": event["event_id"],
            "symbol": event["symbol"],
            "event_ts": t_end,
            "computed_at": now,
        }

        total_count = 0
        sentiment_score = 0.0

        for l1 in _L1_CATEGORIES:
            all_cat = window.filter(pl.col("l1_category") == l1)
            total_cat_cnt = len(all_cat)

            row[f"news_accepted_ratio_{l1}"] = (
                float((all_cat["validation_status"] == "accepted").mean())
                if total_cat_cnt > 0
                else 0.0
            )

            cat = all_cat.filter(pl.col("validation_status") == "accepted")
            cnt = len(cat)
            row[f"news_count_{l1}"] = cnt
            row[f"news_max_conf_{l1}"] = (
                float(cat["confidence"].max()) if cnt > 0 else 0.0
            )
            if cnt > 0:
                # mode of expected_direction
                direction_mode = (
                    cat.group_by("expected_direction")
                    .agg(pl.len().alias("cnt"))
                    .sort("cnt", descending=True)["expected_direction"][0]
                )
                row[f"news_direction_{l1}"] = int(direction_mode)
            else:
                row[f"news_direction_{l1}"] = 0

            total_count += cnt
            if cnt > 0:
                sentiment_score += float(
                    (cat["confidence"] * cat["expected_direction"].cast(pl.Float64)).sum()
                )

        row["news_volume_24h"] = total_count
        row["news_sentiment_score"] = sentiment_score
        row["has_news_24h"] = total_count > 0
        rows.append(row)

    return pl.DataFrame(rows, schema=_NEWS_SCHEMA).lazy()
