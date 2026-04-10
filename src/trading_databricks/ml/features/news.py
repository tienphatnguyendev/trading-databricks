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

    # Use join_where for lazy range join and avoid Python loops
    # Select only required columns from news_features before joining
    impactful_news = news_features.filter(pl.col("is_impactful")).select(
        [
            "published_at",
            "l1_category",
            "validation_status",
            "confidence",
            "expected_direction",
        ]
    )
    lookback_hours = config.news_lookback_hours

    # Join news articles into the lookback window of each event:
    # [event_ts - lookback, event_ts)
    joined = events.join_where(
        impactful_news,
        pl.col("event_ts") > pl.col("published_at"),
        pl.col("event_ts") <= pl.col("published_at") + timedelta(hours=lookback_hours),
    )

    aggs = []
    for l1 in _L1_CATEGORIES:
        cat_mask = pl.col("l1_category") == l1
        accepted_mask = cat_mask & (pl.col("validation_status") == "accepted")

        aggs += [
            pl.col("validation_status")
            .filter(cat_mask)
            .eq("accepted")
            .mean()
            .fill_null(0.0)
            .alias(f"news_accepted_ratio_{l1}"),
            accepted_mask.sum().cast(pl.Int32).alias(f"news_count_{l1}"),
            pl.col("confidence").filter(accepted_mask).max().fill_null(0.0).alias(
                f"news_max_conf_{l1}"
            ),
            pl.col("expected_direction")
            .filter(accepted_mask)
            .mode()
            .first()
            .fill_null(0)
            .cast(pl.Int32)
            .alias(f"news_direction_{l1}"),
        ]

    total_accepted_mask = (pl.col("validation_status") == "accepted") & (
        pl.col("l1_category").is_in(_L1_CATEGORIES)
    )
    aggs += [
        total_accepted_mask.sum().cast(pl.Int32).alias("news_volume_24h"),
        (pl.col("confidence") * pl.col("expected_direction").cast(pl.Float64))
        .filter(total_accepted_mask)
        .sum()
        .fill_null(0.0)
        .alias("news_sentiment_score"),
    ]

    # Aggregate news features per event. Use symbol/event_ts as well to keep them.
    group_keys = ["event_id", "symbol", "event_ts"]
    news_aggregated = joined.group_by(group_keys).agg(aggs)

    # Join back to original events to ensure all events are present
    # (including those with no news)
    result = events.join(news_aggregated, on=group_keys, how="left")

    # Fill default values for events that had no impactful news in their window
    fill_values = {
        "news_volume_24h": 0,
        "news_sentiment_score": 0.0,
        **{f"news_count_{l1}": 0 for l1 in _L1_CATEGORIES},
        **{f"news_max_conf_{l1}": 0.0 for l1 in _L1_CATEGORIES},
        **{f"news_direction_{l1}": 0 for l1 in _L1_CATEGORIES},
        **{f"news_accepted_ratio_{l1}": 0.0 for l1 in _L1_CATEGORIES},
    }

    return (
        result.with_columns(
            [pl.col(col).fill_null(val) for col, val in fill_values.items()]
        )
        .with_columns(
            pl.lit(now).alias("computed_at"),
            (pl.col("news_volume_24h") > 0).alias("has_news_24h"),
        )
        .select(list(_NEWS_SCHEMA.keys()))
    )
