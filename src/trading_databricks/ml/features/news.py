from __future__ import annotations

from datetime import UTC, datetime

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



    lookback_hours = config.news_lookback_hours

    events_with_bounds = events.with_columns(
        (pl.col("event_ts") - pl.duration(hours=lookback_hours)).alias("t_start")
    )

    impactful_news = news_features.filter(pl.col("is_impactful"))

    joined = events_with_bounds.join_where(
        impactful_news,
        pl.col("published_at") >= pl.col("t_start"),
        pl.col("published_at") < pl.col("event_ts")
    )

    agg_exprs = []
    is_accepted = pl.col("validation_status") == "accepted"

    for l1 in _L1_CATEGORIES:
        is_cat = pl.col("l1_category") == l1
        is_cat_accepted = is_cat & is_accepted

        ratio = (
            is_accepted.filter(is_cat)
            .mean()
            .fill_null(0.0)
            .alias(f"news_accepted_ratio_{l1}")
        )
        count = (
            is_cat_accepted.sum()
            .fill_null(0)
            .cast(pl.Int32)
            .alias(f"news_count_{l1}")
        )
        max_conf = (
            pl.col("confidence")
            .filter(is_cat_accepted)
            .max()
            .fill_null(0.0)
            .alias(f"news_max_conf_{l1}")
        )

        direction = (
            pl.col("expected_direction")
            .filter(is_cat_accepted)
            .drop_nulls()
            .mode()
            .max()
            .fill_null(0)
            .cast(pl.Int32)
            .alias(f"news_direction_{l1}")
        )

        agg_exprs.extend([ratio, count, max_conf, direction])

    agg_exprs.extend([
        is_accepted.sum().fill_null(0).cast(pl.Int32).alias("news_volume_24h"),
        (pl.col("confidence") * pl.col("expected_direction").cast(pl.Float64))
        .filter(is_accepted)
        .sum()
        .fill_null(0.0)
        .alias("news_sentiment_score"),
        (is_accepted.sum() > 0).fill_null(False).alias("has_news_24h")
    ])

    agg = joined.group_by("event_id").agg(agg_exprs)

    result = events.join(agg, on="event_id", how="left")

    fill_exprs = []
    for l1 in _L1_CATEGORIES:
        fill_exprs.extend([
            pl.col(f"news_accepted_ratio_{l1}").fill_null(0.0),
            pl.col(f"news_count_{l1}").fill_null(0),
            pl.col(f"news_max_conf_{l1}").fill_null(0.0),
            pl.col(f"news_direction_{l1}").fill_null(0),
        ])
    fill_exprs.extend([
        pl.col("news_volume_24h").fill_null(0),
        pl.col("news_sentiment_score").fill_null(0.0),
        pl.col("has_news_24h").fill_null(False),
        pl.lit(now).alias("computed_at")
    ])

    final_exprs = []
    for col_name, dtype in _NEWS_SCHEMA.items():
        if dtype == pl.Datetime:
             final_exprs.append(pl.col(col_name).dt.replace_time_zone(None).cast(dtype))
        else:
             final_exprs.append(pl.col(col_name).cast(dtype))

    return result.with_columns(fill_exprs).select(final_exprs)
