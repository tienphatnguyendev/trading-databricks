import time
from datetime import datetime, timedelta, UTC
import polars as pl
import numpy as np

# Mocking the constants and types from the original file
_L1_CATEGORIES = [
    "Sentiment", "DeFi", "Supply", "Exchange",
    "Macro", "Adoption", "Security", "Regulation",
]

_NEWS_SCHEMA = {
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

class MockConfig:
    def __init__(self, news_lookback_hours=24):
        self.news_lookback_hours = news_lookback_hours

def _build_zero_frame(events: pl.LazyFrame, now: datetime) -> pl.LazyFrame:
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

def original_compute_news(
    events: pl.LazyFrame,
    news_features: pl.LazyFrame | None,
    config=None,
) -> pl.LazyFrame:
    if config is None: config = MockConfig()
    now = datetime.now(UTC)
    if news_features is None:
        return _build_zero_frame(events, now)

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

def optimized_compute_news(
    events: pl.LazyFrame,
    news_features: pl.LazyFrame | None,
    config=None,
) -> pl.LazyFrame:
    if config is None: config = MockConfig()
    now = datetime.now(UTC)

    if news_features is None:
        return _build_zero_frame(events, now)

    impactful_news = news_features.filter(pl.col("is_impactful"))
    lookback_hours = config.news_lookback_hours

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
            .fill_nan(0.0)
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

    group_keys = ["event_id", "symbol", "event_ts"]
    news_aggregated = joined.group_by(group_keys).agg(aggs)

    result = events.join(news_aggregated, on=group_keys, how="left")

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

def generate_data(n_events=100, n_news=1000):
    base_ts = datetime(2025, 1, 1, tzinfo=UTC)
    events = pl.DataFrame({
        "event_id": [f"e_{i}" for i in range(n_events)],
        "symbol": ["BTCUSDT"] * n_events,
        "event_ts": [base_ts + timedelta(hours=i) for i in range(n_events)],
        "bar_index": list(range(n_events))
    }).lazy()

    news = pl.DataFrame({
        "published_at": [base_ts + timedelta(minutes=i) for i in range(n_news)],
        "l1_category": [np.random.choice(_L1_CATEGORIES) for _ in range(n_news)],
        "confidence": np.random.rand(n_news),
        "expected_direction": np.random.choice([-1, 0, 1], n_news),
        "validation_status": np.random.choice(["accepted", "rejected", "pending"], n_news),
        "is_impactful": [True] * n_news
    }).lazy()

    return events, news

if __name__ == "__main__":
    try:
        events, news = generate_data(n_events=200, n_news=2000)

        print("Starting benchmark for original_compute_news...")
        start = time.perf_counter()
        res_orig = original_compute_news(events, news).collect()
        end = time.perf_counter()
        orig_time = end - start
        print(f"Original took: {orig_time:.4f}s")

        print("Starting benchmark for optimized_compute_news...")
        start = time.perf_counter()
        res_opt = optimized_compute_news(events, news).collect()
        end = time.perf_counter()
        opt_time = end - start
        print(f"Optimized took: {opt_time:.4f}s")

        print(f"Improvement: {orig_time / opt_time:.1f}x faster")

        # Simple verification
        # Sort because group_by might change order
        res_orig = res_orig.sort("event_id").drop("computed_at")
        res_opt = res_opt.sort("event_id").drop("computed_at")

        pd_orig = res_orig.to_pandas()
        pd_opt = res_opt.to_pandas()

        # Comparing with some tolerance for floats if needed, but here it should be exact
        pd.testing.assert_frame_equal(pd_orig, pd_opt, check_dtype=False)
        print("Verification successful: Optimized result matches original!")

    except Exception as e:
        print(f"Could not run benchmark: {e}")
        import traceback
        traceback.print_exc()
