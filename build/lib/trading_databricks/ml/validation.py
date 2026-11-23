import polars as pl


def validate_bars(bars: pl.DataFrame) -> None:
    """Assert bars meet CUSUM/feature pipeline contracts."""
    if bars.is_empty():
        return

    required = {"bar_index", "close", "bar_end_time", "symbol"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"Missing required columns in bars: {missing}")  # noqa: E501

    if bars["bar_index"].is_null().any():
        bad = bars.filter(pl.col("bar_index").is_null()).head(5)
        raise ValueError(f"bar_index contains nulls. Offending rows:\n{bad}")

    if bars["symbol"].is_null().any():
        bad = bars.filter(pl.col("symbol").is_null()).head(5)
        raise ValueError(f"symbol contains nulls. Offending rows:\n{bad}")

    bad_close = bars.filter(pl.col("close") <= 0)
    if not bad_close.is_empty():
        raise ValueError(f"close prices must be > 0. Offending rows:\n{bad_close.head(5)}")  # noqa: E501

    if bars["bar_end_time"].is_null().any():
        bad = bars.filter(pl.col("bar_end_time").is_null()).head(5)
        raise ValueError(f"bar_end_time contains nulls. Offending rows:\n{bad}")  # noqa: E501

    if len(bars["bar_index"]) > 1 and bars["bar_index"].diff().drop_nulls().min() <= 0:
        raise ValueError("bar_index must be strictly monotonically increasing without duplicates.")  # noqa: E501


def validate_events(events: pl.DataFrame) -> None:
    """Assert events meet labeling/feature pipeline contracts."""
    if events.is_empty():
        return

    required = {
        "event_id", "symbol", "event_ts", "bar_index", "log_return",
        "cusum_side", "cusum_value", "threshold_used", "ewma_vol",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Missing required columns in events: {missing}")  # noqa: E501

    if events["event_id"].is_null().any():
        bad = events.filter(pl.col("event_id").is_null()).head(5)
        raise ValueError(f"event_id contains nulls. Offending rows:\n{bad}")

    if events["bar_index"].is_null().any():
        bad = events.filter(pl.col("bar_index").is_null()).head(5)
        raise ValueError(f"bar_index contains nulls. Offending rows:\n{bad}")

    if events["symbol"].is_null().any():
        bad = events.filter(pl.col("symbol").is_null()).head(5)
        raise ValueError(f"symbol contains nulls. Offending rows:\n{bad}")

    bad_vol = events.filter(pl.col("ewma_vol") <= 0)
    if not bad_vol.is_empty():
        raise ValueError(f"ewma_vol must be > 0. Offending rows:\n{bad_vol.head(5)}")  # noqa: E501

    invalid_sides = events.filter(~pl.col("cusum_side").is_in(["up", "down"]))
    if not invalid_sides.is_empty():
        raise ValueError(f"cusum_side must be 'up' or 'down'. Offending rows:\n{invalid_sides.head(5)}")  # noqa: E501

    bad_threshold = events.filter(
        pl.col("threshold_used").is_null() | (pl.col("threshold_used") <= 0)
    )
    if not bad_threshold.is_empty():
        raise ValueError(f"threshold_used must be > 0 and non-null. Offending rows:\n{bad_threshold.head(5)}")  # noqa: E501

    if events["event_ts"].is_null().any():
        bad = events.filter(pl.col("event_ts").is_null()).head(5)
        raise ValueError(f"event_ts contains nulls. Offending rows:\n{bad}")  # noqa: E501


def validate_labels(labels: pl.DataFrame) -> None:
    """Assert labels meet training/feature pipeline contracts."""
    if labels.is_empty():
        return

    required = {
        "event_id", "symbol", "event_ts", "label", "barrier_hit",
        "return_pct", "upper_level", "lower_level", "vertical_bar",
        "matured_at", "avg_uniqueness",
    }
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Missing required columns in labels: {missing}")  # noqa: E501

    if labels["event_id"].is_null().any():
        bad = labels.filter(pl.col("event_id").is_null()).head(5)
        raise ValueError(f"event_id contains nulls. Offending rows:\n{bad}")

    if labels["symbol"].is_null().any():
        bad = labels.filter(pl.col("symbol").is_null()).head(5)
        raise ValueError(f"symbol contains nulls. Offending rows:\n{bad}")

    invalid_labels = labels.filter(~pl.col("label").is_in([-1, 0, 1]))
    if not invalid_labels.is_empty():
        raise ValueError(f"label must be -1, 0, or 1. Offending rows:\n{invalid_labels.head(5)}")  # noqa: E501

    invalid_barriers = labels.filter(
        ~pl.col("barrier_hit").is_in(["upper", "lower", "vertical"])
    )
    if not invalid_barriers.is_empty():
        raise ValueError(f"barrier_hit must be 'upper', 'lower', or 'vertical'. Offending rows:\n{invalid_barriers.head(5)}")  # noqa: E501

    bad_uniq = labels.filter(
        pl.col("avg_uniqueness").is_null()
        | pl.col("avg_uniqueness").is_nan()
        | (pl.col("avg_uniqueness") <= 0)
        | (pl.col("avg_uniqueness") > 1.0)
    )
    if not bad_uniq.is_empty():
        raise ValueError(f"avg_uniqueness must be > 0 and <= 1, and non-null. Offending rows:\n{bad_uniq.head(5)}")  # noqa: E501


def validate_training_dataset(df: pl.DataFrame, *, n_groups: int = 6) -> None:
    """Assert training dataset meets ML promotion/training contracts."""
    if df.is_empty():
        raise ValueError("Training dataset DataFrame is empty.")  # noqa: E501

    if "label" not in df.columns:
        raise ValueError("Missing required column: label")  # noqa: E501

    if df["label"].is_null().any():
        bad = df.filter(pl.col("label").is_null()).head(5)
        raise ValueError(f"label column contains nulls. Offending rows:\n{bad}")  # noqa: E501

    if "cpcv_group" not in df.columns:
        raise ValueError("Missing required column: cpcv_group")  # noqa: E501

    bad_groups = df.filter(
        pl.col("cpcv_group").is_null()
        | (pl.col("cpcv_group") < 1)
        | (pl.col("cpcv_group") > n_groups)
    )
    if not bad_groups.is_empty():
        raise ValueError(f"cpcv_group must be between 1 and n_groups ({n_groups}). Offending rows:\n{bad_groups.head(5)}")  # noqa: E501

    for col in df.columns:
        if df[col].is_null().all():
            raise ValueError(f"Column '{col}' is all nulls, indicating a failed upstream join or feature logic.")  # noqa: E501
