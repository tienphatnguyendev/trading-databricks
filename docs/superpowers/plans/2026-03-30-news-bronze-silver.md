# News Bronze + Silver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `ingest_news.py` (Auto Loader → `bronze.news_raw`) and `clean_news.py` (dedup + cast + normalize + flatten + MERGE → `silver.news_cleaned`) with full unit test coverage.

**Architecture:** Bronze reads S3 NDJSON via Auto Loader with `availableNow=True` trigger, converts `provider` and `ast_description_raw` from STRUCT to JSON strings (to match the STRING bronze schema contract), then appends to `{catalog}.bronze.news_raw`. Silver applies pure DataFrame transforms chained via `.transform()`, then MERGEs into `{catalog}.silver.news_cleaned` on `story_id`.

**Tech Stack:** PySpark 3.5, Delta Lake 3.2, `chispa` for DataFrame assertions, `pytest` with session-scoped SparkSession, `argparse` for CLI entry points.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/news/bronze/ingest_news.py` | Replace stub | Path helpers, `ingest_news()` streaming function, `run_bronze_ingestion()` entry point |
| `tests/news/test_ingest_news.py` | Replace stub | Tests for path helpers and schema |
| `tests/fixtures/news_sample.ndjson` | Create | One-line NDJSON fixture for schema tests |
| `src/news/silver/clean_news.py` | Replace stub | Pure transforms + `clean_news()` chain + `write_silver()` + `run_silver_transform()` |
| `tests/news/test_clean_news.py` | Replace stub | Unit tests for every transform function |

---

## Task 1: Create test fixture and bronze path helpers

**Files:**
- Create: `tests/fixtures/news_sample.ndjson`
- Replace: `src/news/bronze/ingest_news.py`
- Replace: `tests/news/test_ingest_news.py`

The `provider` field in the actual NDJSON is a JSON object `{"id": "reuters", "name": "Reuters", "logo_id": "reuters"}`. Auto Loader with `inferColumnTypes=true` reads it as a STRUCT. We cast it back to a JSON string in the pipeline so the bronze Delta table schema matches the spec (all STRING).

- [ ] **Step 1: Create the fixture file**

```bash
mkdir -p tests/fixtures
```

Write `tests/fixtures/news_sample.ndjson` — one line, a minimal valid record:

```json
{"story_id": "tag:reuters.com,2026:newsml_TEST01", "headline": "  BTC  rises  sharply  ", "url": "https://example.com/1", "published_at": "2026-03-30T10:38:35+00:00", "source": "tradingview", "source_feed": "reuters", "body": "  Bitcoin  moved  higher  ", "ast_description_raw": {"type": "root", "children": []}, "provider": {"id": "reuters", "name": "Reuters", "logo_id": "reuters"}, "scraped_at": "2026-03-30T10:38:45.273285+00:00", "fetch_method": "api", "batch_id": "batch_20260330T103845Z_1d5fae52"}
```

- [ ] **Step 2: Write `ingest_news.py` with path helpers and required columns list**

```python
"""Bronze ingestion: S3 NDJSON → bronze.news_raw via Auto Loader.

Reads news NDJSON files from S3 written by news-scraper-aws.
Uses trigger(availableNow=True) — processes all new files then terminates.
Scheduled every 15 minutes by the DABs job schedule.

Source path: s3://{bucket}/news/tradingview/year=*/month=*/day=*/
Target table: {catalog}.bronze.news_raw
"""
from __future__ import annotations

import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "story_id",
        "headline",
        "url",
        "published_at",
        "source",
        "source_feed",
        "body",
        "ast_description_raw",
        "provider",
        "scraped_at",
        "fetch_method",
        "batch_id",
    }
)


def build_source_path(*, bucket: str) -> str:
    return f"s3://{bucket}/news/tradingview/year=*/month=*/day=*/"


def build_checkpoint_path(*, checkpoint_base: str) -> str:
    return f"{checkpoint_base}/news/bronze"


def ingest_news(
    *,
    spark: SparkSession,
    source_path: str,
    checkpoint_path: str,
    target_table: str,
) -> None:
    """Stream S3 NDJSON into bronze.news_raw.

    Casts provider and ast_description_raw to JSON strings so the bronze
    Delta table schema stays all-STRING as per the contract.
    """
    query = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaLocation", checkpoint_path)
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .load(source_path)
        .withColumn("provider", F.to_json(F.col("provider")))
        .withColumn("ast_description_raw", F.to_json(F.col("ast_description_raw")))
        .writeStream.option("checkpointLocation", checkpoint_path)
        .trigger(availableNow=True)
        .toTable(target_table)
    )
    query.awaitTermination()


def run_bronze_ingestion(*, catalog: str, bucket: str, checkpoint_base: str) -> None:
    spark = SparkSession.builder.getOrCreate()
    ingest_news(
        spark=spark,
        source_path=build_source_path(bucket=bucket),
        checkpoint_path=build_checkpoint_path(checkpoint_base=checkpoint_base),
        target_table=f"{catalog}.bronze.news_raw",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bronze news ingestion")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--checkpoint-base", required=True)
    args = parser.parse_args()
    run_bronze_ingestion(
        catalog=args.catalog,
        bucket=args.bucket,
        checkpoint_base=args.checkpoint_base,
    )
```

- [ ] **Step 3: Write the tests**

```python
"""Tests for bronze news ingestion."""
from __future__ import annotations

from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from src.news.bronze.ingest_news import (
    REQUIRED_COLUMNS,
    build_checkpoint_path,
    build_source_path,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "news_sample.ndjson"


def test_build_source_path() -> None:
    assert build_source_path(bucket="my-bucket") == (
        "s3://my-bucket/news/tradingview/year=*/month=*/day=*/"
    )


def test_build_checkpoint_path() -> None:
    assert build_checkpoint_path(checkpoint_base="s3://bucket/checkpoints") == (
        "s3://bucket/checkpoints/news/bronze"
    )


def test_bronze_schema_contains_required_columns(spark: SparkSession) -> None:
    df = spark.read.json(str(FIXTURE))
    assert REQUIRED_COLUMNS.issubset(set(df.columns))


def test_bronze_null_fields_allowed(spark: SparkSession) -> None:
    null_record = (
        '{"story_id": "x1", "headline": null, "url": null, "published_at": null,'
        ' "source": "tradingview", "source_feed": "test", "body": null,'
        ' "ast_description_raw": null, "provider": null, "scraped_at": null,'
        ' "fetch_method": "api", "batch_id": "batch_test"}'
    )
    df = spark.read.json(spark.sparkContext.parallelize([null_record]))
    assert df.count() == 1
    row = df.first()
    assert row["story_id"] == "x1"
    assert row["headline"] is None
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
uv run pytest tests/news/test_ingest_news.py -v
```

Expected output:
```
tests/news/test_ingest_news.py::test_build_source_path PASSED
tests/news/test_ingest_news.py::test_build_checkpoint_path PASSED
tests/news/test_ingest_news.py::test_bronze_schema_contains_required_columns PASSED
tests/news/test_ingest_news.py::test_bronze_null_fields_allowed PASSED
4 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/news/bronze/ingest_news.py tests/news/test_ingest_news.py tests/fixtures/news_sample.ndjson
git commit -m "feat: implement news bronze ingestion with path helpers and schema contract"
```

---

## Task 2: Silver — `deduplicate()`

**Files:**
- Replace: `src/news/silver/clean_news.py` (write the full file skeleton + this function)
- Replace: `tests/news/test_clean_news.py` (write the full file skeleton + this test)

- [ ] **Step 1: Write failing test**

```python
"""Tests for silver news transformation."""
from __future__ import annotations

import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F

from src.news.silver.clean_news import deduplicate


def test_deduplicate_keeps_latest_scraped_at(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        [
            Row(story_id="s1", scraped_at="2026-03-30T10:00:00+00:00", headline="old"),
            Row(story_id="s1", scraped_at="2026-03-30T11:00:00+00:00", headline="new"),
            Row(story_id="s2", scraped_at="2026-03-30T09:00:00+00:00", headline="only"),
        ]
    )
    result = deduplicate(df)
    assert result.count() == 2
    s1 = result.filter(F.col("story_id") == "s1").first()
    assert s1["headline"] == "new"
```

- [ ] **Step 2: Run — expect FAIL (ImportError)**

```bash
uv run pytest tests/news/test_clean_news.py::test_deduplicate_keeps_latest_scraped_at -v
```

Expected: `ImportError: cannot import name 'deduplicate'`

- [ ] **Step 3: Write `clean_news.py` with `deduplicate()`**

```python
"""Silver transformation: bronze.news_raw → silver.news_cleaned.

Transformations applied in order:
1. deduplicate       — keep latest scraped_at per story_id
2. cast_timestamps   — published_at, scraped_at → TIMESTAMP
3. normalize_text    — collapse/trim whitespace in headline, body
4. flatten_provider  — provider JSON string → provider_id, provider_name, provider_logo_id
5. add_ingested_at   — _ingested_at TIMESTAMP metadata column

Write mode: MERGE on story_id (Delta upsert).
"""
from __future__ import annotations

import argparse

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType
from pyspark.sql.window import Window


def deduplicate(df: DataFrame) -> DataFrame:
    """Keep the row with the latest scraped_at for each story_id."""
    window = Window.partitionBy("story_id").orderBy(F.col("scraped_at").desc())
    return (
        df.withColumn("_row_num", F.row_number().over(window))
        .filter(F.col("_row_num") == 1)
        .drop("_row_num")
    )
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/news/test_clean_news.py::test_deduplicate_keeps_latest_scraped_at -v
```

- [ ] **Step 5: Commit**

```bash
git add src/news/silver/clean_news.py tests/news/test_clean_news.py
git commit -m "feat: silver deduplicate — keep latest scraped_at per story_id"
```

---

## Task 3: Silver — `cast_timestamps()`

**Files:**
- Modify: `src/news/silver/clean_news.py`
- Modify: `tests/news/test_clean_news.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/news/test_clean_news.py`:

```python
from src.news.silver.clean_news import cast_timestamps


def test_cast_timestamps_converts_published_at(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        [Row(published_at="2026-03-30T10:38:35+00:00", scraped_at="2026-03-30T10:38:45.273285+00:00")]
    )
    result = cast_timestamps(df)
    from pyspark.sql.types import TimestampType
    assert dict(result.dtypes)["published_at"] == "timestamp"
    assert dict(result.dtypes)["scraped_at"] == "timestamp"
    row = result.first()
    assert row["published_at"] is not None
    assert row["scraped_at"] is not None


def test_cast_timestamps_null_published_at_stays_null(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        [Row(published_at=None, scraped_at=None)]
    )
    result = cast_timestamps(df)
    row = result.first()
    assert row["published_at"] is None
    assert row["scraped_at"] is None
```

- [ ] **Step 2: Run — expect FAIL (ImportError)**

```bash
uv run pytest tests/news/test_clean_news.py::test_cast_timestamps_converts_published_at -v
```

- [ ] **Step 3: Add `cast_timestamps()` to `clean_news.py`**

```python
def cast_timestamps(df: DataFrame) -> DataFrame:
    """Cast published_at and scraped_at STRING columns to TIMESTAMP.

    Input format: ISO 8601 with timezone offset (e.g. 2026-03-30T10:38:35+00:00).
    Microseconds in scraped_at are truncated to millisecond precision on cast.
    NULL inputs stay NULL.
    """
    return df.withColumn(
        "published_at", F.col("published_at").cast("timestamp")
    ).withColumn("scraped_at", F.col("scraped_at").cast("timestamp"))
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/news/test_clean_news.py::test_cast_timestamps_converts_published_at tests/news/test_clean_news.py::test_cast_timestamps_null_published_at_stays_null -v
```

- [ ] **Step 5: Commit**

```bash
git add src/news/silver/clean_news.py tests/news/test_clean_news.py
git commit -m "feat: silver cast_timestamps — published_at and scraped_at to TIMESTAMP"
```

---

## Task 4: Silver — `normalize_text()`

**Files:**
- Modify: `src/news/silver/clean_news.py`
- Modify: `tests/news/test_clean_news.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/news/test_clean_news.py`:

```python
from src.news.silver.clean_news import normalize_text


def test_normalize_text_trims_whitespace(spark: SparkSession) -> None:
    df = spark.createDataFrame([Row(headline="  BTC rises  ", body="  Bitcoin moved  ")])
    result = normalize_text(df)
    row = result.first()
    assert row["headline"] == "BTC rises"
    assert row["body"] == "Bitcoin moved"


def test_normalize_text_collapses_internal_whitespace(spark: SparkSession) -> None:
    df = spark.createDataFrame([Row(headline="BTC  rises  sharply", body="Bitcoin  moved   higher")])
    result = normalize_text(df)
    row = result.first()
    assert row["headline"] == "BTC rises sharply"
    assert row["body"] == "Bitcoin moved higher"


def test_normalize_text_null_body_stays_null(spark: SparkSession) -> None:
    df = spark.createDataFrame([Row(headline="BTC rises", body=None)])
    result = normalize_text(df)
    row = result.first()
    assert row["headline"] == "BTC rises"
    assert row["body"] is None
```

- [ ] **Step 2: Run — expect FAIL (ImportError)**

```bash
uv run pytest tests/news/test_clean_news.py::test_normalize_text_trims_whitespace -v
```

- [ ] **Step 3: Add `normalize_text()` to `clean_news.py`**

```python
def normalize_text(df: DataFrame) -> DataFrame:
    """Trim leading/trailing whitespace and collapse internal whitespace runs.

    Applied to headline and body columns. NULL values are passed through unchanged.
    """

    def _normalize(col: str) -> F.Column:
        return F.regexp_replace(F.trim(F.col(col)), r"\s+", " ")

    return df.withColumn("headline", _normalize("headline")).withColumn(
        "body", _normalize("body")
    )
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/news/test_clean_news.py::test_normalize_text_trims_whitespace tests/news/test_clean_news.py::test_normalize_text_collapses_internal_whitespace tests/news/test_clean_news.py::test_normalize_text_null_body_stays_null -v
```

- [ ] **Step 5: Commit**

```bash
git add src/news/silver/clean_news.py tests/news/test_clean_news.py
git commit -m "feat: silver normalize_text — trim and collapse whitespace in headline and body"
```

---

## Task 5: Silver — `flatten_provider()`

**Files:**
- Modify: `src/news/silver/clean_news.py`
- Modify: `tests/news/test_clean_news.py`

The `provider` column in bronze is a JSON string: `'{"id": "reuters", "name": "Reuters", "logo_id": "reuters"}'`. This function extracts `provider_id`, `provider_name`, `provider_logo_id` and drops the raw JSON string.

- [ ] **Step 1: Write failing test**

Add to `tests/news/test_clean_news.py`:

```python
from src.news.silver.clean_news import flatten_provider


def test_flatten_provider_extracts_fields(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        [Row(provider='{"id": "reuters", "name": "Reuters", "logo_id": "reuters"}')]
    )
    result = flatten_provider(df)
    assert "provider" not in result.columns
    assert "provider_id" in result.columns
    assert "provider_name" in result.columns
    assert "provider_logo_id" in result.columns
    row = result.first()
    assert row["provider_id"] == "reuters"
    assert row["provider_name"] == "Reuters"
    assert row["provider_logo_id"] == "reuters"


def test_flatten_provider_null_is_safe(spark: SparkSession) -> None:
    df = spark.createDataFrame([Row(provider=None)])
    result = flatten_provider(df)
    row = result.first()
    assert row["provider_id"] is None
    assert row["provider_name"] is None
    assert row["provider_logo_id"] is None
```

- [ ] **Step 2: Run — expect FAIL (ImportError)**

```bash
uv run pytest tests/news/test_clean_news.py::test_flatten_provider_extracts_fields -v
```

- [ ] **Step 3: Add `flatten_provider()` to `clean_news.py`**

```python
_PROVIDER_SCHEMA = StructType(
    [
        StructField("id", StringType()),
        StructField("name", StringType()),
        StructField("logo_id", StringType()),
    ]
)


def flatten_provider(df: DataFrame) -> DataFrame:
    """Parse provider JSON string into typed columns, then drop the raw string.

    Input:  provider STRING  e.g. '{"id":"reuters","name":"Reuters","logo_id":"reuters"}'
    Output: provider_id STRING, provider_name STRING, provider_logo_id STRING
    NULL provider produces NULL in all three output columns.
    """
    parsed = F.from_json(F.col("provider"), _PROVIDER_SCHEMA)
    return (
        df.withColumn("provider_id", parsed.getField("id"))
        .withColumn("provider_name", parsed.getField("name"))
        .withColumn("provider_logo_id", parsed.getField("logo_id"))
        .drop("provider")
    )
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/news/test_clean_news.py::test_flatten_provider_extracts_fields tests/news/test_clean_news.py::test_flatten_provider_null_is_safe -v
```

- [ ] **Step 5: Commit**

```bash
git add src/news/silver/clean_news.py tests/news/test_clean_news.py
git commit -m "feat: silver flatten_provider — extract provider_id, provider_name, provider_logo_id"
```

---

## Task 6: Silver — `add_ingested_at()`

**Files:**
- Modify: `src/news/silver/clean_news.py`
- Modify: `tests/news/test_clean_news.py`

- [ ] **Step 1: Write failing test**

Add to `tests/news/test_clean_news.py`:

```python
from src.news.silver.clean_news import add_ingested_at


def test_add_ingested_at_adds_column(spark: SparkSession) -> None:
    df = spark.createDataFrame([Row(story_id="s1")])
    result = add_ingested_at(df)
    assert "_ingested_at" in result.columns
    from pyspark.sql.types import TimestampType
    assert dict(result.dtypes)["_ingested_at"] == "timestamp"
    assert result.first()["_ingested_at"] is not None
```

- [ ] **Step 2: Run — expect FAIL (ImportError)**

```bash
uv run pytest tests/news/test_clean_news.py::test_add_ingested_at_adds_column -v
```

- [ ] **Step 3: Add `add_ingested_at()` to `clean_news.py`**

```python
def add_ingested_at(df: DataFrame) -> DataFrame:
    """Add _ingested_at TIMESTAMP column set to the current time."""
    return df.withColumn("_ingested_at", F.current_timestamp())
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/news/test_clean_news.py::test_add_ingested_at_adds_column -v
```

- [ ] **Step 5: Commit**

```bash
git add src/news/silver/clean_news.py tests/news/test_clean_news.py
git commit -m "feat: silver add_ingested_at — append _ingested_at metadata column"
```

---

## Task 7: Silver — `clean_news()` pipeline + full integration test

**Files:**
- Modify: `src/news/silver/clean_news.py`
- Modify: `tests/news/test_clean_news.py`

- [ ] **Step 1: Write failing test**

Add to `tests/news/test_clean_news.py`:

```python
from src.news.silver.clean_news import clean_news


def test_clean_news_full_pipeline(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        [
            Row(
                story_id="s1",
                headline="  BTC  rises  ",
                url="https://example.com/1",
                published_at="2026-03-30T10:00:00+00:00",
                source="tradingview",
                source_feed="reuters",
                body="  Bitcoin  moved  ",
                ast_description_raw='{"type":"root","children":[]}',
                provider='{"id":"reuters","name":"Reuters","logo_id":"reuters"}',
                scraped_at="2026-03-30T10:01:00+00:00",
                fetch_method="api",
                batch_id="batch_test",
            ),
            # duplicate: older scraped_at — should be dropped
            Row(
                story_id="s1",
                headline="  old  headline  ",
                url="https://example.com/1",
                published_at="2026-03-30T09:00:00+00:00",
                source="tradingview",
                source_feed="reuters",
                body="  old  body  ",
                ast_description_raw='{"type":"root","children":[]}',
                provider='{"id":"reuters","name":"Reuters","logo_id":"reuters"}',
                scraped_at="2026-03-30T09:01:00+00:00",
                fetch_method="api",
                batch_id="batch_old",
            ),
        ]
    )
    result = clean_news(df)

    assert result.count() == 1
    row = result.first()

    # dedup — kept the newer one
    assert row["headline"] == "BTC rises"
    assert row["body"] == "Bitcoin moved"

    # timestamps cast
    from pyspark.sql.types import TimestampType
    assert dict(result.dtypes)["published_at"] == "timestamp"
    assert dict(result.dtypes)["scraped_at"] == "timestamp"

    # provider flattened
    assert "provider" not in result.columns
    assert row["provider_id"] == "reuters"
    assert row["provider_name"] == "Reuters"

    # metadata added
    assert "_ingested_at" in result.columns
    assert row["_ingested_at"] is not None
```

- [ ] **Step 2: Run — expect FAIL (ImportError)**

```bash
uv run pytest tests/news/test_clean_news.py::test_clean_news_full_pipeline -v
```

- [ ] **Step 3: Add `clean_news()` to `clean_news.py`**

```python
def clean_news(df: DataFrame) -> DataFrame:
    """Apply the full silver transformation pipeline in order."""
    return (
        df.transform(deduplicate)
        .transform(cast_timestamps)
        .transform(normalize_text)
        .transform(flatten_provider)
        .transform(add_ingested_at)
    )
```

- [ ] **Step 4: Run all silver tests — expect all PASS**

```bash
uv run pytest tests/news/test_clean_news.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/news/silver/clean_news.py tests/news/test_clean_news.py
git commit -m "feat: silver clean_news pipeline — chain all transforms via DataFrame.transform()"
```

---

## Task 8: Silver — `write_silver()` + `run_silver_transform()` entry point

**Files:**
- Modify: `src/news/silver/clean_news.py`

`write_silver()` is not unit tested (it writes to Delta, which requires a running cluster). It is integration tested via `databricks bundle run`. `run_silver_transform()` is the CLI entry point.

- [ ] **Step 1: Add `write_silver()` and `run_silver_transform()` to `clean_news.py`**

```python
def write_silver(df: DataFrame, *, spark: SparkSession, target_table: str) -> None:
    """MERGE cleaned DataFrame into silver.news_cleaned on story_id.

    Creates the table on first run if it does not exist.
    """
    if spark.catalog.tableExists(target_table):
        (
            DeltaTable.forName(spark, target_table)
            .alias("target")
            .merge(df.alias("source"), "target.story_id = source.story_id")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        df.write.format("delta").saveAsTable(target_table)


def run_silver_transform(*, catalog: str) -> None:
    spark = SparkSession.builder.getOrCreate()
    bronze_df = spark.table(f"{catalog}.bronze.news_raw")
    cleaned_df = clean_news(bronze_df)
    write_silver(cleaned_df, spark=spark, target_table=f"{catalog}.silver.news_cleaned")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Silver news transformation")
    parser.add_argument("--catalog", required=True)
    args = parser.parse_args()
    run_silver_transform(catalog=args.catalog)
```

- [ ] **Step 2: Run full test suite — all must pass**

```bash
uv run pytest -v
```

Expected: all previously written tests pass, 0 failures.

- [ ] **Step 3: Commit**

```bash
git add src/news/silver/clean_news.py
git commit -m "feat: silver write_silver and run_silver_transform — MERGE into silver.news_cleaned"
```

---

## Task 9: Bundle validate + deploy to dev

- [ ] **Step 1: Validate bundle**

```bash
databricks bundle validate
```

Expected: `Validation OK!`

- [ ] **Step 2: Deploy to dev**

```bash
databricks bundle deploy -t dev
```

Expected: jobs `news-pipeline-dev` and `tick-pipeline-dev` appear in the workspace.

- [ ] **Step 3: Run bronze task to smoke-test the pipeline**

```bash
databricks bundle run news_pipeline -t dev --task bronze_ingest
```

Expected: task completes, `trading-signal-gen-dev.bronze.news_raw` Delta table created with rows.

- [ ] **Step 4: Verify table via Databricks SQL**

```sql
SELECT COUNT(*), MIN(scraped_at), MAX(scraped_at)
FROM `trading-signal-gen-dev`.bronze.news_raw
LIMIT 1;
```

Expected: row count > 0, timestamps in the 2026-03-30 range.

- [ ] **Step 5: Run silver task**

```bash
databricks bundle run news_pipeline -t dev --task silver_clean
```

Expected: task completes, `trading-signal-gen-dev.silver.news_cleaned` Delta table created.

- [ ] **Step 6: Verify silver table**

```sql
SELECT COUNT(*), MIN(published_at), MAX(published_at), COUNT(DISTINCT provider_id)
FROM `trading-signal-gen-dev`.silver.news_cleaned
LIMIT 1;
```

Expected: row count > 0, `published_at` as timestamp (not string), `provider_id` populated.

- [ ] **Step 7: Final commit with bundle state**

```bash
git add -p   # review any changed files
git commit -m "chore: deploy news bronze+silver to dev — smoke test passed"
```

---

## Self-Review

**Spec coverage:**
- Bronze: Auto Loader, `availableNow=True` trigger, 15 min schedule (in job YAML), schema contract, append-only ✓
- Silver: dedup by story_id ✓, cast timestamps ✓, normalize whitespace ✓, flatten provider ✓, `_ingested_at` metadata ✓, MERGE on story_id ✓
- Gold: out of scope (NLP brainstorm required first) ✓
- Tests: all planned test cases from the stubs are covered ✓

**Placeholder scan:** None found.

**Type consistency:**
- All functions take `DataFrame` and return `DataFrame` ✓
- `run_*` functions take keyword-only str args ✓
- `write_silver` takes `df, *, spark, target_table` ✓
- `clean_news` chains `.transform()` calls ✓
