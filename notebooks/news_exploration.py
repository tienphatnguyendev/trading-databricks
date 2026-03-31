# Databricks notebook source
# MAGIC %md
# MAGIC # News Data Exploration
# MAGIC
# MAGIC Exploratory analysis of financial news NDJSON files landed in S3 by `news-scraper-aws`.
# MAGIC
# MAGIC **S3 source:** `s3://ai-trading-raw-ap-southeast-1/news/tradingview/year=*/month=*/day=*/`
# MAGIC
# MAGIC **Unity Catalog external location:** `db_s3_external_databricks-s3-ingest-5dfcc-ai-trading-raw`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

BUCKET = "ai-trading-raw-ap-southeast-1"
SOURCE_PATH = f"s3://{BUCKET}/news/tradingview/year=*/month=*/day=*/"

print(f"Source: {SOURCE_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Read Raw NDJSON (Auto Loader — batch mode)
# MAGIC
# MAGIC Using `cloudFiles` in batch mode (`trigger(availableNow=True)` is for the scheduled job;
# MAGIC here we use `spark.read` for interactive exploration).

# COMMAND ----------

raw_df = (
    spark.read.format("json")
    .option("inferSchema", "true")
    .load(SOURCE_PATH)
)

print(f"Total records: {raw_df.count():,}")
print(f"Columns: {len(raw_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Schema

# COMMAND ----------

raw_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Sample Records

# COMMAND ----------

display(raw_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Data Quality — Nulls

# COMMAND ----------

key_cols = [
    "story_id", "headline", "url", "published_at", "scraped_at",
    "source", "source_feed", "body", "provider", "fetch_method", "batch_id",
]

null_counts = raw_df.select(
    *[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in key_cols if c in raw_df.columns]
)
display(null_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Duplicate `story_id` Check
# MAGIC
# MAGIC The scraper's SSM watermark eliminates ~97–99% of duplicates at source.
# MAGIC Silver dedup is a safety net — this cell shows how many duplicates exist in bronze.

# COMMAND ----------

dupe_check = (
    raw_df.groupBy("story_id")
    .agg(F.count("*").alias("n"))
    .filter(F.col("n") > 1)
    .orderBy(F.col("n").desc())
)

print(f"story_ids with duplicates: {dupe_check.count():,}")
display(dupe_check.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Data Volume Over Time
# MAGIC
# MAGIC Articles per day (UTC) — confirms scraper cadence.

# COMMAND ----------

volume_by_day = (
    raw_df.withColumn(
        "scraped_date",
        F.to_date(F.col("scraped_at").cast("timestamp"))
    )
    .groupBy("scraped_date")
    .agg(
        F.count("*").alias("article_count"),
        F.countDistinct("story_id").alias("unique_stories"),
        F.countDistinct("batch_id").alias("batches"),
    )
    .orderBy("scraped_date")
)

display(volume_by_day)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Source Feed Distribution

# COMMAND ----------

source_dist = (
    raw_df.groupBy("source_feed")
    .agg(
        F.count("*").alias("count"),
        F.countDistinct("story_id").alias("unique_stories"),
    )
    .orderBy(F.col("count").desc())
)

display(source_dist)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Provider Structure
# MAGIC
# MAGIC `provider` is read by Spark as a STRUCT.
# MAGIC In the bronze pipeline we cast it to a JSON string; in silver we flatten it back.

# COMMAND ----------

# Show unique providers
provider_dist = (
    raw_df.groupBy(
        F.col("provider.id").alias("provider_id"),
        F.col("provider.name").alias("provider_name"),
        F.col("provider.logo_id").alias("provider_logo_id"),
    )
    .agg(F.count("*").alias("count"))
    .orderBy(F.col("count").desc())
)

display(provider_dist)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Timestamp Format Inspection
# MAGIC
# MAGIC Confirm `published_at` and `scraped_at` parse cleanly to TIMESTAMP.
# MAGIC Silver will use `.cast("timestamp")`.

# COMMAND ----------

ts_check = raw_df.select(
    F.col("published_at"),
    F.col("scraped_at"),
    F.col("published_at").cast("timestamp").alias("published_at_ts"),
    F.col("scraped_at").cast("timestamp").alias("scraped_at_ts"),
).limit(20)

display(ts_check)

# Confirm no nulls introduced by cast
null_after_cast = raw_df.select(
    F.sum(
        F.col("published_at").cast("timestamp").isNull().cast("int")
    ).alias("published_at_cast_nulls"),
    F.sum(
        F.col("scraped_at").cast("timestamp").isNull().cast("int")
    ).alias("scraped_at_cast_nulls"),
    F.count("*").alias("total"),
)
display(null_after_cast)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. `ast_description_raw` Structure
# MAGIC
# MAGIC Rich AST of the article body. Retained as STRING in silver for gen-AI agent use.
# MAGIC This cell inspects the top-level `type` field distribution.

# COMMAND ----------

ast_types = (
    raw_df.select(
        F.col("ast_description_raw.type").alias("ast_root_type"),
        F.size(F.col("ast_description_raw.children")).alias("num_top_children"),
    )
    .groupBy("ast_root_type")
    .agg(
        F.count("*").alias("count"),
        F.avg("num_top_children").alias("avg_children"),
        F.max("num_top_children").alias("max_children"),
    )
    .orderBy(F.col("count").desc())
)

display(ast_types)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Batch ID Format Check
# MAGIC
# MAGIC Expected: `batch_{YYYYMMDDTHHMMSSZ}_{8-char-hex}`

# COMMAND ----------

import re

batch_format_check = raw_df.select(
    F.col("batch_id"),
    F.regexp_extract(
        F.col("batch_id"),
        r"^batch_(\d{8}T\d{6}Z)_([0-9a-f]{8})$",
        0,
    ).alias("matched"),
).withColumn(
    "format_ok", F.col("matched") != ""
)

summary = batch_format_check.groupBy("format_ok").count()
display(summary)

# Any malformed batch_ids?
display(batch_format_check.filter(~F.col("format_ok")).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Silver Preview — apply transforms manually
# MAGIC
# MAGIC Quick sanity check of the silver transformations before running the job.

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType

_PROVIDER_SCHEMA = StructType([
    StructField("id", StringType()),
    StructField("name", StringType()),
    StructField("logo_id", StringType()),
])

silver_preview = (
    raw_df
    # dedup: keep latest scraped_at per story_id
    .withColumn(
        "_rn",
        F.row_number().over(
            __import__("pyspark.sql.window", fromlist=["Window"])
            .Window.partitionBy("story_id")
            .orderBy(F.col("scraped_at").desc())
        ),
    )
    .filter(F.col("_rn") == 1)
    .drop("_rn")
    # cast timestamps
    .withColumn("published_at", F.col("published_at").cast("timestamp"))
    .withColumn("scraped_at", F.col("scraped_at").cast("timestamp"))
    # normalize whitespace
    .withColumn("headline", F.regexp_replace(F.trim("headline"), r"\s+", " "))
    .withColumn("body", F.regexp_replace(F.trim("body"), r"\s+", " "))
    # flatten provider
    .withColumn("provider_id", F.col("provider.id"))
    .withColumn("provider_name", F.col("provider.name"))
    .withColumn("provider_logo_id", F.col("provider.logo_id"))
    .drop("provider")
    # cast ast_description_raw to JSON string (bronze does this; keep consistent)
    .withColumn("ast_description_raw", F.to_json(F.col("ast_description_raw")))
    # metadata
    .withColumn("_ingested_at", F.current_timestamp())
)

print(f"Silver preview row count: {silver_preview.count():,}")
silver_preview.printSchema()
display(silver_preview.limit(5))
