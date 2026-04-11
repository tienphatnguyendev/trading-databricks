import argparse
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

# Schema matches src/trading_databricks/news/schemas.py NEWS_RAW_SCHEMA
NEWS_RAW_SCHEMA = StructType(
    [
        StructField("story_id", StringType(), nullable=False),
        StructField("headline", StringType(), nullable=True),
        StructField("url", StringType(), nullable=True),
        StructField("published_at", StringType(), nullable=True),
        StructField("source", StringType(), nullable=True),
        StructField("source_feed", StringType(), nullable=True),
        StructField("body", StringType(), nullable=True),
        StructField("ast_description_raw", StringType(), nullable=True),
        StructField("provider", StringType(), nullable=True),
        StructField("scraped_at", StringType(), nullable=False),
        StructField("fetch_method", StringType(), nullable=True),
        StructField("batch_id", StringType(), nullable=False),
    ]
)

def main(catalog: str, bucket: str, limit: int):
    # Initialize SparkSession for serverless
    spark = SparkSession.builder.getOrCreate()
    
    # We read from a specific day partition to keep the read fast and cheap
    # Feel free to expand the glob pattern if you need more variance.
    source_path = f"s3://{bucket}/news/tradingview/year=2026/month=03/day=30/*.ndjson"
    target_table = f"`{catalog}`.bronze.news_agent_eval_dataset"

    print(f"Reading from: {source_path}")
    print(f"Writing to: {target_table}")

    # Read raw JSON
    raw_df = (
        spark.read.format("json")
        .schema(NEWS_RAW_SCHEMA)
        .load(source_path)
    )

    # Filter out empty bodies and limit the sample
    sample_df = (
        raw_df
        .filter(F.col("body").isNotNull())
        .filter(F.trim(F.col("body")) != "")
        .limit(limit)
    )

    count = sample_df.count()
    print(f"Saving {count} records...")

    # Write as Delta table, overwriting any existing dataset
    sample_df.write.format("delta").mode("overwrite").saveAsTable(target_table)

    print(f"Successfully created static eval dataset: {target_table}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create static agent eval dataset")
    parser.add_argument("--catalog", default="smart-trading-dev", help="Target UC catalog")
    parser.add_argument("--bucket", default="ai-trading-raw-ap-southeast-1", help="Source S3 bucket")
    parser.add_argument("--limit", type=int, default=500, help="Number of records to sample")
    
    args = parser.parse_args()
    main(args.catalog, args.bucket, args.limit)
.limit)
