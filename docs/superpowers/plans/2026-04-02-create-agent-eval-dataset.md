# Create Raw Bronze Agent Evaluation Dataset Implementation Plan

**Status:** ✅ FINISHED

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a static, curated sample dataset of raw Bronze news records in the Databricks dev catalog (`trading-signal-gen-dev.bronze.news_agent_eval_dataset`) to use for agent development and evaluation.

**Architecture:** We will create a PySpark script that uses Databricks Serverless compute to read a slice of the raw NDJSON from S3, applies the schema contract, filters out useless rows (null bodies), limits to 500 rows, and writes the result as a Delta table.

**Tech Stack:** PySpark, Databricks Serverless, Delta Lake.

---

### Task 1: Create Data Extraction Script

**Files:**
- Create: `notebooks/create_agent_eval_dataset.py`

- [ ] **Step 1: Write the PySpark script**

Create the file `notebooks/create_agent_eval_dataset.py` with the following content. This script initializes a Spark session, reads a subset of the S3 bucket, filters for valid body content, limits to 500 records, and writes to a Delta table.

```python
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
    target_table = f"{catalog}.bronze.news_agent_eval_dataset"

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
    parser.add_argument("--catalog", default="trading-signal-gen-dev", help="Target UC catalog")
    parser.add_argument("--bucket", default="ai-trading-raw-ap-southeast-1", help="Source S3 bucket")
    parser.add_argument("--limit", type=int, default=500, help="Number of records to sample")
    
    args = parser.parse_args()
    main(args.catalog, args.bucket, args.limit)
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/create_agent_eval_dataset.py
git commit -m "chore: add script to generate static agent eval dataset"
```

### Task 2: Execute Script on Databricks

Since this is a one-off developer task, we will run the script on Databricks Serverless using the MCP Databricks server.

- [ ] **Step 1: Execute the Python file on Databricks**

Use the `run_python_file_on_databricks` MCP tool to run the newly created file. The tool will automatically find a cluster or provision one for you, upload the file, and execute it. 

*(If you don't have access to the MCP tool, use the Databricks CLI: `databricks runs submit --python-task '{"python_file": "notebooks/create_agent_eval_dataset.py"}'` -- but this requires uploading the file to the workspace first. The MCP tool is strongly preferred).*

- [ ] **Step 2: Verify the table creation**

Use the `execute_sql` MCP tool to query the newly created table to ensure data was loaded correctly:

```sql
SELECT count(*) as row_count, count(distinct story_id) as unique_stories 
FROM `trading-signal-gen-dev`.bronze.news_agent_eval_dataset
```

Expected output: `row_count` should be 500 (or close to it if the partition had fewer valid records).