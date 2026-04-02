# Create Raw Bronze AggTrades Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upload a local 4.5GB CSV of Binance aggTrades data to a Databricks Unity Catalog Volume, then use a PySpark script to ingest and transform it into a bronze table (`trading-signal-gen-dev.bronze.raw_aggTrades_samples`) that conforms to the `TICKS_RAW_SCHEMA`.

**Architecture:** 
1. Use `mcp_databricks_upload_file` (or Databricks CLI) to upload `BTCUSDT-aggTrades-2026-02.csv` to `/Volumes/trading-signal-gen-dev/default/uploads/BTCUSDT-aggTrades-2026-02.csv`.
2. Execute a PySpark script (`notebooks/create_agg_trades_eval_dataset.py`) on Databricks Serverless. The script reads the CSV, maps its 8 columns to the expected `TICKS_RAW_SCHEMA` fields (including adding static/derived columns like `symbol="BTCUSDT"` and `side`), and saves it as a Delta table.

**Tech Stack:** PySpark, Databricks Serverless, Unity Catalog Volumes, Delta Lake.

---

### Task 1: Upload the CSV Data to a UC Volume

We need to place the local file into a Databricks volume so Spark can read it efficiently.

- [ ] **Step 1: Verify the target volume exists (or create it)**

Run this command to check if the `default.uploads` volume exists in the dev catalog. If it fails, create it.

```bash
databricks volumes list trading-signal-gen-dev default | grep uploads || databricks volumes create trading-signal-gen-dev default uploads MANAGED
```

- [ ] **Step 2: Upload the local CSV file**

Use the Databricks CLI to upload the large file. Note: This will take a while for a 4.5GB file. 

```bash
databricks fs cp ../../Downloads/BTCUSDT-aggTrades-2026-02.csv dbfs:/Volumes/trading-signal-gen-dev/default/uploads/BTCUSDT-aggTrades-2026-02.csv --overwrite
```

*Note for agent: Alternatively, you can use the `mcp_databricks_upload_file` tool if that is faster/preferred.*

### Task 2: Create the PySpark Ingestion Script

**Files:**
- Create: `notebooks/create_agg_trades_eval_dataset.py`

- [x] **Step 1: Write the PySpark script**

Create `notebooks/create_agg_trades_eval_dataset.py`. The Binance aggTrades CSV has 8 unnamed columns in this order: `[agg_trade_id, price, quantity, first_trade_id, last_trade_id, timestamp, is_buyer_maker, is_best_match]`. We must map this to the `TICKS_RAW_SCHEMA`.

```python
import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, LongType

# Schema matching the raw Binance aggTrades CSV
BINANCE_CSV_SCHEMA = StructType([
    StructField("agg_trade_id", StringType(), nullable=True),
    StructField("price", StringType(), nullable=True),
    StructField("quantity", StringType(), nullable=True),
    StructField("first_trade_id", StringType(), nullable=True),
    StructField("last_trade_id", StringType(), nullable=True),
    StructField("timestamp", LongType(), nullable=True),
    StructField("is_buyer_maker", BooleanType(), nullable=True),
    StructField("is_best_match", BooleanType(), nullable=True),
])

def main(catalog: str, file_path: str):
    spark = SparkSession.builder.getOrCreate()
    
    target_table = f"{catalog}.bronze.raw_aggTrades_samples"

    print(f"Reading from: {file_path}")
    print(f"Writing to: {target_table}")

    # 1. Read the CSV
    df = (
        spark.read.format("csv")
        .option("header", "false")
        .schema(BINANCE_CSV_SCHEMA)
        .load(file_path)
    )

    # 2. Map to TICKS_RAW_SCHEMA expectations
    # Binance timestamp is in milliseconds. TICKS_RAW_SCHEMA expects an ISO8601 string.
    # is_buyer_maker=True means the maker was the buyer, meaning the taker (aggressor) sold to them.
    # So if is_buyer_maker=True, side="sell". If False, side="buy".
    mapped_df = (
        df
        .withColumn("timestamp", F.date_format(F.to_timestamp(F.col("timestamp") / 1000), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"))
        .withColumn("side", F.when(F.col("is_buyer_maker") == True, F.lit("sell")).otherwise(F.lit("buy")))
        .withColumn("trade_id", F.col("agg_trade_id"))
        .withColumn("symbol", F.lit("BTCUSDT"))
        .withColumn("exchange", F.lit("binance"))
        .withColumn("batch_id", F.lit("batch_manual_csv_upload"))
        .select(
            "timestamp", 
            "price", 
            "quantity", 
            "side", 
            "trade_id", 
            "symbol", 
            "exchange", 
            "batch_id"
        )
    )

    print("Sample of mapped data:")
    mapped_df.show(5, truncate=False)
    
    # 3. Write to Delta
    print(f"Writing data to {target_table}...")
    mapped_df.write.format("delta").mode("overwrite").saveAsTable(target_table)
    
    # 4. Print stats
    count = spark.table(target_table).count()
    print(f"Successfully created static eval dataset: {target_table} with {count} rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create static bronze tick eval dataset")
    parser.add_argument("--catalog", default="trading-signal-gen-dev", help="Target UC catalog")
    parser.add_argument("--file_path", default="/Volumes/trading-signal-gen-dev/default/uploads/BTCUSDT-aggTrades-2026-02.csv", help="Path to source CSV file in Volume")
    
    args = parser.parse_args()
    main(args.catalog, args.file_path)
```

- [x] **Step 2: Commit the script**

```bash
git add notebooks/create_agg_trades_eval_dataset.py
git commit -m "chore: add script to ingest raw aggTrades CSV into bronze layer"
```

### Task 3: Execute the Script on Databricks

- [x] **Step 1: Run the Python file on Databricks Serverless**

Use the `mcp_databricks_run_python_file_on_databricks` tool to execute `notebooks/create_agg_trades_eval_dataset.py`. This will provision serverless compute, upload the script, and execute it against the CSV file we uploaded to the Volume.

- [x] **Step 2: Verify the table in DBSQL**

Use the `execute_sql` MCP tool to query the newly created table:

```sql
SELECT 
    count(*) as row_count, 
    min(timestamp) as min_time, 
    max(timestamp) as max_time,
    sum(case when side='buy' then 1 else 0 end) as buys,
    sum(case when side='sell' then 1 else 0 end) as sells
FROM `trading-signal-gen-dev`.bronze.raw_aggTrades_samples
```

Expected output: `row_count` should be exactly `52474665`.
