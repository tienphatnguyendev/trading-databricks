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
