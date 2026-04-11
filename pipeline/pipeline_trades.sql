-- BRONZE LAYER of Tick Data (symbol BTC): Ingest data from Kinesis Stream
CREATE OR REFRESH STREAMING TABLE 
  `smart-trading-dev`.`01_bronze`.trades_bronze
  COMMENT "BRONZE LAYER of Tick Data: Ingest data from Kinesis Stream"
  TBLPROPERTIES ("quality" = "bronze")
AS 
---- Stream 1: BTC
SELECT
  CAST(data AS STRING) AS payload, -- cast json data to string
  partitionKey AS kinesis_partition_key,
  stream AS kinesis_stream_name,
  shardId AS kinesis_shard_id,
  sequenceNumber AS kinesis_sequence_number,
  approximateArrivalTimestamp AS kinesis_arrival_time,
  current_timestamp() AS bronze_arrival_time
FROM STREAM read_kinesis (
  streamName => 'tick-stream-btcusdt',
  serviceCredential => 'databricks-kinesis',
  region => 'ap-southeast-1',
  initialPosition => 'latest')

UNION ALL -- union BTC and ETH data

---- Stream 2: ETH
SELECT
  CAST(data AS STRING) AS payload, -- cast json data to string
  partitionKey AS kinesis_partition_key,
  stream AS kinesis_stream_name,
  shardId AS kinesis_shard_id,
  sequenceNumber AS kinesis_sequence_number,
  approximateArrivalTimestamp AS kinesis_arrival_time,
  current_timestamp() AS bronze_arrival_time

FROM STREAM read_kinesis (
  streamName => 'tick-stream-ethusdt',
  serviceCredential => 'databricks-kinesis',
  region => 'ap-southeast-1',
  initialPosition => 'latest'
);





-- SILVER LAYER of Tick Data: Transform and Validate data
CREATE OR REFRESH STREAMING TABLE 
  `smart-trading-dev`.`02_silver`.trades_silver
  COMMENT "SILVER LAYER of Tick Data: Transform and Validate data"
  TBLPROPERTIES ('quality' = 'silver')
AS 
WITH cte_parse AS (
  SELECT 
    from_json(
      payload,
      'aggregate_trade_id STRING,
      price STRING,
      quantity STRING,
      first_trade_id LONG,
      last_trade_id LONG,
      aggregate_timestamp LONG,
      is_buyer_maker BOOLEAN,
      symbol STRING'
    ) AS parsed_data,
    kinesis_stream_name,
    kinesis_partition_key,
    kinesis_shard_id,
    kinesis_sequence_number,
    kinesis_arrival_time,
    bronze_arrival_time
  FROM STREAM `smart-trading-dev`.`01_bronze`.trades_bronze
)
SELECT 
  parsed_data.aggregate_trade_id AS aggregate_trade_id,
  CAST(parsed_data.price AS DOUBLE) AS price,
  CAST(parsed_data.quantity AS DOUBLE) AS quantity,
  parsed_data.first_trade_id AS first_trade_id,
  parsed_data.last_trade_id AS last_trade_id,
  timestamp_millis(parsed_data.aggregate_timestamp) AS aggregate_timestamp,
  parsed_data.is_buyer_maker AS is_buyer_maker,
  parsed_data.symbol AS symbol,
  kinesis_stream_name,
  kinesis_partition_key,
  kinesis_shard_id,
  kinesis_sequence_number,
  kinesis_arrival_time,
  bronze_arrival_time,
  current_timestamp() AS silver_arrival_time
FROM STREAM cte_parse

