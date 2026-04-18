-- BRONZE LAYER of Tick Data (symbol BTC): Ingest data from Kinesis Stream
CREATE OR REFRESH STREAMING TABLE 
  `smart-trading-dev`.`01_bronze`.ticks_bronze
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
  awsAccessKey => secret('ticks_kinesis', 'aws_access_key_id'),
  awsSecretKey => secret('ticks_kinesis', 'aws_secret_access_key'),
  region => 'ap-southeast-1',
  initialPosition => 'earliest')

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
  awsAccessKey => secret('ticks_kinesis', 'aws_access_key_id'),
  awsSecretKey => secret('ticks_kinesis', 'aws_secret_access_key'),
  region => 'ap-southeast-1',
  initialPosition => 'earliest'
);





-- SILVER LAYER of Tick Data: Transform and Validate data
CREATE OR REFRESH STREAMING TABLE 
  `smart-trading-dev`.`02_silver`.ticks_silver
  COMMENT "SILVER LAYER of Tick Data: Transform and Validate data"
  TBLPROPERTIES ('quality' = 'silver')
AS 
WITH cte_parse AS (
  SELECT 
    from_json(
      payload,
      'timestamp TIMESTAMP,
      price STRING,
      quantity STRING,
      side STRING,
      trade_id STRING,
      symbol STRING,
      exchange STRING,
      batch_id STRING'
    ) AS parsed_data,
    kinesis_stream_name,
    kinesis_partition_key,
    kinesis_shard_id,
    kinesis_sequence_number,
    kinesis_arrival_time,
    bronze_arrival_time
  FROM STREAM `smart-trading-dev`.`01_bronze`.ticks_bronze
)
SELECT 
  parsed_data.timestamp AS timestamp,
  parsed_data.price AS price,
  parsed_data.quantity AS quantity,
  parsed_data.side AS side,
  parsed_data.trade_id AS trade_id,
  parsed_data.symbol AS symbol,
  parsed_data.exchange AS exchange,
  parsed_data.batch_id AS batch_id,
  kinesis_stream_name,
  kinesis_partition_key,
  kinesis_shard_id,
  kinesis_sequence_number,
  kinesis_arrival_time,
  bronze_arrival_time,
  current_timestamp() AS silver_arrival_time
FROM STREAM cte_parse

