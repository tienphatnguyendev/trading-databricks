------------------------- BRONZE LAYER -------------------------
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





------------------------- SILVER LAYER -------------------------
CREATE OR REFRESH STREAMING TABLE 
  `smart-trading-dev`.`02_silver`.trades_silver
  COMMENT "SILVER LAYER of Tick Data: Transform and Validate data"
  TBLPROPERTIES ('quality' = 'silver')
AS 
---- CTE for breaking down the JSON structure stored in column pay_load in Bronze Layer
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
---- structuring the columns from the CTE 
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
FROM STREAM cte_parse;




------------------------- GOLD LAYER -------------------------
-- OHLC by Dates and Symbols
CREATE OR REFRESH MATERIALIZED VIEW `smart-trading-dev`.`03_gold`.ohlc_by_dates_gold
COMMENT "GOLD LAYER of Trades Data: OHLC by Dates and Symbols"
TBLPROPERTIES ('quality' = 'gold')
AS
SELECT 
  -- GROUP BY columns
  CAST(aggregate_timestamp AS DATE) AS trade_date,
  symbol,

  -- OHLC metrics
  min_by(price, aggregate_timestamp) as open_price,
  MAX(price) AS high_price,
  MIN(price) AS low_price,
  max_by(price, aggregate_timestamp) as close_price,

  -- quantity
  SUM(quantity) AS volume_total,
  SUM(CASE WHEN is_buyer_maker=True THEN quantity END) AS volume_buy,
  SUM(CASE WHEN is_buyer_maker=False THEN quantity END) AS volume_sell,

  -- quote
  SUM(price*quantity) AS value_total,
  SUM(CASE WHEN is_buyer_maker=True THEN quantity*price END) AS value_buy,
  SUM(CASE WHEN is_buyer_maker=False THEN quantity*price END) AS value_sell
  
FROM `smart-trading-dev`.`02_silver`.trades_silver
GROUP BY trade_date, symbol;



-- -- OHLC by Dollar BAR (threshold: $1 mil & symbol: BTC)
-- WITH trades AS (
--   SELECT
--     price,
--     quantity,
--     aggregate_timestamp,
--     is_buyer_maker,
--     symbol,
--     price * quantity AS dollar_value
--   FROM
--     `smart-trading-dev`.`02_silver`.trades_silver
--   WHERE
--     symbol = 'BTCUSDT'
-- ),
-- cumulative AS (
--   SELECT
--     *,
--     SUM(dollar_value) OVER (
--         ORDER BY aggregate_timestamp
--         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
--       ) AS cumulative_dollar
--   FROM
--     trades
-- ),
-- bar_assigned AS (
--   SELECT
--     *,
--     FLOOR(cumulative_dollar / 1000000) AS bar_index
--   FROM
--     cumulative
-- )
-- SELECT
--   bar_index,
--   -- Time range of the bar
--   MIN(aggregate_timestamp) AS bar_start_time,
--   MAX(aggregate_timestamp) AS bar_end_time,
--   -- OHLC
--   min_by(price, aggregate_timestamp) AS open_price,
--   MAX(price) AS high_price,
--   MIN(price) AS low_price,
--   max_by(price, aggregate_timestamp) AS close_price,
--   -- Volume & Value
--   SUM(quantity) AS volume_total,
--   SUM(dollar_value) AS dollar_value_total,
--   COUNT(*) AS trade_count,
--   -- Buy/Sell breakdown
--   SUM(
--     CASE
--       WHEN is_buyer_maker = True THEN quantity
--     END
--   ) AS volume_buy,
--   SUM(
--     CASE
--       WHEN is_buyer_maker = False THEN quantity
--     END
--   ) AS volume_sell
-- FROM
--   bar_assigned
-- GROUP BY
--   bar_index
-- ORDER BY
--   bar_index



