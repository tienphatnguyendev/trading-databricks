
-- -- GOLD LAYER of Trades Data: OHLC by Dates and Symbols
-- CREATE OR REFRESH MATERIALIZED VIEW `smart-trading-dev`.`03_gold`.ohlc_gold
-- COMMENT "GOLD LAYER of Trades Data: OHLC by Dates and Symbols"
-- TBLPROPERTIES ('quality' = 'gold')
-- AS
-- SELECT 
--   -- GROUP BY columns
--   CAST(aggregate_timestamp AS DATE) AS trade_date,
--   symbol,

--   -- OHLC metrics
--   min_by(price, aggregate_timestamp) as open_price,
--   MAX(price) AS high_price,
--   MIN(price) AS low_price,
--   max_by(price, aggregate_timestamp) as close_price,

--   -- quantity
--   SUM(quantity) AS volume_total,
--   SUM(CASE WHEN is_buyer_maker=True THEN quantity END) AS volume_buy,
--   SUM(CASE WHEN is_buyer_maker=False THEN quantity END) AS volume_sell,

--   -- quote
--   SUM(price*quantity) AS value_total,
--   SUM(CASE WHEN is_buyer_maker=True THEN quantity*price END) AS value_buy,
--   SUM(CASE WHEN is_buyer_maker=False THEN quantity*price END) AS value_sell
  
  
-- FROM `smart-trading-dev`.`02_silver`.trades_silver
-- GROUP BY trade_date, symbol
