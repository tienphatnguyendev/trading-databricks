-- December 2025 UTC. Each five-minute bar includes its start and excludes its end.
WITH valid_trades AS (
    SELECT
        agg_trade_id,
        STRUCT(transact_time, price, quantity) AS trade
    FROM `smart_trading_dev`.`01_bronze`.`raw_aggtrades` VERSION AS OF __SOURCE_VERSION__
    WHERE symbol = 'BTCUSDT'
        AND trade_date >= DATE '2025-12-01'
        AND trade_date < DATE '2026-01-01'
        AND transact_time >= 1764547200000000
        AND transact_time < 1767225600000000
        AND agg_trade_id >= 0
        AND price > 0
        AND quantity > 0
),
trades_by_id AS (
    SELECT
        agg_trade_id,
        MIN(trade) AS first_trade,
        MAX(trade) AS last_trade,
        COUNT(*) - 1 AS repeated_rows
    FROM valid_trades
    GROUP BY agg_trade_id
),
bucketed_trades AS (
    -- Keep one deterministic row per ID. The runner rejects conflicting IDs.
    SELECT
        ((first_trade.transact_time DIV 300000000) + 1) * 300000000 AS bar_end_us,
        first_trade.transact_time AS transact_time,
        STRUCT(first_trade.transact_time, agg_trade_id) AS trade_order,
        first_trade.price AS price,
        first_trade.quantity AS quantity,
        repeated_rows,
        first_trade <> last_trade AS has_conflict
    FROM trades_by_id
)
SELECT
    bar_end_us,
    MAX(transact_time) AS last_trade_us,
    MIN_BY(price, trade_order) AS `open`,
    MAX(price) AS high,
    MIN(price) AS low,
    MAX_BY(price, trade_order) AS `close`,
    SUM(quantity) AS volume,
    COUNT(*) AS trade_count,
    SUM(repeated_rows) AS duplicate_rows,
    COUNT_IF(has_conflict) AS conflicting_trade_ids
FROM bucketed_trades
GROUP BY bar_end_us
ORDER BY bar_end_us;
