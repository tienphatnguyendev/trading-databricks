"""Catalog and path constants for the BTCUSDT pipeline."""

CATALOG = "smart-trading-dev"
VOLUME  = "00_volume"
BRONZE  = "01_bronze"
SILVER  = "02_silver"
GOLD    = "03_gold"

RAW_AGGTRADES      = "raw_aggtrades"
RAW_AGGTRADES_PATH = f"`{CATALOG}`.`{BRONZE}`.`{RAW_AGGTRADES}`"

RAW_NEWS      = "raw_news"
RAW_NEWS_PATH = f"`{CATALOG}`.`{BRONZE}`.`{RAW_NEWS}`"
