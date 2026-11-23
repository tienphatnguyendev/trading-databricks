"""SDP data quality expectation rules for the ticks domain."""
from __future__ import annotations

from trading_databricks.common.quality import not_null, positive

# ── Bronze expectations ───────────────────────────────────────────────────────

BRONZE_VALID_TRADE_ID = not_null("trade_id")
BRONZE_VALID_TIMESTAMP = not_null("timestamp")
BRONZE_VALID_PRICE = not_null("price")
BRONZE_VALID_QUANTITY = not_null("quantity")

# ── Silver expectations ───────────────────────────────────────────────────────

SILVER_POSITIVE_PRICE = positive("CAST(price AS DOUBLE)")
SILVER_POSITIVE_QUANTITY = positive("CAST(quantity AS DOUBLE)")
SILVER_VALID_TRADE_ID = not_null("trade_id")
SILVER_VALID_TIMESTAMP = not_null("timestamp")
