"""Silver transformation: bronze.ticks_raw → silver.ticks_cleaned.

Planned transformations:
1. Cast timestamp to TIMESTAMP (microsecond precision)
2. Cast price, quantity to DECIMAL(18, 8)
3. Compute dollar_value = price * quantity
4. Filter invalid ticks (price <= 0 or quantity <= 0)
5. Deduplicate by trade_id
6. Add _ingested_at metadata column

Write mode: overwrite.

STATUS: awaiting tick-scraper-aws schema definition.
TODO: implement clean_ticks(), run_silver_transform()
"""
from __future__ import annotations
