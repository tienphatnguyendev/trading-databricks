# Migrate Workspace and Populate Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the MAS codebase to use the new `smart-trading-dev` catalog and `03_gold` / `01_bronze` schemas, populate the taxonomy tables, and provision the Vector Search resources using the Databricks Python SDK.

**Architecture:** Updates hardcoded constants and defaults across the codebase. Replaces mock MCP calls in the provisioning script with actual `databricks.sdk.WorkspaceClient` calls to create the Vector Search endpoint and index. Creates an empty `news_causal_features` table prior to index creation so the Delta Sync index can be provisioned successfully on cold-start. Updates the documentation.

**Tech Stack:** Databricks SDK (`databricks-sdk`), Databricks Connect (`databricks-connect`), PySpark

---

### Task 1: Update Constants and Defaults

**Files:**
- Modify: `src/trading_databricks/agents/config.py`
- Modify: `scripts/seed_taxonomy.py`
- Modify: `scripts/provision_vector_search.py`
- Modify: `src/trading_databricks/agents/pipeline.py`
- Modify: `notebooks/create_agent_eval_dataset.py`
- Modify: `notebooks/create_agg_trades_eval_dataset.py`
- Modify: `notebooks/research_mas_causal_news.py`
- Modify: `notebooks/research_ml_signal_gen.py`

- [ ] **Step 1: Update `src/trading_databricks/agents/config.py`**
Change `catalog`, `schema`, and `vs_index_name` defaults:
```python
    catalog: str = "smart-trading-dev"
    schema: str = "03_gold"
    vs_index_name: str = "smart-trading-dev.03_gold.causal_chains_idx"
```

- [ ] **Step 2: Update `scripts/seed_taxonomy.py`**
Change parser defaults to `smart-trading-dev` and `03_gold`. Add `serverless(True)` to the spark session:
```python
    parser.add_argument("--catalog", default="smart-trading-dev")
    parser.add_argument("--schema", default="03_gold")
    # ...
    spark = DatabricksSession.builder.serverless(True).getOrCreate()
```

- [ ] **Step 3: Update `scripts/provision_vector_search.py` configs**
Change the dev config environment:
```python
    if env == "dev":
        config = MASConfig(
            catalog="smart-trading-dev",
            vs_index_name="smart-trading-dev.03_gold.causal_chains_idx",
        )
```

- [ ] **Step 4: Update `src/trading_databricks/agents/pipeline.py`**
Update `_CATALOG`:
```python
_CATALOG = "smart-trading-dev"
```

- [ ] **Step 5: Update Notebooks**
Change `"smart-trading-dev"` to `"smart-trading-dev"` and `"bronze"` to `"01_bronze"` in the notebooks:
- `notebooks/create_agent_eval_dataset.py`
- `notebooks/create_agg_trades_eval_dataset.py`
- `notebooks/research_mas_causal_news.py`
- `notebooks/research_ml_signal_gen.py`

- [ ] **Step 6: Commit changes**
```bash
git add src/trading_databricks/agents/config.py scripts/seed_taxonomy.py scripts/provision_vector_search.py src/trading_databricks/agents/pipeline.py notebooks/*.py
git commit -m "chore: update catalog and schemas to smart-trading-dev"
```

### Task 2: Populate Taxonomy Data

**Files:**
- Modify: None (Running existing script)

- [ ] **Step 1: Execute `seed_taxonomy.py`**
```bash
uv run python scripts/seed_taxonomy.py
```
Expected: Output showing 34 rows for `treatment_taxonomy` and 40 rows for `mechanism_whitelist`.

### Task 3: Update Vector Search Provisioning Script

**Files:**
- Modify: `scripts/provision_vector_search.py`

- [ ] **Step 1: Update SDK imports and table creation logic**
Replace the `try/except` block importing `MCP_CLIENT` with actual SDK imports and add table creation:
```python
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.vectorsearch import EndpointType, DeltaSyncVectorIndexSpecResponse
    from databricks.connect import DatabricksSession
    from trading_databricks.agents.schemas import CAUSAL_FEATURES_SCHEMA

    w = WorkspaceClient()

    # Ensure source table exists before index creation
    logger.info("Ensuring source table exists...")
    spark = DatabricksSession.builder.serverless(True).getOrCreate()
    source_table = f"{config.catalog}.{config.schema}.news_causal_features"
    spark.createDataFrame([], CAUSAL_FEATURES_SCHEMA).write.format("delta").option("mergeSchema", "true").mode("ignore").saveAsTable(source_table)
```

- [ ] **Step 2: Update Endpoint Creation Logic**
Replace the `mcp__databricks__create_or_update_vs_endpoint` comment with:
```python
    try:
        logger.info(f"Creating/updating VS endpoint: {endpoint_name}...")
        w.vector_search_endpoints.create_endpoint_and_wait(
            name=endpoint_name,
            endpoint_type=EndpointType.STANDARD
        )
        logger.info(f"  ✓ Endpoint '{endpoint_name}' ready")
    except Exception as e:
        logger.error(f"Failed to create endpoint: {e}")
        raise
```

- [ ] **Step 3: Update Index Creation Logic**
Replace the `mcp__databricks__create_or_update_vs_index` comment with:
```python
    try:
        logger.info(f"Creating/updating VS index: {index_name}...")
        w.vector_search_indexes.create_index_and_wait(
            name=index_name,
            endpoint_name=endpoint_name,
            primary_key="story_id",
            index_type="DELTA_SYNC",
            delta_sync_index_spec=DeltaSyncVectorIndexSpecResponse(
                source_table=source_table,
                embedding_source_columns=[
                    {
                        "name": "trigger_event",
                        "embedding_model_endpoint_name": "databricks-gte-large-en",
                    }
                ],
                pipeline_type="TRIGGERED",
            )
        )
        logger.info(f"  ✓ Index '{index_name}' configured")
    except Exception as e:
        logger.error(f"Failed to create index: {e}")
        raise
```

- [ ] **Step 4: Commit changes**
```bash
git add scripts/provision_vector_search.py
git commit -m "feat: use databricks sdk for vector search provisioning"
```

### Task 4: Run Vector Search Provisioning

**Files:**
- Modify: None (Running existing script)

- [ ] **Step 1: Execute `provision_vector_search.py`**
```bash
uv run python scripts/provision_vector_search.py --env dev
```
Expected: Output showing successful creation of the `mas-few-shot` endpoint and the `smart-trading-dev.03_gold.causal_chains_idx` index.

### Task 5: Update Documentation

**Files:**
- Modify: `docs/superpowers/specs/*.md`
- Modify: `docs/advanced-ml-modelling/*.md`
- Modify: `GEMINI.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Replace catalog references in docs**
Run a find-and-replace command to update documentation:
```bash
find docs/ -name "*.md" -type f -exec sed -i '' 's/smart-trading-dev/smart-trading-dev/g' {} +
find docs/ -name "*.md" -type f -exec sed -i '' 's/\.gold\./.03_gold./g' {} +
find docs/ -name "*.md" -type f -exec sed -i '' 's/`03_gold`/`03_gold`/g' {} +
find docs/ -name "*.md" -type f -exec sed -i '' 's/`01_bronze`/`01_bronze`/g' {} +
find docs/ -name "*.md" -type f -exec sed -i '' 's/\.bronze\./.01_bronze./g' {} +

# Update the main project files
sed -i '' 's/smart-trading-dev/smart-trading-dev/g' GEMINI.md
sed -i '' 's/\.gold\./.03_gold./g' GEMINI.md
sed -i '' 's/`03_gold`/`03_gold`/g' GEMINI.md
sed -i '' 's/`01_bronze`/`01_bronze`/g' GEMINI.md
sed -i '' 's/\.bronze\./.01_bronze./g' GEMINI.md

sed -i '' 's/smart-trading-dev/smart-trading-dev/g' CLAUDE.md
sed -i '' 's/\.gold\./.03_gold./g' CLAUDE.md
sed -i '' 's/`03_gold`/`03_gold`/g' CLAUDE.md
sed -i '' 's/`01_bronze`/`01_bronze`/g' CLAUDE.md
sed -i '' 's/\.bronze\./.01_bronze./g' CLAUDE.md
```

- [ ] **Step 2: Commit changes**
```bash
git add docs/ GEMINI.md CLAUDE.md
git commit -m "docs: update catalog and schemas to smart-trading-dev"
```