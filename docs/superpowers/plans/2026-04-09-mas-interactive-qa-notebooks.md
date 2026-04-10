# MAS Interactive QA Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an interactive Databricks Python notebook for data scientists to debug and QA the MAS causality pipeline.
**Architecture:** A single Python file formatted as a Databricks Notebook (using `# COMMAND ----------` and `# MAGIC` blocks) broken into 6 modular sections: Setup, Gatekeeper, RAG/Routing, Extractor, Validator, and E2E MLflow.
**Tech Stack:** PySpark, MLflow Tracing, LangGraph, Databricks Widgets, pandas, spaCy, Python AST (for testing).

---

### Task 1: Setup & Data Ingestion (Section 1)

**Files:**
- Create: `tests/notebooks/test_mas_interactive_qa.py`
- Create: `src/trading_databricks/notebooks/mas_interactive_qa.py`

- [ ] **Step 1: Write the failing test**
Create `tests/notebooks/test_mas_interactive_qa.py`:
```python
import ast
from pathlib import Path

def test_notebook_has_setup_section():
    notebook_path = Path("src/trading_databricks/notebooks/mas_interactive_qa.py")
    assert notebook_path.exists(), "Notebook file not found"
    
    content = notebook_path.read_text()
    assert "Section 1: Setup & Data Ingestion" in content
    
    # Verify imports and dbutils usage
    tree = ast.parse(content)
    dbutils_found = any(
        isinstance(node, ast.Expr) and 
        isinstance(node.value, ast.Call) and 
        isinstance(node.value.func, ast.Attribute) and 
        node.value.func.attr == "text" 
        for node in ast.walk(tree)
    )
    assert dbutils_found, "dbutils.widgets.text not found in notebook"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/notebooks/test_mas_interactive_qa.py -v`
Expected: FAIL with "Notebook file not found"

- [ ] **Step 3: Write minimal implementation**
Create `src/trading_databricks/notebooks/mas_interactive_qa.py`:
```python
# Databricks notebook source
# MAGIC %md
# MAGIC # MAS Interactive QA & Debugging Notebook
# MAGIC This notebook provides tools to evaluate the MAS causality pipeline both macroscopically and microscopically.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Section 1: Setup & Data Ingestion
# MAGIC Initialize connections, pull sample dataset, and map to MASState.

# COMMAND ----------
import pandas as pd
import mlflow
from pyspark.sql import SparkSession

# Fallback for local testing (Databricks injects spark/dbutils)
try:
    dbutils.widgets.text("limit", "10", "Number of articles to sample") # type: ignore
    dbutils.widgets.text("l1_category", "All", "Filter by L1 Category") # type: ignore
except NameError:
    class MockWidgets:
        def text(self, *args, **kwargs): pass
        def get(self, key): return "10" if key == "limit" else "All"
    class MockDbutils:
        widgets = MockWidgets()
    dbutils = MockDbutils()
    spark = SparkSession.builder.getOrCreate()

limit = int(dbutils.widgets.get("limit"))
l1_category = dbutils.widgets.get("l1_category")

print(f"Loading {limit} rows from smart-trading-dev.others.mas_training_dataset...")
df = spark.sql(f"SELECT * FROM `smart-trading-dev`.others.mas_training_dataset LIMIT {limit}")
pdf = df.toPandas()

# Convert to list of MASState dicts for easier manual feeding to nodes
states = []
for _, row in pdf.iterrows():
    states.append({
        "headline": row.get("headline", ""),
        "body": row.get("body", ""),
        "published_at": str(row.get("published_at", "")),
        "story_id": row.get("story_id", "")
    })
print(f"Loaded {len(states)} articles.")
```

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/notebooks/test_mas_interactive_qa.py -v`
Expected: PASS

- [ ] **Step 5: Syntax and Linting**
Run: `uv run ruff check src/trading_databricks/notebooks/`
Fix any resulting errors.

- [ ] **Step 6: Commit**
```bash
git add tests/notebooks/test_mas_interactive_qa.py src/trading_databricks/notebooks/mas_interactive_qa.py
git commit -m "test: add test and implementation for MAS notebook setup section"
```

---

### Task 2: Gatekeeper (Stage 1) Deep Dive (Section 2)

**Files:**
- Modify: `tests/notebooks/test_mas_interactive_qa.py:20-30`
- Modify: `src/trading_databricks/notebooks/mas_interactive_qa.py`

- [ ] **Step 1: Write the failing test**
Append to `tests/notebooks/test_mas_interactive_qa.py`:
```python
def test_notebook_has_gatekeeper_section():
    content = Path("src/trading_databricks/notebooks/mas_interactive_qa.py").read_text()
    assert "Section 2: Gatekeeper (Stage 1) Deep Dive" in content
    assert "is_crypto_relevant" in content
    assert "extract_ner_features" in content
```

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/notebooks/test_mas_interactive_qa.py::test_notebook_has_gatekeeper_section -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**
Append to `src/trading_databricks/notebooks/mas_interactive_qa.py`:
```python
# COMMAND ----------
# MAGIC %md
# MAGIC ## Section 2: Gatekeeper (Stage 1) Deep Dive
# MAGIC Inspect articles dropped by the spaCy EntityRuler (False Negatives).

# COMMAND ----------
from trading_databricks.agents.gatekeeper import is_crypto_relevant, extract_ner_features, build_nlp
from trading_databricks.agents.config import GatekeeperConfig

gk_config = GatekeeperConfig()
nlp = build_nlp()

gatekeeper_results = []
for state in states:
    is_impactful, conf = is_crypto_relevant(state["headline"], state["body"], config=gk_config)
    
    # Deep dive into exact entities
    text = state["headline"] + " " + state["body"][:gk_config.text_window]
    doc = nlp(text)
    features = extract_ner_features(doc)
    extracted_entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    gatekeeper_results.append({
        "story_id": state.get("story_id"),
        "headline": state["headline"],
        "is_impactful": is_impactful,
        "extracted_entities": extracted_entities,
        **features
    })

gk_df = pd.DataFrame(gatekeeper_results)
print("Gatekeeper Breakdown:")
print(gk_df["is_impactful"].value_counts())

# Show false negatives (if you know the dataset contains crypto articles)
print("\nSample articles dropped by Gatekeeper (is_impactful=False):")
display(gk_df[gk_df["is_impactful"] == False].head(5)) # type: ignore
```

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/notebooks/test_mas_interactive_qa.py::test_notebook_has_gatekeeper_section -v`
Expected: PASS

- [ ] **Step 5: Syntax and Linting**
Run: `uv run ruff check src/trading_databricks/notebooks/`

- [ ] **Step 6: Commit**
```bash
git add tests/notebooks/test_mas_interactive_qa.py src/trading_databricks/notebooks/mas_interactive_qa.py
git commit -m "feat: add Gatekeeper deep dive section to QA notebook"
```

---

### Task 3: RAG & Routing Inspection (Section 3)

**Files:**
- Modify: `tests/notebooks/test_mas_interactive_qa.py:31-40`
- Modify: `src/trading_databricks/notebooks/mas_interactive_qa.py`

- [ ] **Step 1: Write the failing test**
Append to `tests/notebooks/test_mas_interactive_qa.py`:
```python
def test_notebook_has_routing_section():
    content = Path("src/trading_databricks/notebooks/mas_interactive_qa.py").read_text()
    assert "Section 3: RAG & Routing Inspection" in content
    assert "few_shot_retriever" in content
    assert "uncertainty_router" in content
```

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/notebooks/test_mas_interactive_qa.py::test_notebook_has_routing_section -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**
Append to `src/trading_databricks/notebooks/mas_interactive_qa.py`:
```python
# COMMAND ----------
# MAGIC %md
# MAGIC ## Section 3: RAG & Routing Inspection
# MAGIC Evaluate Vector Search similarity scores to tune routing thresholds.

# COMMAND ----------
from trading_databricks.agents.nodes.few_shot_retriever import few_shot_retriever
from trading_databricks.agents.nodes.uncertainty_router import uncertainty_router
import matplotlib.pyplot as plt

# We wrap this in try-except because it requires active VS/Spark connection
try:
    routing_results = []
    for state in states[:5]:  # Limit to 5 for fast sandbox iteration
        # Run Few Shot Retriever
        retrieved_state = few_shot_retriever(state.copy(), config=MASConfig())
        
        # Run Uncertainty Router
        routed_state = uncertainty_router(retrieved_state, config=MASConfig())
        
        examples = retrieved_state.get("few_shot_examples", [])
        top_sim = examples[0].get("similarity_score", 0.0) if examples else 0.0
        
        routing_results.append({
            "headline": state["headline"],
            "top_similarity": top_sim,
            "extraction_path": routed_state.get("extraction_path")
        })

    routing_df = pd.DataFrame(routing_results)
    display(routing_df) # type: ignore

    # Plot histogram if enough data
    if len(routing_df) > 1:
        routing_df['top_similarity'].plot(kind='hist', bins=20, title='Top Similarity Score Distribution')
        plt.axvline(x=MASConfig().router.high_similarity_threshold, color='g', linestyle='--', label='Fast Threshold')
        plt.axvline(x=MASConfig().router.low_similarity_threshold, color='r', linestyle='--', label='Careful Threshold')
        plt.legend()
        plt.show()
except Exception as e:
    print(f"Skipping RAG execution (requires active VS/Spark): {e}")
```

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/notebooks/test_mas_interactive_qa.py::test_notebook_has_routing_section -v`
Expected: PASS

- [ ] **Step 5: Syntax and Linting**
Run: `uv run ruff check src/trading_databricks/notebooks/`

- [ ] **Step 6: Commit**
```bash
git add tests/notebooks/test_mas_interactive_qa.py src/trading_databricks/notebooks/mas_interactive_qa.py
git commit -m "feat: add RAG and Routing inspection section to QA notebook"
```

---

### Task 4: LLM Causality Extractor (Section 4)

**Files:**
- Modify: `tests/notebooks/test_mas_interactive_qa.py:41-50`
- Modify: `src/trading_databricks/notebooks/mas_interactive_qa.py`

- [ ] **Step 1: Write the failing test**
Append to `tests/notebooks/test_mas_interactive_qa.py`:
```python
def test_notebook_has_extractor_section():
    content = Path("src/trading_databricks/notebooks/mas_interactive_qa.py").read_text()
    assert "Section 4: LLM Causality Extractor (Autopsy)" in content
    assert "causality_extractor" in content
    assert "build_system_prompt" in content
```

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/notebooks/test_mas_interactive_qa.py::test_notebook_has_extractor_section -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**
Append to `src/trading_databricks/notebooks/mas_interactive_qa.py`:
```python
# COMMAND ----------
# MAGIC %md
# MAGIC ## Section 4: LLM Causality Extractor (Autopsy)
# MAGIC Inspect exact prompts and raw LLM responses for the careful path.

# COMMAND ----------
from trading_databricks.agents.nodes.causality_extractor import causality_extractor
from trading_databricks.agents.prompts import build_system_prompt
from trading_databricks.agents.taxonomy import load_taxonomy, load_whitelist

try:
    # Sandbox: Manually set a state to 'careful' path to observe majority voting
    sandbox_state = states[0].copy()
    sandbox_state["extraction_path"] = "careful"
    sandbox_state["few_shot_examples"] = [] # Empty for cold-start test
    
    print(f"Testing extraction for headline: {sandbox_state['headline']}")
    
    # Show what the LLM sees
    taxonomy = load_taxonomy(spark, catalog=MASConfig().catalog, schema=MASConfig().schema)
    whitelist = load_whitelist(spark, catalog=MASConfig().catalog, schema=MASConfig().schema)
    system_prompt = build_system_prompt(taxonomy, whitelist, [], "careful")
    print(f"\n--- SYSTEM PROMPT (Truncated) ---\n{system_prompt[:500]}...\n---------------------------------")
    
    # Run the extractor (calls databricks-claude-sonnet)
    extracted_state = causality_extractor(sandbox_state, config=MASConfig())
    print("\nFinal Majority Vote Extraction:")
    print(pd.Series(extracted_state.get("extraction", {})))

except Exception as e:
    print(f"Skipping Extractor execution (requires active MLflow deployments): {e}")
```

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/notebooks/test_mas_interactive_qa.py::test_notebook_has_extractor_section -v`
Expected: PASS

- [ ] **Step 5: Syntax and Linting**
Run: `uv run ruff check src/trading_databricks/notebooks/`

- [ ] **Step 6: Commit**
```bash
git add tests/notebooks/test_mas_interactive_qa.py src/trading_databricks/notebooks/mas_interactive_qa.py
git commit -m "feat: add LLM Causality Extractor autopsy to QA notebook"
```

---

### Task 5: Mechanism Validator (Section 5)

**Files:**
- Modify: `tests/notebooks/test_mas_interactive_qa.py:51-60`
- Modify: `src/trading_databricks/notebooks/mas_interactive_qa.py`

- [ ] **Step 1: Write the failing test**
Append to `tests/notebooks/test_mas_interactive_qa.py`:
```python
def test_notebook_has_validator_section():
    content = Path("src/trading_databricks/notebooks/mas_interactive_qa.py").read_text()
    assert "Section 5: Mechanism Validator (Constraints Debugging)" in content
    assert "validate_extraction" in content
    assert "_validate_evidence_spans" in content
```

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/notebooks/test_mas_interactive_qa.py::test_notebook_has_validator_section -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**
Append to `src/trading_databricks/notebooks/mas_interactive_qa.py`:
```python
# COMMAND ----------
# MAGIC %md
# MAGIC ## Section 5: Mechanism Validator (Constraints Debugging)
# MAGIC Debug evidence span substring matching and semantic relevance scoring (Module B).

# COMMAND ----------
from trading_databricks.agents.nodes.mechanism_validator import validate_extraction, _validate_evidence_spans

# Create a mock embedding function for sandbox testing
def mock_embed_fn(text: str) -> list[float]:
    import numpy as np
    return list(np.random.rand(1024))

try:
    # Sandbox: Create a problematic extraction missing exact evidence
    mock_extraction = {
        "treatment_class": "Macro.rate_decision",
        "mechanism": "liquidity_contraction",
        "expected_direction": "bearish",
        "trigger_event": "Fed raises rates",
        "evidence_spans": ["The Fed raised rates by 50 basis points"] # Intentionally paraphrased
    }
    
    body = "The Federal Reserve increased interest rates by 50 basis points today."
    
    print(f"Evidence Spans: {mock_extraction['evidence_spans']}")
    print(f"Article Body: {body}")
    
    # Test strict substring matching
    passed = _validate_evidence_spans(mock_extraction['evidence_spans'], body) # type: ignore
    print(f"\nStrict Substring Match Passed: {passed}")
    if not passed:
        print("Data Scientist Action: The LLM paraphrased the evidence. Need to adjust the system prompt to enforce 'VERBATIM' copying.")
    
    # Run full validator
    validation_result = validate_extraction(
        extraction=mock_extraction, # type: ignore
        article_body=body,
        whitelist=whitelist, # from section 4
        config=MASConfig().validator,
        embed_fn=mock_embed_fn
    )
    
    print("\nValidation Result:")
    print(pd.Series(validation_result))

except Exception as e:
    print(f"Skipping Validator execution: {e}")
```

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/notebooks/test_mas_interactive_qa.py::test_notebook_has_validator_section -v`
Expected: PASS

- [ ] **Step 5: Syntax and Linting**
Run: `uv run ruff check src/trading_databricks/notebooks/`

- [ ] **Step 6: Commit**
```bash
git add tests/notebooks/test_mas_interactive_qa.py src/trading_databricks/notebooks/mas_interactive_qa.py
git commit -m "feat: add Mechanism Validator debugging section to QA notebook"
```

---

### Task 6: E2E MLflow Evaluation (Section 6)

**Files:**
- Modify: `tests/notebooks/test_mas_interactive_qa.py:61-70`
- Modify: `src/trading_databricks/notebooks/mas_interactive_qa.py`

- [ ] **Step 1: Write the failing test**
Append to `tests/notebooks/test_mas_interactive_qa.py`:
```python
def test_notebook_has_e2e_section():
    content = Path("src/trading_databricks/notebooks/mas_interactive_qa.py").read_text()
    assert "Section 6: End-to-End Evaluation via MLflow" in content
    assert "build_mas_graph" in content
    assert "mlflow.start_run" in content
```

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/notebooks/test_mas_interactive_qa.py::test_notebook_has_e2e_section -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**
Append to `src/trading_databricks/notebooks/mas_interactive_qa.py`:
```python
# COMMAND ----------
# MAGIC %md
# MAGIC ## Section 6: End-to-End Evaluation via MLflow
# MAGIC Execute the full compiled LangGraph on a test batch and log traces to MLflow.

# COMMAND ----------
from trading_databricks.agents.graph import build_mas_graph
import time

try:
    # Compile graph
    graph = build_mas_graph(config=MASConfig())
    
    # Run a small E2E batch (e.g., 2 articles) to avoid massive LLM costs in the sandbox
    batch_states = states[:2]
    
    e2e_results = []
    
    with mlflow.start_run(run_name="Interactive_QA_Batch"):
        for state in batch_states:
            print(f"Processing story_id: {state.get('story_id')}...")
            start_time = time.time()
            
            # Invoke Graph
            final_state = graph.invoke(state)
            
            latency = time.time() - start_time
            
            e2e_results.append({
                "story_id": final_state.get("story_id"),
                "headline": final_state.get("headline"),
                "extraction_path": final_state.get("extraction_path"),
                "validation_status": final_state.get("validation_status"),
                "latency_sec": round(latency, 2)
            })
            
    e2e_df = pd.DataFrame(e2e_results)
    print("\nE2E Evaluation Summary:")
    display(e2e_df) # type: ignore
    
    print("\nMetrics:")
    print(e2e_df["validation_status"].value_counts())
    print("\nCheck the MLflow Experiment UI to view the full execution traces for these runs.")

except Exception as e:
    print(f"Skipping E2E execution (requires active dependencies): {e}")
```

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/notebooks/test_mas_interactive_qa.py::test_notebook_has_e2e_section -v`
Expected: PASS

- [ ] **Step 5: Syntax and Linting**
Run: `uv run ruff check src/trading_databricks/notebooks/`

- [ ] **Step 6: Commit**
```bash
git add tests/notebooks/test_mas_interactive_qa.py src/trading_databricks/notebooks/mas_interactive_qa.py
git commit -m "feat: add E2E MLflow evaluation section to QA notebook"
```
