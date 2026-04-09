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

def test_notebook_has_gatekeeper_section():
    content = Path("src/trading_databricks/notebooks/mas_interactive_qa.py").read_text()
    assert "Section 2: Gatekeeper (Stage 1) Deep Dive" in content
    assert "is_crypto_relevant" in content
    assert "extract_ner_features" in content

def test_notebook_has_routing_section():
    content = Path("src/trading_databricks/notebooks/mas_interactive_qa.py").read_text()
    assert "Section 3: RAG & Routing Inspection" in content
    assert "few_shot_retriever" in content
    assert "uncertainty_router" in content
