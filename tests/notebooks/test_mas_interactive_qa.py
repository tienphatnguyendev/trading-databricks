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
