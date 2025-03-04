"""Process query files."""
from pathlib import Path
from typing import List

def read_queries(queries_path: str) -> List[str]:
    """
    Read queries from a file.
    
    Args:
        queries_path: Path to the queries file
        
    Returns:
        List[str]: List of queries
    """
    path = Path(queries_path)
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    return queries