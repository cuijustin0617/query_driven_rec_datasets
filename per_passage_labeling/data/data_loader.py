"""Load and process dense retrieval results."""
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

def read_dense_results(file_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Read dense retrieval results from a JSON file.
    
    Args:
        file_path: Path to the dense retrieval results JSON file
        
    Returns:
        Dict mapping queries to documents and their passages
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dense results file not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Organize data by query -> document -> [passages]
    organized_data = {}
    
    for query, doc_data in data.items():
        organized_data[query] = {}
        
        for doc_id, content in doc_data.items():
            # The content is a list where index 0 is a score and index 1 is the list of passages
            if isinstance(content, list) and len(content) >= 2:
                score = content[0]  # We can store this if needed
                passages = content[1]
                
                # Store the passages for this document
                organized_data[query][doc_id] = passages
    
    return organized_data

def get_query_document_passages(dense_results: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Get all queries with their associated documents and passages.
    
    Args:
        dense_results: Dense retrieval results
        
    Returns:
        Dict mapping queries to documents and their passages
    """
    return dense_results
