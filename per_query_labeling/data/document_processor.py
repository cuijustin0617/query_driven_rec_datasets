"""Process document files."""
from pathlib import Path
from typing import Dict, List, Tuple

def read_document(doc_path: Path) -> str:
    """
    Read the content of a document.
    
    Args:
        doc_path: Path to the document
        
    Returns:
        str: Document content
    """
    with open(doc_path, 'r', encoding='utf-8') as f:
        return f.read()

def get_documents(docs_dir: str) -> List[Tuple[str, Path]]:
    """
    Get all documents in the directory.
    
    Args:
        docs_dir: Path to the documents directory
        
    Returns:
        List[Tuple[str, Path]]: List of (restaurant_name, document_path) tuples
    """
    path = Path(docs_dir)
    if not path.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
    
    doc_paths = list(path.glob('*.txt'))
    return [(doc_path.stem, doc_path) for doc_path in doc_paths]
