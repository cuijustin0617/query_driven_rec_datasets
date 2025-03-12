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
    # Try UTF-8 first
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, try with 'latin-1' or 'cp1252' which can handle most Windows characters
        try:
            with open(doc_path, 'r', encoding='cp1252') as f:
                return f.read()
        except UnicodeDecodeError:
            # Last resort - latin-1 can read any byte values
            with open(doc_path, 'r', encoding='latin-1') as f:
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
    
    # Get all files in the directory, regardless of extension
    doc_paths = [p for p in path.iterdir() if p.is_file()]
    
    if not doc_paths:
        raise FileNotFoundError(f"No documents found in {docs_dir}")
    return [(doc_path.stem, doc_path) for doc_path in doc_paths]
