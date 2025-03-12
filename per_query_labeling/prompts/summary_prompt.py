"""Prompt template for document summarization."""
from per_query_labeling.config import DOMAIN

def get_summary_prompt(document_content: str, query: str) -> str:
    """
    Get the prompt for summarizing a document with respect to a query.
    
    Args:
        document_content: The content of the {DOMAIN} document (reviews)
        query: The query to summarize the document for
        
    Returns:
        str: The formatted prompt
    """
    return f"""You are an expert at analyzing {DOMAIN} reviews and extracting key information.

{DOMAIN.upper()} REVIEWS:
{document_content}

QUERY:
{query}

TASK:
Generate a concise 1-2 sentence summary of this {DOMAIN} that extracts information relevant to the query. 
Focus on factual details, or any specific 
characteristics mentioned in the reviews.

Your summary should extract only objective information from the document without making judgments about 
relevance to the query. Don't include general praise or critique unless it contains specific details that 
might be related to the query.

SUMMARY:"""
