"""Prompt template for document summarization."""

def get_summary_prompt(document_content: str, query: str) -> str:
    """
    Get the prompt for summarizing a document with respect to a query.
    
    Args:
        document_content: The content of the restaurant document (reviews)
        query: The query to summarize the document for
        
    Returns:
        str: The formatted prompt
    """
    return f"""You are an expert at analyzing restaurant reviews and extracting key information.

RESTAURANT REVIEWS:
{document_content}

QUERY:
{query}

TASK:
Generate a concise 1-2 sentence summary of this restaurant specifically addressing how it relates to the query. 
Focus on relevant aspects like cuisine type, ambiance, price range, special features, or any specific requirements 
mentioned in the query.

Your summary should help determine if this restaurant is relevant to the query. Don't include general praise 
or critique unless directly relevant to the query.

SUMMARY:"""
