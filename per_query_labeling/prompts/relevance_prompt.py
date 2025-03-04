"""Prompt template for relevance determination."""

def get_relevance_prompt(query: str, restaurant_summaries: str) -> str:
    """
    Get the prompt for determining relevant restaurants based on summaries.
    
    Args:
        query: The search query
        restaurant_summaries: Text containing restaurant names and their summaries
        
    Returns:
        str: The formatted prompt
    """
    return f"""You are an expert restaurant recommendation system that carefully analyzes restaurant information.

QUERY:
{query}

RESTAURANT SUMMARIES:
{restaurant_summaries}

TASK:
Based on the above restaurant summaries, determine which restaurants are relevant to the query. 
A restaurant is relevant if it would be a good match for someone searching with this query.

Return your answer as a JSON array of restaurant names that are relevant to the query.
Include only restaurant names, with no additional commentary.

Example response format:
["Restaurant A", "Restaurant B", "Restaurant C"]

RELEVANT RESTAURANTS:"""