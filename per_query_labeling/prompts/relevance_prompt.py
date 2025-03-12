"""Prompt template for relevance determination."""
from per_query_labeling.config import DOMAIN

def get_relevance_prompt(query: str, domain_summaries: str) -> str:
    """
    Get the prompt for determining relevant items based on summaries.
    
    Args:
        query: The search query
        domain_summaries: Text containing item names and their summaries
        
    Returns:
        str: The formatted prompt
    """
    return f"""You are an expert {DOMAIN} recommendation system that carefully analyzes {DOMAIN} information.

QUERY:
{query}

{DOMAIN.upper()} SUMMARIES:
{domain_summaries}

TASK:
Based on the above {DOMAIN} summaries, determine the relevance score for each {DOMAIN} to the query on a scale of 0-3:

0 = The {DOMAIN} summary indicates it is not relevant to the query's needs or requirements
1 = The {DOMAIN} summary shows some relation to the query but doesn't address the specific needs well
2 = The {DOMAIN} summary indicates it partially addresses the query's needs with some relevant features
3 = The {DOMAIN} summary shows it directly addresses the query's needs with highly relevant features

For each {DOMAIN}, consider:
1. How directly the summary addresses the specific requirements in the query
2. How many of the query's key requirements are met by the {DOMAIN}
3. Whether the {DOMAIN} offers unique features that specifically match the query intent

Return your answer as a JSON object with {DOMAIN} names as keys and relevance scores (integers 0-3) as values.
Include only {DOMAIN} names and scores, with no additional commentary.

Example response format:
{{
  "{DOMAIN.capitalize()} A": 3,
  "{DOMAIN.capitalize()} B": 2,
  "{DOMAIN.capitalize()} C": 1,
  "{DOMAIN.capitalize()} D": 0
}}

RELEVANCE SCORES:"""