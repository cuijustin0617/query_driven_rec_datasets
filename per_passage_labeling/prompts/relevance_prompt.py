"""Prompt template for passage relevance scoring."""
from ..config import DOMAIN
from typing import List


def get_passage_relevance_prompt(query: str, passages: List[str]) -> str:
    """
    Get the prompt for scoring the relevance of passages to a query.
    
    Args:
        query: The search query
        passages: List of passages to evaluate
        
    Returns:
        str: The formatted prompt
    """
    passages_text = ""
    for i, passage in enumerate(passages, 1):
        passages_text += f"PASSAGE {i}:\n{passage}\n\n"

    return f"""You are an expert {DOMAIN} recommendation system that carefully analyzes {DOMAIN} information.

QUERY:
{query}

I will provide you with {len(passages)} passages from {DOMAIN} descriptions or reviews. Please evaluate the relevance of each passage to the query.

{passages_text}

TASK:
Rate each passage's relevance to the query on a scale of 0-3:

0 = The passage is not relevant to the query's needs or requirements
1 = The passage shows some relation to the query but doesn't address the specific needs well
2 = The passage partially addresses the query's needs with some relevant features
3 = The passage directly addresses the query's needs with highly relevant features

For each passage, consider:
1. How directly the passage addresses the specific requirements in the query
2. How many of the query's key requirements are met in the passage
3. Whether the passage offers unique features that specifically match the query intent

Return your answer as a JSON object with passage numbers as keys and relevance scores (integers 0-3) as values.

Example response format:
{{
  "1": 3,
  "2": 2,
  "3": 1,
  "4": 0,
  "5": 2
}}

RELEVANCE SCORES:"""
