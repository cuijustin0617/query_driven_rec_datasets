"""Determine relevance based on summaries."""
import json
from pathlib import Path
from typing import Dict, List

from ..models.llm_client import LLMClient
from ..prompts.relevance_prompt import get_relevance_prompt
from ..config import RELEVANCE_DIR

class RelevanceJudge:
    """Judge relevance of restaurants to queries based on summaries."""
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize relevance judge.
        
        Args:
            llm_client: LLM client for API calls
        """
        self.llm_client = llm_client
        # Create relevance directory if it doesn't exist
        Path(RELEVANCE_DIR).mkdir(parents=True, exist_ok=True)
    
    def determine_relevance(self, query: str, summaries: Dict[str, str]) -> List[str]:
        """
        Determine which restaurants are relevant to the query.
        
        Args:
            query: The query to judge relevance for
            summaries: Dictionary mapping restaurant names to summaries
            
        Returns:
            List[str]: List of relevant restaurant names
        """
        # Format restaurant summaries for the prompt
        restaurant_summaries = ""
        for restaurant, summary in summaries.items():
            restaurant_summaries += f"RESTAURANT: {restaurant}\n"
            restaurant_summaries += f"SUMMARY: {summary}\n\n"
        
        prompt = get_relevance_prompt(query, restaurant_summaries)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that determines restaurant relevance."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm_client.get_completion(messages)
        print(f"relevance of {query}: {response}")
        
        # Parse the JSON response
        try:
            # Clean up the response to handle potential formatting issues
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            
            relevant_restaurants = json.loads(response)
            
            # Ensure it's a list
            if not isinstance(relevant_restaurants, list):
                raise ValueError("Response is not a list")
            
            return relevant_restaurants
        except json.JSONDecodeError:
            # Fallback: try to extract a list from the text
            try:
                # Look for something that looks like a JSON array
                if "[" in response and "]" in response:
                    list_text = response[response.find("["):response.rfind("]")+1]
                    return json.loads(list_text)
                else:
                    # Last resort: try to parse a comma-separated list
                    restaurants = [r.strip().strip('"\'') for r in response.split(",")]
                    return [r for r in restaurants if r]
            except:
                print(f"Failed to parse response: {response}")
                return []
    
    def save_relevance(self, query: str, relevant_restaurants: List[str]) -> None:
        """
        Save relevance judgments for a query.
        
        Args:
            query: The query
            relevant_restaurants: List of relevant restaurant names
        """
        query_id = query.replace(" ", "_")[:50]  # Create a safe filename
        output_path = Path(RELEVANCE_DIR) / f"{query_id}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "query": query,
                "relevant_restaurants": relevant_restaurants
            }, f, indent=2)

        
        
