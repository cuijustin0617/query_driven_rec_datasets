"""Determine relevance based on summaries."""
import json
from pathlib import Path
from typing import Dict, List, Tuple

from ..models.llm_client import LLMClient
from ..prompts.relevance_prompt import get_relevance_prompt
from ..config import RELEVANCE_DIR, DOMAIN

class RelevanceJudge:
    """Judge relevance of items to queries based on summaries."""
    
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
        Determine which items are relevant to the query.
        
        Args:
            query: The query to judge relevance for
            summaries: Dictionary mapping item names to summaries
            
        Returns:
            List[str]: List of highly relevant item names (score 3)
        """
        # Check if relevance judgments already exist for this query
        query_id = self._get_query_id(query)
        output_path = Path(RELEVANCE_DIR) / f"{query_id}.json"
        
        if output_path.exists():
            print(f"Relevance file for '{query}' already exists. Loading saved data.")
            relevance_data = self._load_relevance(output_path)
            self._relevance_scores = relevance_data.get("relevance_scores", {})
            return relevance_data.get("relevant_items", [])
        
        # If no saved data exists, proceed with LLM-based relevance judgment
        # Format item summaries for the prompt
        item_summaries = ""
        for item, summary in summaries.items():
            item_summaries += f"{DOMAIN.capitalize()}: {item}\n"
            item_summaries += f"SUMMARY: {summary}\n\n"
        
        prompt = get_relevance_prompt(query, item_summaries)
        
        messages = [
            {"role": "system", "content": f"You are a helpful assistant that determines {DOMAIN} relevance."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm_client.get_completion(messages)
        print(f"relevance of {query}: {response}")
        
        # Parse the JSON response and get both full scores and filtered list
        relevance_scores, relevant_items = self._parse_response(response)
        
        # Store the full scores for later use
        self._relevance_scores = relevance_scores
        
        return relevant_items
    
    def _parse_response(self, response: str) -> Tuple[Dict[str, int], List[str]]:
        """
        Parse the LLM response to extract relevance scores and highly relevant items.
        
        Args:
            response: The LLM response text
            
        Returns:
            Tuple[Dict[str, int], List[str]]: Full relevance scores and list of highly relevant items
        """
        try:
            # Clean up the response to handle potential formatting issues
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            
            relevance_scores = json.loads(response)
            
            # Ensure it's a dictionary
            if not isinstance(relevance_scores, dict):
                raise ValueError("Response is not a dictionary")
            
            # Filter for items with score 3 (highly relevant)
            relevant_items = [item for item, score in relevance_scores.items() if score == 3]
            
            return relevance_scores, relevant_items
            
        except json.JSONDecodeError:
            # Fallback: try to extract a dictionary from the text
            try:
                # Look for something that looks like a JSON object
                if "{" in response and "}" in response:
                    dict_text = response[response.find("{"):response.rfind("}")+1]
                    relevance_scores = json.loads(dict_text)
                    relevant_items = [item for item, score in relevance_scores.items() if score == 3]
                    return relevance_scores, relevant_items
                else:
                    print(f"Failed to parse response as JSON object: {response}")
                    return {}, []
            except:
                print(f"Failed to parse response: {response}")
                return {}, []
    
    def _get_query_id(self, query: str) -> str:
        """
        Create a safe filename from a query.
        
        Args:
            query: The query string
            
        Returns:
            str: A filename-safe representation of the query
        """
        return query.replace(" ", "_")[:50]  # Create a safe filename
    
    def _load_relevance(self, file_path: Path) -> Dict:
        """
        Load relevance data from a JSON file.
        
        Args:
            file_path: Path to the relevance JSON file
            
        Returns:
            Dict: The loaded relevance data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading relevance file {file_path}: {e}")
            return {}
    
    def save_relevance(self, query: str, relevant_items: List[str]) -> None:
        """
        Save relevance judgments for a query.
        
        Args:
            query: The query
            relevant_items: List of relevant item names
        """
        query_id = self._get_query_id(query)  # Use the helper method
        output_path = Path(RELEVANCE_DIR) / f"{query_id}.json"
        
        # Save both the full scores (if available) and the filtered list
        data = {
            "query": query,
            "relevant_items": relevant_items
        }
        
        # Add full scores if available
        if hasattr(self, '_relevance_scores') and self._relevance_scores:
            data["relevance_scores"] = self._relevance_scores
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        
        
