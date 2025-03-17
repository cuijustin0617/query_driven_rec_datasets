"""Process passages and determine their relevance to queries."""
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from tqdm import tqdm
import re

from ..models.llm_client import LLMClient
from ..prompts.relevance_prompt import get_passage_relevance_prompt
from ..config import PASSAGES_PER_BATCH, RELEVANCE_DIR, DOMAIN

class PassageProcessor:
    """Process passages and determine their relevance to queries."""
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize passage processor.
        
        Args:
            llm_client: LLM client for API calls
        """
        self.llm_client = llm_client
        # Create relevance directory if it doesn't exist
        Path(RELEVANCE_DIR).mkdir(parents=True, exist_ok=True)
    
    def process_query(self, query: str, doc_passages: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Process a query against all documents and their passages.
        
        Args:
            query: The query to process
            doc_passages: Dictionary mapping document IDs to lists of passages
            
        Returns:
            Dict[str, float]: Dictionary mapping document IDs to relevance scores
        """
        # Check if relevance judgments already exist for this query
        query_id = self._get_query_id(query)
        output_path = Path(RELEVANCE_DIR) / f"{query_id}.json"
        
        if output_path.exists():
            print(f"Relevance file for '{query}' already exists. Loading saved data.")
            with open(output_path, 'r', encoding='utf-8') as f:
                relevance_data = json.load(f)
            return relevance_data.get("document_scores", {})
        
        # Process each document
        document_scores = {}
        
        # Add progress bar for document processing
        print(f"Processing {len(doc_passages)} documents for query: '{query}'")
        for doc_id, passages in tqdm(doc_passages.items(), desc="Processing documents", unit="doc"):
            doc_score = self._process_document_passages(query, doc_id, passages)
            document_scores[doc_id] = doc_score
            # Show intermediate results
            print(f"Document '{doc_id}' score: {doc_score:.2f}")
        
        # Save the results
        relevance_data = {
            "query": query,
            "document_scores": document_scores
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(relevance_data, f, indent=2)
        
        return document_scores
    
    def _process_document_passages(self, query: str, doc_id: str, passages: List[str]) -> float:
        """
        Process passages from a document and compute the document's relevance score.
        
        Args:
            query: The query to judge relevance for
            doc_id: Document identifier
            passages: List of passages from the document
            
        Returns:
            float: Average relevance score for the document
        """
        # If there are no passages, return 0
        if not passages:
            return 0.0
        
        # Randomly shuffle passages to avoid batch bias
        random.shuffle(passages)
        
        # Process passages in batches
        passage_scores = []
        
        # Process all passages in batches of PASSAGES_PER_BATCH
        num_batches = (len(passages) + PASSAGES_PER_BATCH - 1) // PASSAGES_PER_BATCH
        for i in tqdm(range(0, len(passages), PASSAGES_PER_BATCH), 
                     desc=f"Evaluating passages for {doc_id}", 
                     unit="batch", 
                     total=num_batches,
                     leave=False):
            batch = passages[i:i + PASSAGES_PER_BATCH]
            batch_scores = self._evaluate_passage_batch(query, batch)
            passage_scores.extend(batch_scores)
        
        # Average the passage scores to get the document score
        if not passage_scores:
            return 0.0
        
        return sum(passage_scores) / len(passage_scores)
    
    def _evaluate_passage_batch(self, query: str, passages: List[str]) -> List[float]:
        """
        Evaluate a batch of passages for relevance to the query.
        
        Args:
            query: The query to judge relevance for
            passages: List of passages to evaluate
            
        Returns:
            List[float]: List of relevance scores for each passage
        """
        prompt = get_passage_relevance_prompt(query, passages)
        
        messages = [
            {"role": "system", "content": f"You are a helpful assistant that determines {DOMAIN} relevance."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm_client.get_completion(messages)
        
        # Parse the JSON response
        scores = self._parse_response(response, len(passages), query, passages)
        
        return scores
    
    def _parse_response(self, response: str, expected_count: int, query: str, passages: List[str]) -> List[float]:
        """
        Parse the LLM response to extract relevance scores.
        
        Args:
            response: The LLM response text
            expected_count: Expected number of passage scores
            query: The query being processed
            passages: The passages being scored
            
        Returns:
            List[float]: List of relevance scores
        """
        try:
            # Clean up the response to handle potential formatting issues
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            
            scores_dict = json.loads(response)
            
            # Ensure it's a dictionary
            if not isinstance(scores_dict, dict):
                raise ValueError("Response is not a dictionary")
            
            # Convert to list of scores preserving order
            scores = []
            
            # Print detailed results for each passage
            print("\n=== PASSAGE SCORING DETAILS ===")
            print(f"QUERY: {query}")
            print("-" * 80)
            
            for i in range(1, expected_count + 1):
                score = scores_dict.get(str(i))
                if score is None:
                    print(f"Warning: Missing score for passage {i}")
                    score = 0
                
                scores.append(float(score))
                
                # Get the passage snippet (first 100 chars)
                if i-1 < len(passages):
                    passage_snippet = passages[i-1][:100] + "..." if len(passages[i-1]) > 100 else passages[i-1]
                    print(f"PASSAGE {i} (Score: {score}):\n{passage_snippet}\n")
            
            print("-" * 80)
            return scores
            
        except json.JSONDecodeError:
            # Fallback: try to extract a dictionary from the text
            try:
                # Look for something that looks like a JSON object
                if "{" in response and "}" in response:
                    dict_text = response[response.find("{"):response.rfind("}")+1]
                    scores_dict = json.loads(dict_text)
                    
                    # Print detailed results for each passage
                    print("\n=== PASSAGE SCORING DETAILS (FALLBACK PARSING) ===")
                    print(f"QUERY: {query}")
                    print("-" * 80)
                    
                    # Convert to list of scores preserving order
                    scores = []
                    for i in range(1, expected_count + 1):
                        score = scores_dict.get(str(i))
                        if score is None:
                            print(f"Warning: Missing score for passage {i}")
                            score = 0
                        
                        scores.append(float(score))
                        
                        # Get the passage snippet (first 100 chars)
                        if i-1 < len(passages):
                            passage_snippet = passages[i-1][:100] + "..." if len(passages[i-1]) > 100 else passages[i-1]
                            print(f"PASSAGE {i} (Score: {score}):\n{passage_snippet}\n")
                    
                    print("-" * 80)
                    return scores
                else:
                    print(f"Failed to parse response as JSON object: {response}")
                    return [0.0] * expected_count
            except:
                print(f"Failed to parse response: {response}")
                return [0.0] * expected_count
    
    def _get_query_id(self, query: str) -> str:
        """
        Create a safe filename from a query using only alphanumeric and underscore characters.
        
        Args:
            query: The query string
            
        Returns:
            str: A filename-safe representation of the query
        """
        # Replace all non-alphanumeric characters with underscores
        safe_id = re.sub(r'[^a-zA-Z0-9]', '_', query)
        # Truncate to prevent excessively long filenames
        return safe_id[:50]
