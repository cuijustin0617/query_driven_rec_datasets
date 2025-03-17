"""Main pipeline for passage-based relevance judgments."""
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

from ..models.llm_client import LLMClient
from ..data.data_loader import read_dense_results
from .passage_processor import PassageProcessor
from ..config import DENSE_RESULTS_PATH, GROUND_TRUTH_PATH, OUTPUT_DIR, QUERY_START_INDEX, QUERY_END_INDEX

class PassageRelevancePipeline:
    """Pipeline for determining document relevance based on passage-level judgments."""

    def __init__(self, llm_client: LLMClient):
        """
        Initialize the pipeline.
        
        Args:
            llm_client: LLM client for API calls
        """
        self.llm_client = llm_client
        self.passage_processor = PassageProcessor(llm_client)
    
    def run(self) -> Dict[str, List[str]]:
        """
        Run the pipeline.
        
        Returns:
            Dict[str, List[str]]: Ground truth mapping queries to ranked relevant documents
        """
        # Read dense retrieval results
        dense_results = read_dense_results(DENSE_RESULTS_PATH)
        
        # Filter queries based on the configured range
        all_queries = list(dense_results.keys())
        
        # Ensure indices are within bounds
        start_idx = max(0, min(QUERY_START_INDEX, len(all_queries) - 1))
        end_idx = max(start_idx, min(QUERY_END_INDEX, len(all_queries) - 1))
        
        selected_queries = all_queries[start_idx:end_idx + 1]
        
        # Save selected queries to a file
        queries_output_path = Path(OUTPUT_DIR) / "queries.txt"
        with open(queries_output_path, 'w', encoding='utf-8') as f:
            for query in selected_queries:
                f.write(f"{query}\n")
        
        print(f"Processing {len(selected_queries)} queries (from index {start_idx} to {end_idx})")
        print(f"Selected queries saved to {queries_output_path}")
        
        ground_truth = {}
        
        # Process each selected query
        for query in selected_queries:
            print(f"Processing query: {query}")
            
            # Get document scores based on passage relevance
            document_scores = self.passage_processor.process_query(query, dense_results[query])
            
            # Rank documents by score (higher score = more relevant)
            ranked_documents = self._rank_documents(document_scores)
            
            # Save to ground truth
            ground_truth[query] = ranked_documents
        
        # Save ground truth
        self.save_ground_truth(ground_truth)
        
        return ground_truth
    
    def _rank_documents(self, document_scores: Dict[str, float]) -> List[str]:
        """
        Rank documents by their relevance scores.
        
        Args:
            document_scores: Dictionary mapping document IDs to relevance scores
            
        Returns:
            List[str]: List of document IDs ordered by relevance (highest first)
        """
        # Sort documents by score (descending)
        ranked_documents = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Extract just the document IDs
        return [doc_id for doc_id, _ in ranked_documents if _ > 0]
    
    def save_ground_truth(self, ground_truth: Dict[str, List[str]]) -> None:
        """
        Save ground truth to a JSON file.
        
        Args:
            ground_truth: Ground truth mapping queries to ranked relevant documents
        """
        # Create output directory if it doesn't exist
        Path(GROUND_TRUTH_PATH).parent.mkdir(parents=True, exist_ok=True)
        
        with open(GROUND_TRUTH_PATH, 'w', encoding='utf-8') as f:
            json.dump(ground_truth, f, indent=2)
