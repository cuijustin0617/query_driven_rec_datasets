import json
from pathlib import Path
from typing import Dict, List

from ..models.llm_client import LLMClient
from ..data.query_processor import read_queries
from ..data.document_processor import get_documents
from .summarizer import DocumentSummarizer
from .relevance_judge import RelevanceJudge
from ..config import QUERIES_PATH, DOCS_DIR, GROUND_TRUTH_PATH

class RelevancePipeline:
    """Pipeline for determining item relevance to queries"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.summarizer = DocumentSummarizer(llm_client)
        self.relevance_judge = RelevanceJudge(llm_client)
    
    def run(self) -> Dict[str, List[str]]:
        """
        Run the pipeline.
        
        Returns:
            Dict[str, List[str]]: Ground truth mapping queries to relevant items
        """
        queries = read_queries(QUERIES_PATH)
        documents = get_documents(DOCS_DIR)

        ground_truth = {}
        for query in queries:
            print(f"Processing query: {query}")

            # step 1: generate summaries
            print("Generating summaries...")
            summaries = self.summarizer.process_query(query, documents)

            # step 2: determine relevance
            print("Determining relevance...")
            relevant_items = self.relevance_judge.determine_relevance(query, summaries)

            # step 3: save relevance judgments
            print("Saving relevance judgments...")
            self.relevance_judge.save_relevance(query, relevant_items)

            # step 4: update ground truth
            print("Updating ground truth...")
            ground_truth[query] = relevant_items

        # Save ground truth
        self.save_ground_truth(ground_truth)
        
        return ground_truth
    
    def save_ground_truth(self, ground_truth: Dict[str, List[str]]) -> None:
        """
        Save ground truth to a JSON file.
        
        Args:
            ground_truth: Ground truth mapping queries to relevant items
        """
        with open(GROUND_TRUTH_PATH, 'w', encoding='utf-8') as f:
            json.dump(ground_truth, f, indent=2)
