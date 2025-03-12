import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from ..models.llm_client import LLMClient
from ..data.document_processor import read_document
from ..prompts.summary_prompt import get_summary_prompt
from ..config import SUMMARIES_DIR, DOMAIN

class DocumentSummarizer:
    """ Summarize documents with respect to queries"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        Path(SUMMARIES_DIR).mkdir(parents=True, exist_ok=True)

    def summarize_document(self, document_path: Path, query: str) -> str:
        """Summarize a document with respect to a query."""
        document_content = read_document(document_path)
        prompt = get_summary_prompt(document_content, query)

        messages = [
            {"role": "system", "content": f"You are a helpful assistant that summarizes {DOMAIN} information."},
            {"role": "user", "content": prompt},
        ]
        response = self.llm_client.get_completion(messages)
        print(f"summary of {document_path}: {response}")
        return response
    
    def process_query(self, query: str, documents: List[Tuple[str, Path]]) -> Dict[str, str]:
        """
        Process a query against all documents.
        
        Args:
            query: The query to process
            documents: List of (item name, document_path) tuples
            
        Returns:
            Dict[str, str]: Dictionary mapping item names to summaries
        """
        query_id = query.replace(" ", "_")[:50]
        output_path = Path(SUMMARIES_DIR) / f"{query_id}.csv"

        #check if already processed
        if output_path.exists():
            print(f"Skipping query: {query} because it ALREADY exists")
            #load existing summaries
            summaries = {}
            with open(output_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader) #skip header
                for row in reader:
                    if len(row) >= 2:
                        summaries[row[0]] = row[1]
            return summaries
        
        summaries = {}
        print(f"Processing query: {query}")

        with tqdm(total=len(documents), desc="Generating summaries") as pbar:

            for item_name, doc_path in documents:
                summary = self.summarize_document(doc_path, query)
                summaries[item_name] = summary
                pbar.update(1)
        
        #save summaries
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([DOMAIN.capitalize(), "Summary"])
            for item_name, summary in sorted(summaries.items()):
                writer.writerow([item_name, summary])

        return summaries
                

    
    
