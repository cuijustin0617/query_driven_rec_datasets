"""
Script for generating binary relevance labels using configured LLM.
"""
import os
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
import argparse

from .config import LabelingConfig, get_default_config
from .prompts import get_binary_umbrella_prompt
from .llm_clients import get_llm_client, parse_binary_response

def load_queries(query_file: str) -> List[str]:
    """Load queries from a text file."""
    print(f"Loading queries from {query_file}")
    with open(query_file, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(queries)} queries")
    return queries

def load_corpus_documents(corpus_dir: str) -> Dict[str, str]:
    """
    Load documents from the corpus directory.
    Each document is a text file with the entity name as the filename.
    """
    documents = {}
    print(f"Loading corpus documents from {corpus_dir}")
    
    for file_path in Path(corpus_dir).glob("*.txt"):
        entity_name = file_path.stem  # Get filename without extension
        with open(file_path, 'r', encoding='utf-8') as f:
            document_text = f.read().strip()
        documents[entity_name] = document_text
    
    print(f"Loaded {len(documents)} documents from corpus")
    return documents

def load_existing_results(output_path: str) -> Dict[str, Dict[str, bool]]:
    """
    Load existing results from the output CSV file.
    
    Returns:
        Dict mapping query -> {entity -> relevance}
    """
    existing_results = {}
    
    if os.path.exists(output_path):
        print(f"Loading existing results from {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip header
            
            if header:  # Make sure file isn't empty
                for row in reader:
                    if len(row) >= 3:
                        query, entity, relevance_str = row[0], row[1], row[2]
                        # Convert relevance string to boolean
                        relevance = relevance_str.lower() == 'true'
                        
                        if query not in existing_results:
                            existing_results[query] = {}
                        existing_results[query][entity] = relevance
        
        print(f"Loaded {sum(len(entities) for entities in existing_results.values())} existing labeled pairs")
    
    return existing_results

def save_results(output_path: str, results: List[Tuple[str, str, bool]], config: LabelingConfig) -> None:
    """
    Save results to CSV file.
    
    Args:
        output_path: Path to the output CSV file
        results: List of (query, entity, relevance) tuples
        config: LabelingConfig instance
    """
    # Check if file exists to decide whether to write header
    file_exists = os.path.exists(output_path)
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    mode = 'a' if file_exists else 'w'
    with open(output_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(config.get_csv_headers())
        
        # Convert boolean relevance to string
        formatted_results = [(q, e, str(r)) for q, e, r in results]
        writer.writerows(formatted_results)
    
    print(f"Saved {len(results)} results to {output_path}")

def generate_label(query: str, entity: str, document: str, llm_client: Any, domain_mappings: Dict[str, str]) -> Optional[bool]:
    """
    Generate a relevance label for a query-entity pair using the configured LLM.
    
    Args:
        query: The search query
        entity: The entity name (e.g., city, restaurant, hotel)
        document: The document text for the entity
        llm_client: The LLM client to use
        domain_mappings: Domain-specific terminology mappings
        
    Returns:
        Boolean relevance or None if generation failed
    """
    prompt = get_binary_umbrella_prompt(domain_mappings)
    formatted_prompt = prompt.format(
        query=query,
        entity_name=entity,
        document=document
    )
    
    messages = [
        {"role": "system", "content": f"You are a helpful assistant that evaluates relevance for {domain_mappings['singular']}-related query-document pairs."},
        {"role": "user", "content": formatted_prompt},
    ]
    
    response_text = llm_client.generate(messages, temperature=0.0)
    if response_text is None:
        print(f"Failed to get response for {entity}")
        return None
    
    relevance = parse_binary_response(response_text)
    if relevance is None:
        print(f"Failed to parse relevance from response: {response_text}")
        return None
    
    return relevance

def convert_csv_to_ground_truth_json(csv_path: str, json_path: str) -> None:
    """
    Convert the CSV file with labeling results to a ground truth JSON file.
    
    Args:
        csv_path: Path to the CSV file
        json_path: Path to output the JSON file
    """
    print(f"Converting CSV results from {csv_path} to ground truth JSON at {json_path}")
    
    # Initialize ground truth dictionary
    ground_truth = {}
    
    # Read CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 3:
                query, entity, relevance_str = row[0], row[1], row[2]
                # Only include relevant (True) items
                if relevance_str.lower() == 'true':
                    if query not in ground_truth:
                        ground_truth[query] = []
                    ground_truth[query].append(entity)
    
    # Write JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"Successfully created ground truth JSON with {len(ground_truth)} queries")

def is_labeling_complete(all_queries: List[str], corpus: Dict[str, str], existing_results: Dict[str, Dict[str, bool]]) -> bool:
    """
    Check if the labeling process is complete.
    
    Args:
        all_queries: List of all queries
        corpus: Dictionary of corpus documents
        existing_results: Dictionary of existing results
        
    Returns:
        True if all query-entity pairs have been labeled, False otherwise
    """
    expected_pairs_count = len(all_queries) * len(corpus)
    actual_pairs_count = sum(len(entities) for entities in existing_results.values())
    
    return expected_pairs_count == actual_pairs_count

def main(config_override: Optional[Dict[str, Any]] = None):
    """
    Main function for generating labels.
    
    Args:
        config_override: Optional dictionary to override default config values
    """
    # Load configuration
    config = get_default_config()
    
    # Apply overrides if provided
    if config_override:
        for key, value in config_override.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Initialize LLM client
    llm_client = get_llm_client(config)
    
    # Load queries
    queries = load_queries(config.queries_file)
    
    # Load corpus documents
    corpus = load_corpus_documents(config.corpus_dir)
    
    # Load existing results
    existing_results = load_existing_results(config.output_path)
    
    # Process queries and entities
    new_results = []
    save_frequency = 10  # Save after every 10 processed items
    processed_count = 0
    
    for query_idx, query in enumerate(queries):
        print(f"\nProcessing query {query_idx+1}/{len(queries)}: {query}")
        
        for entity_idx, (entity, document) in enumerate(corpus.items()):
            # Skip if already processed
            if query in existing_results and entity in existing_results[query]:
                print(f"  - Skipping already processed {config.domain_mappings['singular']}: {entity}")
                continue
            
            print(f"  - Processing {config.domain_mappings['singular']}: {entity}")
            
            # Generate label
            relevance = generate_label(
                query, 
                entity, 
                document, 
                llm_client, 
                config.domain_mappings
            )
            
            if relevance is not None:
                print(f"    Relevance: {relevance}")
                new_results.append((query, entity, relevance))
                processed_count += 1
                
                # Save results periodically
                if processed_count % save_frequency == 0:
                    save_results(config.output_path, new_results, config)
                    new_results = []  # Clear after saving
    
    # Save any remaining results
    if new_results:
        save_results(config.output_path, new_results, config)
    
    print(f"\nLabeling process complete. Results saved to {config.output_path}")
    
    # Check if labeling is complete
    if is_labeling_complete(queries, corpus, load_existing_results(config.output_path)):
        print("All query-entity pairs have been labeled. Generating ground truth JSON...")
        # Generate ground truth JSON
        output_dir = os.path.dirname(config.output_path)
        json_path = os.path.join(output_dir, "ground_truth.json")
        convert_csv_to_ground_truth_json(config.output_path, json_path)
        print(f"Ground truth JSON generated at {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate binary relevance labels for a corpus using an LLM")
    parser.add_argument("--domain", help="Domain (city, restaurant, hotel)", type=str)
    parser.add_argument("--corpus-dir", help="Path to corpus directory", type=str)
    parser.add_argument("--queries-file", help="Path to queries file", type=str)
    parser.add_argument("--output-dir", help="Output directory", type=str)
    parser.add_argument("--output-filename", help="Output filename", type=str)
    parser.add_argument("--llm-client", help="LLM client to use", type=str)
    parser.add_argument("--model-name", help="Model name for the LLM", type=str)
    
    args = parser.parse_args()
    
    # Build config override dict from args
    config_override = {k: v for k, v in vars(args).items() if v is not None}
    
    # Replace hyphens with underscores in keys
    config_override = {k.replace("-", "_"): v for k, v in config_override.items()}
    
    main(config_override) 