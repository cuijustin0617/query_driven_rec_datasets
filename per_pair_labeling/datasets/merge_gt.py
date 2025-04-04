import csv
import json
import os
from pathlib import Path
from collections import defaultdict

def load_valid_queries(queries_file):
    """
    Load valid queries from a text file.
    
    Args:
        queries_file: Path to the text file containing valid queries
        
    Returns:
        set: Set of valid queries
    """
    valid_queries = set()
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            query = line.strip()
            if query:  # Skip empty lines
                valid_queries.add(query)
    return valid_queries

def process_csv_file(csv_path, ground_truth_dict, valid_queries):
    """
    Process a CSV file and add relevant hotel entries to the ground truth dictionary.
    Only entries with a relevance score of 3 AND queries in valid_queries are considered.
    
    Args:
        csv_path: Path to the CSV file
        ground_truth_dict: Dictionary to store ground truth data
        valid_queries: Set of valid queries to include
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        
        for row in reader:
            if len(row) >= 3:
                query = row[0]
                hotel = row[1]
                relevance_score = row[2]
                
                # Only consider entries with a relevance score of 3 AND query in valid_queries
                if relevance_score == '3' and query in valid_queries:
                    if query not in ground_truth_dict:
                        ground_truth_dict[query] = []
                    
                    # Avoid duplicate entries
                    if hotel not in ground_truth_dict[query]:
                        ground_truth_dict[query].append(hotel)
    
    return ground_truth_dict

def create_ground_truth_json(input_dir, output_file, queries_file):
    """
    Create a ground truth JSON file from CSV files in the input directory.
    Only include queries that exist in the queries_file.
    
    Args:
        input_dir: Directory containing CSV files
        output_file: Path to the output JSON file
        queries_file: Path to file containing valid queries
    """
    # Load valid queries
    valid_queries = load_valid_queries(queries_file)
    print(f"Found {len(valid_queries)} valid queries")
    
    # Initialize the ground truth dictionary
    ground_truth = {}
    
    # Get paths to CSV files
    csv_path1 = os.path.join(input_dir, 'gemini_labels.csv')
    csv_path2 = os.path.join(input_dir, 'gemini_labels_v2.csv')
    
    # Process both CSV files
    process_csv_file(csv_path1, ground_truth, valid_queries)
    process_csv_file(csv_path2, ground_truth, valid_queries)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write the ground truth JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    
    print(f"Ground truth JSON file created at {output_file}")
    print(f"Contains {len(ground_truth)} queries out of {len(valid_queries)} possible queries")
    # Print queries that are missing from ground truth
    missing_queries = set(valid_queries) - set(ground_truth.keys())
    if missing_queries:
        print("\nMissing queries:")
        for query in missing_queries:
            print(f"- {query}")
    else:
        print("\nAll queries from the file are present in the ground truth data")

if __name__ == "__main__":
    # Input directory and output file
    input_directory = "per_pair_labeling/datasets/hotel_nyc"
    output_file = "per_pair_labeling/datasets/hotel_nyc/ground_truth.json"
    queries_file = "queries/queries_hotel_under30.txt"
    
    create_ground_truth_json(input_directory, output_file, queries_file)
