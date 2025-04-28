#!/usr/bin/env python3
import json
import os
import argparse
from typing import Dict, List, Any


def read_queries_from_file(query_path: str) -> List[str]:
    """
    Read queries from a text file, one query per line.
    
    Args:
        query_path: Path to the text file containing queries
        
    Returns:
        List of query strings
    """
    with open(query_path, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    return queries


def sample_gt_from_queries(gt_path: str, query_path: str, output_path: str) -> None:
    """
    Filter ground truth JSON file to only include entries matching queries from the query file.
    
    Args:
        gt_path: Path to the ground truth JSON file
        query_path: Path to the text file containing queries
        output_path: Path where the filtered JSON will be saved
    """
    # Read queries
    queries = read_queries_from_file(query_path)
    print(f"Loaded {len(queries)} queries from {query_path}")
    
    # Read ground truth JSON
    with open(gt_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    print(f"Loaded ground truth with {len(ground_truth)} entries from {gt_path}")
    
    # Filter ground truth to only include entries with keys matching the queries
    filtered_gt = {k: v for k, v in ground_truth.items() if k in queries}
    
    print(f"Filtered ground truth to {len(filtered_gt)} entries")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save filtered ground truth to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_gt, f, indent=2, ensure_ascii=False)
    
    print(f"Saved filtered ground truth to {output_path}")


def main():
    # Define paths as variables that can be easily changed
    gt_path = "ground_truth/restaurant/phi/ground_truth_apr15_207.json"  #### TO CHANGE ####
    query_path = "FINAL_RESULTS_APR15/restaurant/restaurant_apr15_100.txt"  #### TO CHANGE ####
    output_path = "FINAL_RESULTS_APR15/restaurant/gt_phi_100.json"  #### TO CHANGE ####
    
    # Call the function with the defined paths
    sample_gt_from_queries(gt_path, query_path, output_path)


if __name__ == "__main__":
    main()
