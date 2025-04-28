#!/usr/bin/env python3
import pandas as pd
import json
import os
from collections import defaultdict, Counter

# Path definitions
nyc_dir = "per_pair_labeling/datasets/hotel_apr15/nyc"   #### TO CHANGE ####
montreal_dir = "per_pair_labeling/datasets/hotel_apr15/montreal"  #### TO CHANGE ####
london_dir = "per_pair_labeling/datasets/hotel_apr15/london"  #### TO CHANGE ####
chicago_dir = "per_pair_labeling/datasets/hotel_apr15/chicago"  #### TO CHANGE ####

# Get CSV file paths
# NYC and London have three parts each
nyc_csv_part1 = os.path.join(nyc_dir, f"gemini_labels_apr15_part1.csv")
nyc_csv_part2 = os.path.join(nyc_dir, f"gemini_labels_apr15_part2.csv")
nyc_csv_part3 = os.path.join(nyc_dir, f"gemini_labels_apr15_part3.csv")
# Montreal and Chicago have only one part
montreal_csv = os.path.join(montreal_dir, f"gemini_labels_apr15.csv")
# London has three parts
london_csv_part1 = os.path.join(london_dir, f"gemini_labels_apr15_part1.csv")
london_csv_part2 = os.path.join(london_dir, f"gemini_labels_apr15_part2.csv")
london_csv_part3 = os.path.join(london_dir, f"gemini_labels_apr15_part3.csv")
chicago_csv = os.path.join(chicago_dir, f"gemini_labels_apr15.csv")

# Thresholds
MIN_THRESHOLD = 0.005  # 1.5%  #### TO CHANGE ####
MAX_THRESHOLD = 0.33    # 30%  #### TO CHANGE ####

# Required row counts for each city
NYC_REQUIRED_ROWS = 260  #### TO CHANGE ####
MONTREAL_REQUIRED_ROWS = 64  #### TO CHANGE ####
LONDON_REQUIRED_ROWS = 266  #### TO CHANGE ####
CHICAGO_REQUIRED_ROWS = 74  #### TO CHANGE ####

def process_city_data(csv_files, city_name, required_row_count):
    """Process data for one city and return query statistics"""
    # Load data for the city - handling multiple files if necessary
    if isinstance(csv_files, list):
        # Load and concatenate multiple dataframes
        dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        # Single file case
        df = pd.read_csv(csv_files)
        
    print(f"Loaded {city_name}: {len(df)} rows")
    
    # Count total rows and relevance=3 rows per query
    query_stats = {}
    for query in df['Query'].unique():
        query_rows = df[df['Query'] == query]
        total_rows = len(query_rows)
        rel3_rows = len(query_rows[query_rows['Relevance Score'] == 3])
        rel3_percentage = rel3_rows / total_rows if total_rows > 0 else 0
        
        has_required_rows = total_rows == required_row_count
        rel3_in_range = MIN_THRESHOLD <= rel3_percentage <= MAX_THRESHOLD
        
        query_stats[query] = {
            'total_rows': total_rows,
            'rel3_rows': rel3_rows,
            'rel3_percentage': rel3_percentage,
            'has_required_rows': has_required_rows,
            'rel3_in_range': rel3_in_range,
            'qualified': has_required_rows and rel3_in_range
        }
    
    # Print detailed stats
    total_queries = len(query_stats)
    total_qualified = sum(1 for stats in query_stats.values() if stats['qualified'])
    wrong_row_count = sum(1 for stats in query_stats.values() if not stats['has_required_rows'])
    rel3_out_of_range = sum(1 for stats in query_stats.values() if not stats['rel3_in_range'])
    
    print(f"Total queries for {city_name}: {total_queries}")
    print(f"  - {total_qualified} qualified queries")
    print(f"  - {wrong_row_count} queries disqualified due to wrong row count (expected {required_row_count})")
    print(f"  - {rel3_out_of_range} queries disqualified due to relevance score=3 percentage out of range")
    
    return query_stats, df

# Process all cities
print("Processing NYC data...")
nyc_stats, nyc_df = process_city_data([nyc_csv_part1, nyc_csv_part2, nyc_csv_part3], "NYC", NYC_REQUIRED_ROWS)
print("\nProcessing Montreal data...")
montreal_stats, montreal_df = process_city_data(montreal_csv, "Montreal", MONTREAL_REQUIRED_ROWS)
print("\nProcessing London data...")
london_stats, london_df = process_city_data([london_csv_part1, london_csv_part2, london_csv_part3], "London", LONDON_REQUIRED_ROWS)
print("\nProcessing Chicago data...")
chicago_stats, chicago_df = process_city_data(chicago_csv, "Chicago", CHICAGO_REQUIRED_ROWS)

# Find overlapping qualified queries across all cities
nyc_qualified = {q for q, stats in nyc_stats.items() if stats['qualified']}
montreal_qualified = {q for q, stats in montreal_stats.items() if stats['qualified']}
london_qualified = {q for q, stats in london_stats.items() if stats['qualified']}
chicago_qualified = {q for q, stats in chicago_stats.items() if stats['qualified']}

overlapping_qualified = nyc_qualified.intersection(montreal_qualified, london_qualified, chicago_qualified)

print(f"\nOverlapping qualified queries across all cities: {len(overlapping_qualified)}")

# Create ground truth files for each city
def create_ground_truth(df, qualified_queries, output_file):
    ground_truth = {}
    
    for query in qualified_queries:
        # Get all hotels with relevance = 3 for this query
        rel3_hotels = df[(df['Query'] == query) & (df['Relevance Score'] == 3)]['Hotel'].tolist()
        ground_truth[query] = rel3_hotels
    
    with open(output_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    return ground_truth

# Create ground truth files
nyc_gt = create_ground_truth(nyc_df, overlapping_qualified, os.path.join(nyc_dir, "ground_truth.json"))    #### TO CHANGE ####
montreal_gt = create_ground_truth(montreal_df, overlapping_qualified, os.path.join(montreal_dir, "ground_truth.json"))    #### TO CHANGE ####   
london_gt = create_ground_truth(london_df, overlapping_qualified, os.path.join(london_dir, "ground_truth.json"))    #### TO CHANGE ####
chicago_gt = create_ground_truth(chicago_df, overlapping_qualified, os.path.join(chicago_dir, "ground_truth.json"))    #### TO CHANGE ####

print(f"\nCreated ground truth files:")
print(f"NYC: {len(nyc_gt)} queries in ground_truth.json")
print(f"Montreal: {len(montreal_gt)} queries in ground_truth.json")
print(f"London: {len(london_gt)} queries in ground_truth.json")
print(f"Chicago: {len(chicago_gt)} queries in ground_truth.json")

# Print overall statistics
print("\nOverall statistics:")
print(f"NYC total rows with relevance=3: {sum(stats['rel3_rows'] for stats in nyc_stats.values())}")
print(f"Montreal total rows with relevance=3: {sum(stats['rel3_rows'] for stats in montreal_stats.values())}")
print(f"London total rows with relevance=3: {sum(stats['rel3_rows'] for stats in london_stats.values())}")
print(f"Chicago total rows with relevance=3: {sum(stats['rel3_rows'] for stats in chicago_stats.values())}")
