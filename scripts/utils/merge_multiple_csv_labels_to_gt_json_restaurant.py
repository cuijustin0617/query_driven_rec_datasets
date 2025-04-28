#!/usr/bin/env python3
import pandas as pd
import json
import os
from collections import defaultdict, Counter

# Path definitions
phi_dir = "per_pair_labeling/datasets/restaurant_apr15/phi"
new_orl_dir = "per_pair_labeling/datasets/restaurant_apr15/new_orl"

# Get CSV file paths for each part
phi_csv_part1 = os.path.join(phi_dir, "gemini_labels_apr15_part1.csv")
phi_csv_part2 = os.path.join(phi_dir, "gemini_labels_apr15_part2.csv")
phi_csv_part3 = os.path.join(phi_dir, "gemini_labels_apr15_part3.csv")

new_orl_csv_part1 = os.path.join(new_orl_dir, "gemini_labels_apr15_part1.csv")
new_orl_csv_part2 = os.path.join(new_orl_dir, "gemini_labels_apr15_part2.csv")
new_orl_csv_part3 = os.path.join(new_orl_dir, "gemini_labels_apr15_part3.csv")

# Thresholds
MIN_THRESHOLD = 0.005  # 0.5%
MAX_THRESHOLD = 0.35    # 40%

# Required row counts for each city
PHI_REQUIRED_ROWS = 637
NEW_ORL_REQUIRED_ROWS = 515

def process_city_data(csv_files, city_name, required_row_count):
    """Process data for one city and return query statistics"""
    # Load and combine data from all parts
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            print(f"Loaded {os.path.basename(csv_file)}: {len(df)} rows")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    # Combine all parts
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined {city_name}: {len(df)} rows")
    
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
    print(f"Total for {city_name}: {total_qualified} qualified queries")
    print(f"  - {wrong_row_count} queries disqualified due to wrong row count (expected {required_row_count})")
    print(f"  - {rel3_out_of_range} queries disqualified due to relevance score=3 percentage out of range")
    
    return query_stats, df

# Process all cities
print("Processing Philadelphia data...")
phi_stats, phi_df = process_city_data(
    [phi_csv_part1, phi_csv_part2, phi_csv_part3], 
    "Philadelphia", 
    PHI_REQUIRED_ROWS
)

print("\nProcessing New Orleans data...")
new_orl_stats, new_orl_df = process_city_data(
    [new_orl_csv_part1, new_orl_csv_part2, new_orl_csv_part3], 
    "New Orleans", 
    NEW_ORL_REQUIRED_ROWS
)

# Find overlapping qualified queries across all cities
phi_qualified = {q for q, stats in phi_stats.items() if stats['qualified']}
new_orl_qualified = {q for q, stats in new_orl_stats.items() if stats['qualified']}

overlapping_qualified = phi_qualified.intersection(new_orl_qualified)

print(f"\nOverlapping qualified queries across both cities: {len(overlapping_qualified)}")

# Create ground truth files for each city
def create_ground_truth(df, qualified_queries, output_file):
    ground_truth = {}
    
    for query in qualified_queries:
        # Get all restaurants with relevance = 3 for this query
        rel3_restaurants = df[(df['Query'] == query) & (df['Relevance Score'] == 3)]['Restaurant'].tolist()
        ground_truth[query] = rel3_restaurants
    
    with open(output_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    return ground_truth

# Create ground truth files
phi_gt = create_ground_truth(phi_df, overlapping_qualified, os.path.join(phi_dir, "ground_truth.json"))    #### TO CHANGE ####
new_orl_gt = create_ground_truth(new_orl_df, overlapping_qualified, os.path.join(new_orl_dir, "ground_truth.json"))    #### TO CHANGE ####

print(f"\nCreated ground truth files:")
print(f"Philadelphia: {len(phi_gt)} queries in ground_truth.json")
print(f"New Orleans: {len(new_orl_gt)} queries in ground_truth.json")

# Print overall statistics
print("\nOverall statistics:")
print(f"Philadelphia total rows with relevance=3: {sum(stats['rel3_rows'] for stats in phi_stats.values())}")
print(f"New Orleans total rows with relevance=3: {sum(stats['rel3_rows'] for stats in new_orl_stats.values())}")
