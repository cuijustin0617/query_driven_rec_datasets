#!/usr/bin/env python3
import pandas as pd
import json
import os
from collections import defaultdict, Counter

# Path definitions
nyc_dir = "per_pair_labeling/datasets/hotel_apr10/nyc"   #### TO CHANGE ####
montreal_dir = "per_pair_labeling/datasets/hotel_apr10/montreal"  #### TO CHANGE ####
london_dir = "per_pair_labeling/datasets/hotel_apr10/london"  #### TO CHANGE ####
chicago_dir = "per_pair_labeling/datasets/hotel_apr10/chicago"  #### TO CHANGE ####

# Get CSV file paths
nyc_csv = os.path.join(nyc_dir, f"gemini_labels_apr10.csv")    #### TO CHANGE ####
montreal_csv = os.path.join(montreal_dir, f"gemini_labels_apr10.csv")  #### TO CHANGE ####
london_csv = os.path.join(london_dir, f"gemini_labels_apr10.csv")  #### TO CHANGE ####
chicago_csv = os.path.join(chicago_dir, f"gemini_labels_apr10.csv")  #### TO CHANGE ####

# Thresholds
MIN_THRESHOLD = 0.005  # 1.5%
MAX_THRESHOLD = 0.35    # 30%

# Required row counts for each city
NYC_REQUIRED_ROWS = 260
MONTREAL_REQUIRED_ROWS = 64
LONDON_REQUIRED_ROWS = 266
CHICAGO_REQUIRED_ROWS = 74

def process_city_data(csv_file, city_name, required_row_count):
    """Process data for one city and return query statistics"""
    # Load data for the city
    df = pd.read_csv(csv_file)
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
    total_qualified = sum(1 for stats in query_stats.values() if stats['qualified'])
    wrong_row_count = sum(1 for stats in query_stats.values() if not stats['has_required_rows'])
    rel3_out_of_range = sum(1 for stats in query_stats.values() if not stats['rel3_in_range'])
    
    print(f"Total for {city_name}: {total_qualified} qualified queries")
    print(f"  - {wrong_row_count} queries disqualified due to wrong row count (expected {required_row_count})")
    print(f"  - {rel3_out_of_range} queries disqualified due to relevance score=3 percentage out of range")
    
    return query_stats, df

# Process all cities
print("Processing NYC data...")
nyc_stats, nyc_df = process_city_data(nyc_csv, "NYC", NYC_REQUIRED_ROWS)
print("\nProcessing Montreal data...")
montreal_stats, montreal_df = process_city_data(montreal_csv, "Montreal", MONTREAL_REQUIRED_ROWS)
print("\nProcessing London data...")
london_stats, london_df = process_city_data(london_csv, "London", LONDON_REQUIRED_ROWS)
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
nyc_gt = create_ground_truth(nyc_df, overlapping_qualified, os.path.join(nyc_dir, "ground_truth.json"))
montreal_gt = create_ground_truth(montreal_df, overlapping_qualified, os.path.join(montreal_dir, "ground_truth.json"))
london_gt = create_ground_truth(london_df, overlapping_qualified, os.path.join(london_dir, "ground_truth.json"))
chicago_gt = create_ground_truth(chicago_df, overlapping_qualified, os.path.join(chicago_dir, "ground_truth.json"))

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
