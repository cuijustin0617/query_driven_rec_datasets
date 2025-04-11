import csv
import sys
from collections import defaultdict

def analyze_query_ratings(csv_file_path):
    """
    Analyze a CSV file containing query-restaurant-relevance data
    and output the number of restaurants with relevance score 3
    out of the total for each unique query.
    """
    # Initialize dictionaries to store counts
    query_totals = defaultdict(int)
    query_threes = defaultdict(int)
    
    # Read the CSV file
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        # Skip header row
        header = next(csv_reader)
        
        # Process each row
        for row in csv_reader:
            if len(row) >= 3:  # Ensure row has enough columns
                query = row[0]
                relevance_score = row[2]
                
                # Increment total count for this query
                query_totals[query] += 1
                
                # If relevance score is 3, increment the threes count
                if relevance_score == '3':
                    query_threes[query] += 1
    
    # Print results
    print("Query | 3s / Total | Percentage")
    print("-" * 50)
    
    for query in query_totals:
        threes = query_threes[query]
        total = query_totals[query]
        percentage = (threes / total) * 100 if total > 0 else 0
        
        print(f"{query[:40]}... | {threes}/{total} | {percentage:.2f}%")

if __name__ == "__main__":
    # Check if file path is provided
    if len(sys.argv) > 1:
        csv_file_path = sys.argv[1]
    else:
        # Default to a sample path if none provided
        #csv_file_path = "per_pair_labeling/datasets/restaurant/new_orl_gemini_labels/restaurant_recommendation/gemini_labels.csv"
        #csv_file_path = "per_pair_labeling/datasets/travel_dest/sample_5_gemini_labels.csv"
        csv_file_path = "per_pair_labeling/datasets/restaurant/new_orl/gemini_labels_apr9_part1.csv"
    analyze_query_ratings(csv_file_path)
