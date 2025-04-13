import csv
import sys
from collections import defaultdict

def analyze_query_ratings(csv_file_paths):
    """
    Analyze multiple CSV files containing query-restaurant-relevance data
    and output the number of restaurants with relevance score 3
    out of the total for each unique query for each file.
    """
    for csv_file_path in csv_file_paths:
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
        
        # Print results for the current file
        print(f"\nResults for {csv_file_path}:")
        print("Query | 3s / Total | Percentage")
        print("-" * 50)
        
        for query in query_totals:
            threes = query_threes[query]
            total = query_totals[query]
            percentage = (threes / total) * 100 if total > 0 else 0
            
            print(f"{query[:40]}... | {threes}/{total} | {percentage:.2f}%")

if __name__ == "__main__":
    # Check if file paths are provided
    if len(sys.argv) > 1:
        csv_file_paths = sys.argv[1:]  # Accept multiple file paths
    else:
        # Default to a sample path if none provided
        csv_file_paths = [
            "per_pair_labeling/datasets/restaurant_apr12/new_orl/gemini_labels_apr12_part1.csv",
            "per_pair_labeling/datasets/restaurant_apr12/new_orl/gemini_labels_apr12_part2.csv",
            "per_pair_labeling/datasets/restaurant_apr12/new_orl/gemini_labels_apr12_part3.csv",
            "per_pair_labeling/datasets/restaurant_apr12/phi/gemini_labels_apr12_part1.csv",
            "per_pair_labeling/datasets/restaurant_apr12/phi/gemini_labels_apr12_part2.csv",
            "per_pair_labeling/datasets/restaurant_apr12/phi/gemini_labels_apr12_part3.csv",
            # Add more default paths if needed
        ]
    analyze_query_ratings(csv_file_paths)
