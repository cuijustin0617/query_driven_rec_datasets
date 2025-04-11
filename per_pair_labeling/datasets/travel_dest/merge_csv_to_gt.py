import pandas as pd
import json
import os

# CSV files to process
csv_files = [
    "gemini_labels_apr8_part1.csv",
    "gemini_labels_apr8_part2.csv",
    "gemini_labels_apr8_part3.csv"
]

# Function to process each CSV file
def process_csv(file_path):
    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)
    
    # Get all unique queries
    unique_queries = df['Query'].unique()
    print(f"Total unique queries in {file_path}: {len(unique_queries)}")
    
    qualified_queries = []
    disqualified_count = 0
    disqualified_too_high = 0
    disqualified_too_low = 0
    
    # For each query, check how many rows have relevant score = 3
    for query in unique_queries:
        query_rows = df[df['Query'] == query]
        
        # Check if we have the expected number of rows for this query
        if len(query_rows) != 774:
            print(f"Warning: Query '{query}' has {len(query_rows)} rows instead of 774.")
        
        # Count rows with relevant score = 3
        count_rel_3 = len(query_rows[query_rows['Relevance Score'] == 3])
        percentage_rel_3 = (count_rel_3 / len(query_rows)) * 100
        
        # Disqualify if less than 1.5% or more than 30% have relevant score = 3
        if percentage_rel_3 >= 30:
            disqualified_count += 1
            disqualified_too_high += 1
        elif percentage_rel_3 < 1.5:
            disqualified_count += 1
            disqualified_too_low += 1
        else:
            # Get the cities for this query with relevance = 3
            relevant_cities = query_rows[query_rows['Relevance Score'] == 3]['City'].tolist()
            qualified_queries.append((query, relevant_cities))
    
    print(f"Qualified queries in {file_path}: {len(qualified_queries)}")
    print(f"Disqualified queries in {file_path}: {disqualified_count}")
    print(f"  - Disqualified for having ≥ 30% relevance 3: {disqualified_too_high}")
    print(f"  - Disqualified for having < 1.5% relevance 3: {disqualified_too_low}")
    return qualified_queries

# Process all CSV files and collect qualified queries
all_qualified_queries = {}
for csv_file in csv_files:
    file_path = os.path.join(os.path.dirname(__file__), csv_file)
    qualified_queries = process_csv(file_path)
    
    # Add to the combined results dictionary
    for query, cities in qualified_queries:
        if query in all_qualified_queries:
            # Combine cities, removing duplicates
            all_qualified_queries[query] = list(set(all_qualified_queries[query] + cities))
        else:
            all_qualified_queries[query] = cities

# Print total count of qualified queries
print(f"\nTotal unique qualified queries across all files: {len(all_qualified_queries)}")
total_queries = sum(len(pd.read_csv(os.path.join(os.path.dirname(__file__), csv_file))['Query'].unique()) for csv_file in csv_files)
print(f"Total disqualified queries: {total_queries - len(all_qualified_queries)}")

# Create ground truth JSON file
output_file = os.path.join(os.path.dirname(__file__), "ground_truth.json")
with open(output_file, 'w') as f:
    json.dump(all_qualified_queries, f, indent=2)

print(f"Ground truth file created: {output_file}")
print(f"Number of queries in ground truth file: {len(all_qualified_queries)}")
