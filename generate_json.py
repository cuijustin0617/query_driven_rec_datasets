import os
import csv
import json
from pathlib import Path

def generate_structured_json():
    # Initialize dictionary to store query -> restaurants mapping
    query_restaurants = {}
    
    # Get all CSV files in the judgements directory
    judgements_dir = Path('all_judgements')
    csv_files = list(judgements_dir.glob('*.csv'))
    
    # Process each CSV file
    for csv_path in csv_files:
        restaurant_name = csv_path.stem  # Get filename without extension
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Process each row in the CSV
            for row in reader:
                query = row['Query'].strip('"')  # Remove quotes from query
                relevant = row['Relevant'].lower() == 'true'
                
                # If query is relevant, add restaurant to the list
                if relevant:
                    if query not in query_restaurants:
                        query_restaurants[query] = []
                    query_restaurants[query].append(restaurant_name)
    
    # Write the structured JSON file
    output_path = 'ground_truth_2k.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(query_restaurants, f, indent=2)
    
    print(f"Generated JSON file at {output_path}")

if __name__ == "__main__":
    generate_structured_json()
