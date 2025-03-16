import json

# Path to files
queries_path = "queries_hotel_v2.txt"
ground_truth_path = "ground_truth.json"
output_path = "extracted_gt.json"

# Read the queries from the text file
with open(queries_path, 'r', encoding='utf-8') as query_file:
    queries = [line.strip() for line in query_file if line.strip()]

# Read the full ground truth data
with open(ground_truth_path, 'r', encoding='utf-8') as gt_file:
    ground_truth = json.load(gt_file)

# Extract only the key-value pairs where the key is in the queries list
extracted_data = {query: ground_truth.get(query, []) for query in queries}

# Save the extracted data to a new JSON file
with open(output_path, 'w', encoding='utf-8') as output_file:
    json.dump(extracted_data, output_file, indent=2)

# Print statistics
print(f"Extracted {len(extracted_data)} queries from {len(ground_truth)} total queries")
print(f"Extracted ground truth data saved to {output_path}")
