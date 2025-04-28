import json
import math
import os

# Input file path
input_file = "data/dense_results/restaurant/new_orl/dense_result_apr15.json"  ##### TO CHANGE #####

# Output directory
output_dir = "data/dense_results/restaurant/new_orl"  ##### TO CHANGE #####
os.makedirs(output_dir, exist_ok=True)

# Read the input JSON file
with open(input_file, 'r') as f:
    data = json.load(f)

# Count total keys and calculate size for each part
total_keys = len(data)
keys_per_file = math.ceil(total_keys / 3)  ##### TO CHANGE #####

# Split data into three parts
keys = list(data.keys())
parts = []

for i in range(3):    ##### TO CHANGE #####
    start_idx = i * keys_per_file
    end_idx = min((i + 1) * keys_per_file, total_keys)
    
    # Create a subset of the original data
    subset = {}
    for key in keys[start_idx:end_idx]:
        subset[key] = data[key]
    
    parts.append(subset)

# Write each part to a separate JSON file
for i, part in enumerate(parts):
    output_file = os.path.join(output_dir, f"dense_result_apr15_part{i+1}.json")  ##### TO CHANGE #####
    with open(output_file, 'w') as f:
        json.dump(part, f, indent=4)

print(f"Successfully split into 3 files with approximately {keys_per_file} keys each.")