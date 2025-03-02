import json
import numpy as np
from statistics import median

# Read the JSON data with explicit UTF-8 encoding
with open('Philadelphia/ground_truth_new_merged.json', 'r', encoding='utf-8') as f:    ############TO CHANGE
    data = json.load(f)

# Read the original 50 queries with explicit UTF-8 encoding
with open('queries/queries_new_100 copy.txt', 'r', encoding='utf-8') as f:    ############TO CHANGE
    original_queries = set(line.strip() for line in f)

# Get counts for queries that are in the original 50
counts = []
for query in original_queries:
    if query in data:
        count = len(data[query])
        counts.append(count)
        print(f"Query: {query[:50]}... -> {count} answers")
    else:
        print("NOT FOUND:", query)

# Convert to numpy array for easier analysis
counts = np.array(counts)

# Calculate statistics
avg = np.mean(counts)
med = np.median(counts)
percentiles = np.percentile(counts, [25, 50, 75, 90])

print("\nStatistics:")
print(f"Number of queries analyzed: {len(counts)}")
print(f"Average answers per query: {avg:.2f}")
print(f"Median answers per query: {med:.2f}")
print(f"25th percentile: {percentiles[0]:.2f}")
print(f"50th percentile: {percentiles[1]:.2f}")
print(f"75th percentile: {percentiles[2]:.2f}")
print(f"90th percentile: {percentiles[3]:.2f}")
print(f"Min answers: {np.min(counts)}")
print(f"Max answers: {np.max(counts)}")