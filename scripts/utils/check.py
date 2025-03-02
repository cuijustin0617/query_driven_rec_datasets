import json

def check_queries_in_ground_truth():
    # Read the queries file
    with open('queries/queries_concat_50_refined.txt', 'r', encoding='utf-8') as f:
        # Skip empty lines and get queries, handling line numbers correctly
        queries = []
        for line in f:
            queries.append(line.strip())


    # Read the ground truth JSON file
    with open('Philadelphia/ground_truth_concat_2k.json', 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    # Check each query
    missing_queries = []
    print("Total queries found:", len(queries))
    
    for i, query in enumerate(queries, 1):
        if query not in ground_truth:
            missing_queries.append(query)
            print(f"Processing query {i}: Missing")
        else:
            print(f"Processing query {i}: Found")

    # Print results
    if missing_queries:
        print(f"\nFound {len(missing_queries)} missing queries:")
        for query in missing_queries:
            print(f"- {query}")
    else:
        print("\nAll queries exist in the ground truth file!")

if __name__ == "__main__":
    check_queries_in_ground_truth()
