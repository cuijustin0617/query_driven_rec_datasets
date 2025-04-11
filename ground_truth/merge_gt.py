import json

def merge_json_files(input_path1, input_path2, output_path):
    # Load the first JSON file
    with open(input_path1, 'r') as file1:
        data1 = json.load(file1)
    
    # Load the second JSON file
    with open(input_path2, 'r') as file2:
        data2 = json.load(file2)
    
    # Print the number of keys in the original files
    print(f"Number of keys in {input_path1}: {len(data1)}")
    print(f"Number of keys in {input_path2}: {len(data2)}")
    
    # Merge the two dictionaries
    merged_data = {**data1, **data2}
    
    # Save the merged data to the output file
    with open(output_path, 'w') as outfile:
        json.dump(merged_data, outfile, indent=4)
    
    # Print the number of keys in the merged output
    print(f"Number of keys in merged output: {len(merged_data)}")

if __name__ == "__main__":
    # Define input and output paths
    input_path1 = 'ground_truth/restaurant/nor/ground_truth_apr10_71.json'  # Change this to your first JSON file path
    input_path2 = 'ground_truth/restaurant/nor/ground_truth_original.json'  # Change this to your second JSON file path
    output_path = 'ground_truth/restaurant/nor/ground_truth_apr10_final.json'   # Change this to your desired output file path
    
    merge_json_files(input_path1, input_path2, output_path)
