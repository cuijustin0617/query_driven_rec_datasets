import json

def merge_json_files(input_path1, input_path2, output_path):
    # Load the first JSON file
    with open(input_path1, 'r') as file1:
        data1 = json.load(file1)
    
    # Load the second JSON file
    with open(input_path2, 'r') as file2:
        data2 = json.load(file2)
    
    # # Load the third JSON file
    # with open(input_path3, 'r') as file3:
    #     data3 = json.load(file3)
    
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
    input_path1 = 'ground_truth/restaurant/nor/ground_truth_apr13_158.json'  # Change this to your first JSON file path
    input_path2 = 'ground_truth/restaurant/nor/ground_truth_apr15_49.json'  # Change this to your second JSON file path
    output_path = 'ground_truth/restaurant/nor/ground_truth_apr15_207.json'   # Change this to your desired output file path
    
    # input_path1 = 'ground_truth/hotel/chicago/ground_truth_apr15_124.json'  # Change this to your first JSON file path
    # input_path2 = 'ground_truth/hotel/chicago/ground_truth_final_143.json'  # Change this to your second JSON file path
    # output_path = 'ground_truth/hotel/chicago/ground_truth_full_267.json'   # Change this to your desired output file path
    
    merge_json_files(input_path1, input_path2, output_path)
