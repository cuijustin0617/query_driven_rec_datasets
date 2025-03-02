import json

def merge_json_files():
    # Initialize empty dict to store merged data
    merged_data = {}
    
    # List of input files
    input_files = [
        'New_Orleans/ground_truth_new_1.json',    ############TO CHANGE
        'New_Orleans/ground_truth_new_2.json',      ############TO CHANGE
                
    ]
    
    # Read and merge each file
    for file_path in input_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Print number of keys in this file
                print(f"{file_path}: {len(data)} keys")
                # Update merged_data with new data
                merged_data.update(data)
        except FileNotFoundError:
            print(f"Warning: Could not find file {file_path}")
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in file {file_path}")
    
    # Print total number of keys in merged data
    print(f"\nTotal keys in merged file: {len(merged_data)}")
    
    # Write merged data to output file
    output_file = 'New_Orleans/ground_truth_new_merged.json'   ############TO CHANGE
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully merged files into {output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    merge_json_files()
