import json

def extract_keys_to_txt(json_file_path, output_txt_path):
    """
    Extracts all keys from a JSON file and writes them to a text file, one key per row.

    Args:
        json_file_path (str): Path to the input JSON file.
        output_txt_path (str): Path to the output text file.
    """
    try:
        # Read the JSON file
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        # Extract keys
        keys = data.keys()

        # Write keys to the text file
        with open(output_txt_path, 'w') as txt_file:
            for key in keys:
                txt_file.write(f"{key}\n")

        print(f"Successfully wrote keys to {output_txt_path}")

    except FileNotFoundError:
        print(f"Error: The file {json_file_path} does not exist.")
    except json.JSONDecodeError:
        print(f"Error: The file {json_file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
if __name__ == "__main__":
    json_file_path = "ground_truth/restaurant/phi/ground_truth_apr13_158.json"  # Replace with your JSON file path
    output_txt_path = "ground_truth/restaurant/phi/wi.txt"   # Replace with your desired output file path
    extract_keys_to_txt(json_file_path, output_txt_path)
