#!/usr/bin/env python3
import os
import json
from pathlib import Path

def merge_json_files(file1, file2, output_file):
    """
    Merge two JSON files containing queries and scores into one file.
    Combines the dictionaries instead of concatenating content.
    """
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read and parse JSON from both files
    with open(file1, 'r') as f1:
        data1 = json.load(f1)
    
    with open(file2, 'r') as f2:
        data2 = json.load(f2)
    
    # Combine the dictionaries
    combined_data = {**data1, **data2}
    
    # Write merged content to output file
    with open(output_file, 'w') as out:
        json.dump(combined_data, out, indent=2)

def merge_folders(source1, source2, destination):
    """
    Recursively merge JSON files from two source folders into a destination folder.
    Only files with the same relative path in both source folders are merged.
    """
    # Convert to Path objects for easier path manipulation
    source1_path = Path(source1)
    source2_path = Path(source2)
    destination_path = Path(destination)
    
    # Create destination directory if it doesn't exist
    os.makedirs(destination_path, exist_ok=True)
    
    # Walk through the first source directory
    for root, dirs, files in os.walk(source1_path):
        # Get the relative path from source1
        rel_path = Path(root).relative_to(source1_path)
        
        # Create corresponding directories in destination
        if rel_path != Path('.'):  # Skip the root directory itself
            os.makedirs(destination_path / rel_path, exist_ok=True)
        
        # Process each file
        for file in files:
            # Source paths
            file1 = Path(root) / file
            file2 = source2_path / rel_path / file
            
            # Destination path
            dest_file = destination_path / rel_path / file
            
            # Only merge if the file exists in both directories
            if file2.exists():
                print(f"Merging: {file1} and {file2} -> {dest_file}")
                merge_json_files(file1, file2, dest_file)
            else:
                print(f"Warning: {file2} does not exist, skipping merge.")

def main():
    source1 = "restaurant_35"
    source2 = "restaurant_123"
    destination = "restaurant_158"
    
    # Check if source directories exist
    if not os.path.exists(source1):
        print(f"Error: Source directory '{source1}' does not exist.")
        return
    
    if not os.path.exists(source2):
        print(f"Error: Source directory '{source2}' does not exist.")
        return
    
    # Create destination directory if it doesn't exist
    if os.path.exists(destination):
        print(f"Warning: Destination directory '{destination}' already exists. Files may be overwritten.")
    else:
        os.makedirs(destination)
    
    # Merge the folders
    merge_folders(source1, source2, destination)
    print(f"Merged data from '{source1}' and '{source2}' into '{destination}'.")

if __name__ == "__main__":
    main()