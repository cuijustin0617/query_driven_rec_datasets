import os
import re

def clean_filename(filename):
    """Remove commas, periods, and question marks from a filename."""
    return re.sub(r'[,.\?]', '', filename)

def rename_files(root_dir):
    """
    Recursively walk through directories and rename JSON and CSV files
    that contain commas, periods, or question marks in their names.
    """
    count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Only process JSON and CSV files
            if filename.lower().endswith(('.json', '.csv')):
                # Check if filename contains any of the characters to be removed
                if any(char in filename for char in [',', '.', '?', '’']):
                    # Make sure we don't remove the file extension
                    name_parts = filename.split('.')
                    extension = name_parts[-1]
                    name_without_ext = '.'.join(name_parts[:-1])
                    
                    # Clean the filename (excluding extension)
                    cleaned_name = clean_filename(name_without_ext)
                    
                    # Create new filename with original extension
                    new_filename = f"{cleaned_name}.{extension}"
                    
                    # Get full paths
                    old_path = os.path.join(dirpath, filename)
                    new_path = os.path.join(dirpath, new_filename)
                    
                    # Rename the file
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                    count += 1
    print(f"Total files renamed: {count}")

if __name__ == "__main__":
    # Use the current directory as the root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rename_files(current_dir)
    print("Renaming process completed!")