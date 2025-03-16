import os
import shutil

def convert_to_txt():
    """
    Reads every file in the current directory, creates a copy with .txt extension,
    and deletes the original files. Files already ending with .txt are skipped.
    """
    # Get the current directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get all files in the current directory
    files = [f for f in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, f))]
    
    # Skip this script itself
    script_name = os.path.basename(__file__)
    if script_name in files:
        files.remove(script_name)
    
    # Process each file
    for filename in files:
        # Skip files that already have .txt extension
        if filename.endswith('.txt'):
            continue
        
        # Create the new filename with .txt extension
        source_path = os.path.join(current_dir, filename)
        txt_filename = filename + '.txt'
        target_path = os.path.join(current_dir, txt_filename)
        
        # Copy the file with new extension, preserving content
        shutil.copy2(source_path, target_path)
        print(f"Created: {txt_filename}")
        
        # Delete the original file
        os.remove(source_path)
        print(f"Deleted original: {filename}")

if __name__ == "__main__":
    convert_to_txt()
    print("Conversion complete!")