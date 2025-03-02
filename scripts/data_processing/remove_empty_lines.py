import os

def remove_empty_lines(file_path):
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Remove empty lines while preserving content
    non_empty_lines = []
    for line in lines:
        if line.strip():  # Keep lines that have content (not just whitespace)
            non_empty_lines.append(line)
    
    # Write the cleaned content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(non_empty_lines)

def main():
    # Get all .txt files in the current directory
    txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]
    
    # Process each .txt file
    for txt_file in txt_files:
        print(f"Processing: {txt_file}")
        remove_empty_lines(txt_file)
        print(f"Completed: {txt_file}")

if __name__ == "__main__":
    main()
