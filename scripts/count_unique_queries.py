import pandas as pd
import sys

def count_unique_queries(csv_file):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Get the first column name
        first_column = df.columns[0]
        
        # Count unique values in the first column
        unique_count = df[first_column].nunique()
        
        print(f"Total number of unique queries in the first column: {unique_count}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_unique_queries.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    count_unique_queries(csv_file) 