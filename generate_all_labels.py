import os
import csv
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import time

load_dotenv()

def read_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def process_restaurant(doc_path, queries_path, output_path, client):
    # Read restaurant content
    
    restaurant_content = read_file_content(doc_path)
    restaurant_name = Path(doc_path).stem
    
    # Read queries and number them with period after number to match output format
    queries = read_file_content(queries_path).strip().split('\n')
    numbered_queries = [f"{i+1}. {query}" for i, query in enumerate(queries)]
    queries_text = '\n'.join(numbered_queries)
    
    prompt = f"""Given the following restaurant information:
{restaurant_content}

For each numbered query below, determine if it is relevant to the restaurant information provided.
{queries_text}

Provide your answer as a numbered list matching the query numbers exactly.
Your response must follow this format:
1. True
2. False
3. True
etc.

Important: Use the same numbering format (number followed by period) as in the queries.
Respond with ONLY the numbered True/False values, no other text."""

    print("Waiting for response...")

    # Keep trying until successful
    attempt = 1
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",  ################TO CHANGE
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that determines if queries are relevant to restaurant information."},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )
            print("Response received")
            break  # If successful, exit the retry loop
        except Exception as e:
            print(f"Attempt {attempt} failed: {str(e)}. Retrying in 10 seconds...")
            time.sleep(10)
            attempt += 1

    # Parse response
    response_lines = response.choices[0].message.content.strip().split('\n')
    results = []
    
    for i, query in enumerate(queries):
        # Extract just the True/False value from response line
        relevance = response_lines[i].split('. ')[1].strip()
        results.append([query, relevance])
    
    # Save results to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Query', 'Relevant'])
        writer.writerows(results)
        
    print(f"Processed {restaurant_name}: Results saved to {output_path}")

def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_eqr"))  ################TO CHANGE
    
    # Directory containing restaurant text documents to process
    docs_dir = Path('New_Orleans/new_orleans_1k')   ################TO CHANGE
    
    # txt file containing the queries to evaluate against each restaurant -- one query per line
    queries_path = 'queries_multi.txt'  ################TO CHANGE
    
    # Output directory for storing relevance judgement CSV files, one csv per restaurant
    output_folder = Path('New_Orleans/judgements_multi_1k')  ################TO CHANGE



    # Process each restaurant document
    for doc_path in docs_dir.glob('*.txt'):
        output_path = output_folder / f"{doc_path.stem}.csv"
        if not output_path.exists():
            print(f"Processing {doc_path}...")
            process_restaurant(doc_path, queries_path, output_path, client)


        else:
            print(f"Skipping {doc_path} as it already has a judgement.")

if __name__ == "__main__":
    main()
