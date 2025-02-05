import os
import csv
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def read_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def process_restaurant(doc_path, queries_path, client):
    # Read restaurant content
    restaurant_content = read_file_content(doc_path)
    restaurant_name = Path(doc_path).stem
    
    # Read queries
    queries = read_file_content(queries_path).strip().split('\n')
    
    # Prepare output path
    judgements_dir = Path('New_Orleans/judgements')
    judgements_dir.mkdir(exist_ok=True)
    output_path = judgements_dir / f"{restaurant_name}.csv"
    
    results = []
    
    # Process each query
    i=0
    for query in queries:
        i+=1
        print("processing query", i)
        prompt = f"""Given the following restaurant information:
{restaurant_content}

Query: {query}

Is this query relevant to the restaurant information provided? 
Answer with just 'True' if relevant, 'False' if not relevant."""
        print("waiting for response")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that determines if queries are relevant to restaurant information."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        print("response received")
        
        relevance = response.choices[0].message.content.strip()
        results.append([query, relevance])
        
    # Save results to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Query', 'Relevant'])
        writer.writerows(results)
        
    print(f"Processed {restaurant_name}: Results saved to {output_path}")

def main():
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY_0617"), base_url="https://api.deepseek.com")
    docs_dir = Path('New_Orleans/docs')
    queries_path = 'queries.txt'
    
    # Process each restaurant document
    for doc_path in docs_dir.glob('*.txt'):
        output_path = Path('New_Orleans/judgements') / f"{doc_path.stem}.csv"
        if not output_path.exists():
            process_restaurant(doc_path, queries_path, client)
        else:
            print(f"Skipping {doc_path} as it already has a judgement.")

if __name__ == "__main__":
    main()