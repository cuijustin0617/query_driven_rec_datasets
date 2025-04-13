import json
import matplotlib.pyplot as plt
import argparse
from google import genai
import time
import numpy as np
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze query specificity and plot against scores')
    parser.add_argument('--json_file', type=str, required=True, help='Path to the JSON file containing queries')
    return parser.parse_args()

def rate_all_queries(queries, client):
    # Prepare a numbered list of all queries
    query_list = "\n".join([f"{i+1}. \"{query}\"" for i, query in enumerate(queries)])
    
    prompt = f"""
    For each of the following queries, rate them on a scale of 1-10 based on how specific they are:
    1 = very broad, vague, or unclear
    10 = very specific, detailed, with clear needs
    
    {query_list}
    
    Provide your ratings in a numbered list format matching the query numbers above. 
    Each line should be in the format: [query_number]. [rating]
    For example: 1. 7
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        
        # Parse the response to extract ratings
        ratings = {}
        response_text = response.text.strip()
        
        # Use regex to extract ratings
        pattern = r'(\d+)\.\s*(\d+(?:\.\d+)?)'
        matches = re.findall(pattern, response_text)
        
        for match in matches:
            query_num = int(match[0])
            rating = float(match[1])
            if 1 <= query_num <= len(queries) and 1 <= rating <= 10:
                ratings[query_num] = rating
        
        # Ensure we have ratings for all queries
        if len(ratings) != len(queries):
            print(f"Warning: Only got {len(ratings)} ratings for {len(queries)} queries")
        
        # Convert to ordered list
        result = [ratings.get(i+1, None) for i in range(len(queries))]
        return result
    
    except Exception as e:
        print(f"Error rating queries: {e}")
        return [None] * len(queries)

def main():
    args = parse_args()
    api_key = "AIzaSyD3fnGbKojcbSYiD2eKJQvum0oF4N5iWlA"
    
    # Initialize Gemini API client
    client = genai.Client(api_key=api_key)
    
    # Load the JSON file
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    queries_dict = data["query_ranks"]
    
    # Extract queries and scores
    query_texts = list(queries_dict.keys())
    query_scores = list(queries_dict.values())
    
    # Get ratings for all queries at once
    print(f"Rating {len(query_texts)} queries...")
    specificity_ratings = rate_all_queries(query_texts, client)
    
    # Filter out None values (if any errors occurred)
    valid_data = [(text, score, rating) 
                 for text, score, rating in zip(query_texts, query_scores, specificity_ratings) 
                 if rating is not None]
    
    if not valid_data:
        print("No valid ratings received. Exiting.")
        return
        
    # Unpack the filtered data
    filtered_texts, filtered_scores, filtered_ratings = zip(*valid_data)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.scatter(filtered_ratings, filtered_scores)
    
    # Add labels and title
    plt.xlabel('Specificity Rating (1=broad, 10=specific)')
    plt.ylabel('Query Score')
    plt.title('Query Specificity vs. Score')
    
    # Add a trend line
    z = np.polyfit(filtered_ratings, filtered_scores, 1)
    p = np.poly1d(z)
    plt.plot(filtered_ratings, p(filtered_ratings), "r--")
    
    # Save the plot
    plt.savefig('query_specificity_vs_score.png')
    
    # Also save the raw data for further analysis
    results = {
        "queries": filtered_texts,
        "specificity_ratings": filtered_ratings,
        "query_scores": filtered_scores
    }
    
    with open('query_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis complete. Results saved to query_analysis_results.json and plot saved to query_specificity_vs_score.png")

if __name__ == "__main__":
    main() 