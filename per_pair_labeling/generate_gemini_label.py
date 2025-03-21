import os
import csv
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Union, List, Dict, Any
import json
import time
import re
from dotenv import load_dotenv

load_dotenv()
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Define the schema for the response
class Score(BaseModel):
    score: int

### API Keys
GEMINI_PAID_API_KEY = os.getenv("GEMINI_PAID_API_KEY")

# Get multiple free Gemini API keys
GEMINI_FREE_API_KEYS = []
for i in range(1, 10):  # Try keys from 1-9
    key = os.getenv(f"GEMINI_API_KEY_{i}")
    if key:
        GEMINI_FREE_API_KEYS.append(key)

class ChatGemini:
    """Client for Google Gemini API with multiple API key support."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash", paid_api_key: Optional[str] = None, free_api_keys: Optional[List[str]] = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google generativeai package is not installed. Install it with 'pip install google-generativeai'")
        
        self.model_name = model_name
        self.paid_api_key = paid_api_key
        self.free_api_keys = free_api_keys or []
        
        if not (self.paid_api_key or self.free_api_keys):
            raise ValueError("No API keys provided for Gemini client")
        
        self.using_paid_key = False  # Default to using free keys
        self.current_key_index = 0
        self.key_failure_counts = {i: 0 for i in range(len(self.free_api_keys))}
        self.max_failures_per_key = 2
        self.consecutive_failures = 0
        self.max_consecutive_failures = 20
        
        # Initialize with free key first even if paid key is available
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the client with the current API key."""
        if self.using_paid_key and self.paid_api_key:
            current_key = self.paid_api_key
            key_type = "paid"
        else:
            current_key = self.free_api_keys[self.current_key_index]
            key_type = f"free {self.current_key_index + 1}/{len(self.free_api_keys)}"
            
        self.client = genai.Client(api_key=current_key)
        print(f"Using Gemini {key_type} API key")
    
    def _rotate_key(self):
        """Rotate to the next available API key."""
        # If already using paid key, stay with it
        if self.using_paid_key:
            print("Already using paid key, sticking with it")
            return
        
        # Move to the next free key
        self.key_failure_counts[self.current_key_index] = 0
        self.current_key_index = (self.current_key_index + 1) % len(self.free_api_keys)
        
        # If we've tried all free keys and have a paid key, switch to paid key
        if self.current_key_index == 0 and self.paid_api_key and self.consecutive_failures >= self.max_consecutive_failures:
            self.using_paid_key = True
            self.consecutive_failures = 0  # Reset counter when switching to paid key
            print(f"Switching to paid key after {self.max_consecutive_failures} consecutive failures")
        
        # Initialize with new key
        self._initialize_client()
    
    def generate(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> Union[str, None]:
        """Get a response from Gemini with retry logic."""
        max_retries = 10
        for attempt in range(max_retries):
            try: 
                result = self._call_api(messages, temperature)
                self.consecutive_failures = 0  # Reset consecutive failures on success
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    sleep_time = 2**attempt
                    print(f"Error: {e}. Attempt {attempt + 1} failed. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"Failed after {max_retries} attempts: {str(e)}")
                    return None
    
    def _call_api(self, messages: List[Dict[str, Any]], temperature: float) -> str:
        """Make API call to Gemini with automatic key rotation on failures."""
        # Extract system prompt if present
        system_content = None
        user_model_contents = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] in ["user", "human"]:
                user_model_contents.append(msg["content"])
            else:
                raise ValueError(f"Unsupported message role: {msg['role']}")
        
        try:
            # Call API with system instruction if provided
            if system_content:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    config=types.GenerateContentConfig(
                        system_instruction=system_content
                    ),
                    contents=user_model_contents
                )
            else:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_model_contents
                )
            
            # Reset failure count on success
            if not self.using_paid_key:
                self.key_failure_counts[self.current_key_index] = 0
                self.consecutive_failures = 0
            
            return response.text.strip()
            
        except Exception as e:
            # Increment failure count for this key
            if not self.using_paid_key:
                self.key_failure_counts[self.current_key_index] += 1
                self.consecutive_failures += 1
                print(f"Consecutive failures across all free keys: {self.consecutive_failures}/{self.max_consecutive_failures}")
            
            # If we've failed too many times with this key, try another one
            if (self.using_paid_key or 
                self.key_failure_counts[self.current_key_index] >= self.max_failures_per_key or
                (not self.using_paid_key and self.consecutive_failures >= self.max_consecutive_failures and self.paid_api_key)):
                self._rotate_key()
                # Retry with the new key without doubling wait time
                return self._call_api(messages, temperature)
            
            # If we don't rotate or there's only one key, propagate the exception
            raise

def parse_gemini_score_response(response_text: str) -> int:
    """Parse score from Gemini's text response."""
    if not response_text:
        return None
    
    try:
        # First try to see if the response is just a single digit
        if response_text.strip() in ["0", "1", "2", "3"]:
            return int(response_text.strip())
        
    except Exception as e:
        print(f"Error parsing score: {e}")
    
    return None

# Gemini-specific prompt with explicit output formatting
UMBRELA_PROMPT = '''
Given a query and a document about a city, provide a score from 0 to 3, where higher scores mean the city is more relevant and helpful for a traveler with this query.

Scoring Guidelines:
0 = The city has little to do with the query.
1 = The city is loosely related to the query, but offers little or no useful information for a traveler.
2 = The city has some relevant features, but they are unclear, not emphasized, or mixed with unrelated details.
3 = The city clearly meets or supports the query goal in a practical or helpful way.

Instructions:
- Assign 3 if the city fulfills the query in any meaningful way, even with extra info.
- Assign 2 if the city shows potential to satisfy the query but lacks key details, or the relevance is partial, vague, or not well-supported.
- Assign 1 if the city is not very helpful for the query.
- Assign 0 only if the city has little relevance to the query.

Query: {query}
City Info: {document}

Split this problem into steps:
Consider the underlying intent of the search.
Measure how well the content matches a likely intent of the query (M).
Measure how trustworthy the passage is (T).
Consider the aspects above and the relative importance of each, and decide on a final score (O). 

Your response must be a single integer between 0-3. Output only the score number and nothing else.
'''

UMBRELA_PROMPT_2 ='''
Given a query and a description of a city, assign a score from 0 to 3 based on how suitable this city is as a travel destination for someone with that interest.
Scoring:
0 = The city is completely unrelated to the query.
1 = The city is weakly related, but is clearly not a good match for what the traveler is looking for.
2 = The city has some relevant aspects, but overall is not a strong or clear match.
3 = The city is a reasonably good or strong match for what the traveler is looking for — even if it's only partially relevant.
Important Instructions:
- Assign a score of 3 if the city has any practical or relevant qualities that could make it a suitable destination for the query.
- Assign 1 only if the city clearly does not fit what the traveler likely wants.
- Assign 0 only if the city has nothing to do with the query.
Query: {query}
City Description: {document}
Steps:
- Consider what kind of destination a traveler is seeking from the query.
- Judge how well the city fits that intent or interest.
- Make a final decision.

Your response must be a single integer between 0-3. Output only the score number and nothing else.
'''

UMBRELA_PROMPT_3 = '''
You are tasked with evaluating how suitable **{city}** is as a travel destination for a traveler based on the provided query and city description. Assign a score from **0 to 3** primarily based on your internal knowledge of the city and the supportive evidence from the provided text.

### **Scoring Guidelines:**
- **0** = The city is irrelevant to the query or contradicts the user's intent.  
- **1** = The city is loosely related to the query but provides little value or relevance for the traveler.  
- **2** = The city has some relevant features to the query, but it is not an ideal fit for the traveler's intent.  
- **3** = The city clearly matches the query goal and is highly suitable as a travel destination.  

### **Input:**
- **Query:** {query}  
- **City:** {city}  
- **City Description:** {document}  

### **Evaluation Steps:**
1. Identify the type of destination or experience the traveler seeks based on the query.  
2. Assess the city's overall strength as a travel destination (e.g., general popularity and tourist appeal).  
3. Evaluate how well the city matches the query's intent based on your internal knowledge.  
4. Cross-check the provided city description for supporting details.  
5. Assign a single final score.  

### **Additional Instructions:**
- Be strict in your rating — only top-tier cities should receive a score of 3. Most cities should be rated 0 or 1 unless they have clear relevance and strong alignment with the query.  
- Consider the overall popularity of the city as a travel destination — some cities have a general popularity advantage over others for most queries.  
- The final score should reflect both the city's specific relevance to the query and its general strength as a travel destination.

Your response must be a single integer between 0-3. Output only the score number and nothing else.
'''

def process_json(input_json_path, output_csv_path, gemini_client):
    # Check if output file exists and load existing results
    existing_results = {}
    if os.path.exists(output_csv_path):
        print(f"Found existing output file: {output_csv_path}")
        with open(output_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 3:
                    query, city, score = row[0], row[1], row[2]
                    if query not in existing_results:
                        existing_results[query] = {}
                    existing_results[query][city] = score
        print(f"Loaded {sum(len(cities) for cities in existing_results.values())} existing results")

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    total_queries = len(data)
    query_counter = 0
    processed_count = 0
    save_frequency = 10  # Save after every 10 processed items
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    for query, city_data in data.items():
        city_counter = 0
        query_counter += 1
        print(f"\nProcessing query {query_counter}/{total_queries}: {query}")
        
        for city, documents in city_data.items():
            # Skip if this query-city pair has already been processed
            if query in existing_results and city in existing_results[query]:
                print(f"  - Skipping already processed city: {city}")
                # Add to results to maintain a complete output file
                results.append([query, city, existing_results[query][city]])
                continue
                
            city_counter += 1
            print(f"  - Processing city: {city} with {len(documents)} documents")
            # concatenate list of passages in a document into a string
            document = "\n".join(documents)
            
            prompt = UMBRELA_PROMPT_3.format(query=query, document=document, city=city)  ## change prompt

            messages = [
                {"role": "system", "content": "You are a helpful assistant that evaluates relevance scores for travel-related query-document pairs."},
                {"role": "user", "content": prompt},
            ]

            response_text = gemini_client.generate(messages, temperature=0.0)
            print(f"    Response: {response_text}")

            if response_text is None:
                print(f"    Failed to get response for city: {city}")
                final_score = "Error"
            else:
                try:
                    final_score = parse_gemini_score_response(response_text)
                    if final_score is None:
                        print(f"    Failed to parse score from response: {response_text}")
                        final_score = "ParsingError"
                except Exception as e:
                    print(f"    Parsing error for city {city}: {e}")
                    final_score = "ParsingError"
            
            results.append([query, city, final_score])
            processed_count += 1
            
            # Save intermediate results periodically
            if processed_count % save_frequency == 0:
                save_results_to_csv(output_csv_path, results)
                print(f"Saved intermediate results after processing {processed_count} items")

    # Final save of all results
    save_results_to_csv(output_csv_path, results)
    print(f"\nProcessing completed. Results saved to {output_csv_path}")

def save_results_to_csv(output_csv_path, results):
    """Save results to CSV file."""
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Query', 'City', 'Relevance Score'])
        writer.writerows(results)

def main():
    # Check if we have any valid API keys before creating the client
    if not GEMINI_PAID_API_KEY and not GEMINI_FREE_API_KEYS:
        print("Error: No Gemini API keys found. Please set GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc. or GEMINI_PAID_API_KEY in your .env file.")
        return
        
    gemini_client = ChatGemini(
        model_name="gemini-2.0-flash", 
        paid_api_key=GEMINI_PAID_API_KEY,
        free_api_keys=GEMINI_FREE_API_KEYS
    )

    input_json_path = "data/dense_results/travel_dest/sample_5_dense_result.json"
    output_csv_path = "per_pair_labeling/datasets/travel_dest/sample_5_gemini_labels.csv"

    process_json(input_json_path, output_csv_path, gemini_client)

if __name__ == "__main__":
    main()