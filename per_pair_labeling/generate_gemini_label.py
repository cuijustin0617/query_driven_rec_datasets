import os
import csv
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Union, List, Dict, Any
import json
import time
import re
from dotenv import load_dotenv
from config import PipelineConfig, get_default_config

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
for i in range(1, 16):  # Try keys from 1-20
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
        self.successful_calls_count = 0  # Track successful API calls for current key
        self.max_successful_calls = 10   # Rotate after this many successful calls
        
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
        self.successful_calls_count = 0  # Reset successful calls counter
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
                self.successful_calls_count += 1
                
                # Check if we've reached the max successful calls threshold
                if self.successful_calls_count >= self.max_successful_calls:
                    print(f"Rotating key after {self.successful_calls_count} successful calls")
                    self._rotate_key()
            
            return response.text.strip()
            
        except Exception as e:
            # Increment failure count for this key
            if not self.using_paid_key:
                self.key_failure_counts[self.current_key_index] += 1
                self.consecutive_failures += 1
                self.successful_calls_count = 0  # Reset successful calls on failure
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

def process_json(input_json_path, output_csv_path, gemini_client, config):
    """Process JSON data using the configured domain."""
    # Check if output file exists and load existing results
    existing_results = {}
    if os.path.exists(output_csv_path):
        print(f"Found existing output file: {output_csv_path}")
        with open(output_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 3:
                    query, entity, score = row[0], row[1], row[2]
                    if query not in existing_results:
                        existing_results[query] = {}
                    existing_results[query][entity] = score
        print(f"Loaded {sum(len(entities) for entities in existing_results.values())} existing results")

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    total_queries = len(data)
    query_counter = 0
    processed_count = 0
    save_frequency = 10  # Save after every 10 processed items
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Get the appropriate prompt based on config
    prompts = config.format_prompts()
    selected_prompt = prompts["UMBRELA_PROMPT_3"]  # Using prompt 3 as default
    entity_type = config.domain_mappings["singular"]
    
    for query, entity_data in data.items():
        entity_counter = 0
        query_counter += 1
        print(f"\nProcessing query {query_counter}/{total_queries}: {query}")
        
        for entity, value_list in entity_data.items():
            # Skip if this query-entity pair has already been processed
            if query in existing_results and entity in existing_results[query]:
                print(f"  - Skipping already processed {entity_type}: {entity}")
                # Add to results to maintain a complete output file
                results.append([query, entity, existing_results[query][entity]])
                continue
                
            entity_counter += 1
            
            # Extract the document list from the second element of value_list
            # Each entity maps to a list where the 2nd element contains the actual documents
            if isinstance(value_list, list) and len(value_list) >= 2 and isinstance(value_list[1], list):
                documents = value_list[1]  # Get the actual document list (second element)
                print(f"  - Processing {entity_type}: {entity} with {len(documents)} documents")
            else:
                print(f"  - Warning: Unexpected format for {entity}. Treating entire value as document.")
                documents = [str(value_list)]
            
            # Convert all document items to strings before joining
            documents_as_strings = [str(doc) for doc in documents]
            document = "\n".join(documents_as_strings)
            
            # Format the prompt with entity name
            prompt = selected_prompt.format(
                query=query, 
                document=document, 
                entity_name=entity  # Use the actual entity name for personalization
            )
            # print(f"    Prompt: {prompt}")

            messages = [
                {"role": "system", "content": f"You are a helpful assistant that evaluates relevance scores for {entity_type}-related query-document pairs."},
                {"role": "user", "content": prompt},
            ]

            response_text = gemini_client.generate(messages, temperature=0.0)
            print(f"    Response: {response_text}")

            if response_text is None:
                print(f"    Failed to get response for {entity_type}: {entity}")
                final_score = "Error"
            else:
                try:
                    final_score = parse_gemini_score_response(response_text)
                    if final_score is None:
                        print(f"    Failed to parse score from response: {response_text}")
                        final_score = "ParsingError"
                except Exception as e:
                    print(f"    Parsing error for {entity_type} {entity}: {e}")
                    final_score = "ParsingError"
            
            results.append([query, entity, final_score])
            processed_count += 1
            
            # Save intermediate results periodically
            if processed_count % save_frequency == 0:
                save_results_to_csv(output_csv_path, results, config)
                print(f"Saved intermediate results after processing {processed_count} items")

    # Final save of all results
    save_results_to_csv(output_csv_path, results, config)
    print(f"\nProcessing completed. Results saved to {output_csv_path}")

def save_results_to_csv(output_csv_path, results, config):
    """Save results to CSV file with domain-specific headers."""
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(config.get_csv_headers())
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

    # Load configuration
    config = get_default_config()
    
    # Use paths from config
    process_json(config.input_json_path, config.output_csv_path, gemini_client, config)

if __name__ == "__main__":
    main()