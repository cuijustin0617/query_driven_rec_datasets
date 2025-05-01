"""
LLM clients for different models used in the labeling process.
"""
import os
import time
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

load_dotenv()

# Try importing Gemini
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class ChatGemini:
    """Client for Google Gemini API."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google generativeai package is not installed. Install it with 'pip install google-generativeai'")
        
        self.model_name = model_name
        
        # Get API key from environment variables
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("No API key provided for Gemini client. Set GEMINI_API_KEY in your .env file.")
        
        # Initialize client
        self.client = genai.Client(api_key=self.api_key)
        print(f"Using Gemini API with model: {model_name}")
    
    def generate(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> Union[str, None]:
        """Get a response from Gemini with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try: 
                result = self._call_api(messages, temperature)
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    sleep_time = 1.2**attempt
                    print(f"Error: {e}. Attempt {attempt + 1} failed. Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"Failed after {max_retries} attempts: {str(e)}")
                    return None
    
    def _call_api(self, messages: List[Dict[str, Any]], temperature: float) -> str:
        """Make API call to Gemini."""
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
        
        return response.text.strip()

def parse_binary_response(response_text: str) -> Optional[bool]:
    """
    Parse True/False from LLM's text response.
    If response is not exactly 'true' (case insensitive), defaults to False.
    Returns None only if there was no response at all.
    """
    if not response_text:
        return None
    
    try:
        # Convert to lowercase and strip whitespace
        clean_text = response_text.strip().lower()
        
        # Only return True if exactly "true"
        if clean_text == "true":
            return True
        else:
            # Default to False for any other response
            return False
        
    except Exception as e:
        print(f"Error parsing binary response: {e}")
        return False

def get_llm_client(config):
    """Factory function to get the appropriate LLM client based on config."""
    if config.llm_client.lower() == "gemini":
        return ChatGemini(model_name=config.model_name)
    else:
        raise ValueError(f"Unsupported LLM client: {config.llm_client}") 