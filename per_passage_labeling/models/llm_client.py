import time
import json
import os
import random
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod

from openai import OpenAI
from google import genai
from google.genai import types

from ..config import LLM_PROVIDER, LLM_MODEL, MAX_RETRIES

# Load environment variables from .env file
load_dotenv()

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    def get_response(self, messages: List[Dict[str,Any]]) -> str:
        """Get a response from the LLM."""
        for attempt in range(MAX_RETRIES):
            try: 
                result = self._call_api(messages)
                return result
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    sleep_time = 2**attempt
                    print(f"Error: {e}. Attempt {attempt + 1} failed. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    raise Exception(f"Failed after {MAX_RETRIES} attempts: {str(e)}")

    @abstractmethod
    def _call_api(self, messages: List[Dict[str, Any]]) -> str:
        pass

class OpenAIClient(BaseLLMClient):
    """Client for interacting with the OpenAI API."""
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        if OpenAI is None:
            raise ImportError("OpenAI is not installed. Please install it with `pip install openai`.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content.strip()

class GeminiClient(BaseLLMClient):
    """Client for Google Gemini API with multiple API key support."""
    
    def __init__(self, api_keys: List[str], model: str):
        """
        Initialize Gemini client with multiple API keys.
        
        Args:
            api_keys: List of Google API keys
            model: Model name (e.g., "gemini-pro")
        """
        if genai is None:
            raise ImportError("Google generativeai package is not installed. Install it with 'pip install google-generativeai'")
        
        if not api_keys:
            raise ValueError("No API keys provided for Gemini client")
        
        self.api_keys = api_keys
        self.model = model
        self.current_key_index = 0
        self.key_failure_counts = {i: 0 for i in range(len(api_keys))}
        self.max_failures_per_key = 2
        
        # Initialize with the first key
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the client with the current API key."""
        self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
    
    def _rotate_key(self):
        """Rotate to the next available API key."""
        # Reset failure count for the current key
        self.key_failure_counts[self.current_key_index] = 0
        
        # Move to the next key
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        
        # Initialize client with the new key
        self._initialize_client()
        print(f"Switched to Gemini API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def _call_api(self, messages: List[Dict[str, str]]) -> str:
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
                    model=self.model,
                    config=types.GenerateContentConfig(
                        system_instruction=system_content
                    ),
                    contents=user_model_contents
                )
            else:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_model_contents
                )
            
            # Reset failure count on success
            self.key_failure_counts[self.current_key_index] = 0
            
            return response.text.strip()
            
        except Exception as e:
            # Increment failure count for this key
            self.key_failure_counts[self.current_key_index] += 1
            
            # If we've failed too many times with this key, try another one
            if self.key_failure_counts[self.current_key_index] >= self.max_failures_per_key:
                if len(self.api_keys) > 1:
                    self._rotate_key()
                    # Retry with the new key
                    return self._call_api(messages)
            
            # If we don't rotate or there's only one key, propagate the exception
            raise

def create_llm_client(
    provider: str = LLM_PROVIDER,
    model: str = LLM_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> BaseLLMClient:
    """
    Create an LLM client based on the specified provider.
    
    Args:
        provider: LLM provider name ('openai', 'deepseek', 'gemini')
        model: Model name
        api_key: API key (if None, will be taken from environment)
        base_url: Base URL for the API (if applicable)
        
    Returns:
        An LLM client instance
    """
    provider = provider.lower()
    
    if provider == "openai":
        from ..config import OPENAI_API_KEY
        return OpenAIClient(
            api_key=api_key or OPENAI_API_KEY,
            model=model,
            base_url=base_url
        )
    elif provider == "gemini":
        # Get Gemini API keys from config
        from ..config import GEMINI_API_KEYS
        
        # Ensure we have at least one API key
        if not GEMINI_API_KEYS:
            raise ValueError("No Gemini API keys found. Please set GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc. in your .env file.")
        
        return GeminiClient(
            api_keys=GEMINI_API_KEYS,
            model=model
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported providers are: openai, deepseek, gemini")


# Main client class for external use
class LLMClient:
    """Main LLM client that delegates to the appropriate provider client."""
    
    def __init__(self):
        """Initialize LLM client with the configured provider."""
        self._client = create_llm_client()
    
    def get_completion(self, messages: List[Dict[str, str]]) -> str:
        """Get completion from the LLM."""
        return self._client.get_response(messages)