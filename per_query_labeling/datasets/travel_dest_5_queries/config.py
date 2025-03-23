"""Configuration settings for the restaurant query relevance pipeline."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DOMAIN = "travel destination"

# File paths
BASE_DIR = Path(__file__).parent.parent  # root directory
QUERIES_PATH = BASE_DIR / "queries/test_queries.txt"
DOCS_DIR = BASE_DIR / "data/docs/travel_dest_corpus"
OUTPUT_DIR = BASE_DIR / "per_query_labeling/datasets/travel_dest_5_queries"

# Create output directories
SUMMARIES_DIR = Path(OUTPUT_DIR) / "summaries"
RELEVANCE_DIR = Path(OUTPUT_DIR) / "relevance"
GROUND_TRUTH_PATH = Path(OUTPUT_DIR) / "ground_truth.json"

# LLM settings
LLM_PROVIDER = "gemini"  # Options: openai, deepseek, gemini
LLM_MODEL = "gemini-2.0-flash"

# Provider-specific API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# Gemini API key range settings
GEMINI_API_KEY_START_INDEX = 1  # Start index (inclusive), None means start from 1
GEMINI_API_KEY_END_INDEX = 3    # End index (inclusive), None means use all available keys

# Gemini API keys - list of keys for rotation
GEMINI_API_KEYS = []

# Load numbered Gemini API keys from environment variables
i = 1
while True:
    key = os.environ.get(f"GEMINI_API_KEY_{i}")
    if not key:
        break
    GEMINI_API_KEYS.append(key)
    i += 1

# For backward compatibility, if no numbered keys found, use the main key
if not GEMINI_API_KEYS:
    main_key = os.environ.get("GEMINI_API_KEY", "")
    if main_key:
        GEMINI_API_KEYS.append(main_key)

# Filter API keys based on configured range
if GEMINI_API_KEY_START_INDEX is not None or GEMINI_API_KEY_END_INDEX is not None:
    start = GEMINI_API_KEY_START_INDEX or 1
    end = GEMINI_API_KEY_END_INDEX or len(GEMINI_API_KEYS)
    
    # Convert to 0-based indices for list access
    start_idx = max(0, start - 1)
    end_idx = min(len(GEMINI_API_KEYS), end)
    
    # Filter keys by range
    GEMINI_API_KEYS = GEMINI_API_KEYS[start_idx:end_idx]
    
    print(f"Using Gemini API keys {start} through {end} ({len(GEMINI_API_KEYS)} keys)")

MAX_RETRIES = 11
