"""Configuration settings for the passage-based relevance pipeline."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Domain settings
DOMAIN = "travel destination"  # or "restaurant"

# File paths
BASE_DIR = Path(__file__).parent.parent  # root directory
DENSE_RESULTS_PATH = BASE_DIR / "data/dense_results/travel_dest/dense_result.json"
OUTPUT_DIR = BASE_DIR / "per_passage_labeling/datasets/travel_dest_10_19"

# Create output directories
RELEVANCE_DIR = Path(OUTPUT_DIR) / "relevance"
GROUND_TRUTH_PATH = Path(OUTPUT_DIR) / "ground_truth.json"

# Query selection settings
QUERY_START_INDEX = 10  # Start index (inclusive)
QUERY_END_INDEX = 19    # End index (inclusive) - set to process first 10 queries

# Passage sampling settings
PASSAGES_PER_BATCH = 10  # Number of passages to evaluate per LLM call

# LLM settings
LLM_PROVIDER = "gemini"  # Options: openai, deepseek, gemini
LLM_MODEL = "gemini-2.0-flash"

# Provider-specific API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

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

MAX_RETRIES = 10