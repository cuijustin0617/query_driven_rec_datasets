"""Configuration settings for the restaurant query relevance pipeline."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DOMAIN = "hotel"

# File paths
BASE_DIR = Path(__file__).parent.parent  # root directory
QUERIES_PATH = BASE_DIR / "queries/queries_hotel.txt"
DOCS_DIR = BASE_DIR / "data/docs/nyc_hotels/docs_258"
OUTPUT_DIR = BASE_DIR / "per_query_labeling/datasets/nyc_258"

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

MAX_RETRIES = 8
