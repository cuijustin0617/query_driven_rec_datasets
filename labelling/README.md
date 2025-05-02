# Query-Document Relevance Labeling Pipeline

This system generates binary relevance labels (True/False) for query-document pairs using LLMs. It's designed to be modular, configurable, and easy to use across different domains (cities, restaurants, hotels).

## Setup

1. Install the required dependencies:
   ```
   pip install -q -U google-genai
   pip install dotenv
   ```

2. Set up your API key in a `.env` file:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Configuration

There are two ways to configure the system:

### Option 1: Edit the configuration file directly (recommended)

Open `labelling/config.py` and modify the settings at the top of the file:

```python
# Change these settings as needed for your labeling project
DEFAULT_DOMAIN = "city"                               # Options: "city", "restaurant", "hotel"
DEFAULT_CORPUS_DIR = "datasets/Traveldest/corpus"     # Directory containing your corpus text files
DEFAULT_QUERIES_FILE = "datasets/Traveldest/queries_travel.txt"  # File with your queries

# Output configuration - customize these settings for your output files
DEFAULT_OUTPUT_DIR = "labelling/output/traveldest"    # Where to save results (folder will be created if it doesn't exist)
DEFAULT_OUTPUT_FILENAME = "gemini_labels.csv"         # Name of the output CSV file (results will be saved here)

# LLM configuration
DEFAULT_LLM_CLIENT = "gemini"                         # Currently only "gemini" is supported
DEFAULT_MODEL_NAME = "gemini-2.0-flash"               # Model to use
```

### Option 2: Use command line arguments

You can also override the default configuration using command-line arguments:

```
python -m labelling.label --domain hotel --corpus-dir datasets/Hotels/corpus --queries-file datasets/Hotels/queries.txt --output-dir my_results --output-filename hotel_labels.csv
```

### Domain-specific Configuration

The system supports different domains, each with its own terminology:
- `city`: For travel destinations
- `restaurant`: For dining options
- `hotel`: For accommodations

## Corpus Format

Each document in your corpus should be a separate text file in the corpus directory. The filename (without extension) will be used as the entity name.

## Queries Format

Queries should be provided in a plain text file, with one query per line.

## Usage

Run the labeling script with the default configuration (after setting your preferences in config.py):

```
python -m labelling.label
```

Or with custom configuration via command line:

```
python -m labelling.label --domain hotel --corpus-dir datasets/Hotels/corpus --queries-file datasets/Hotels/queries.txt --output-dir my_results --output-filename hotel_labels.csv
```

## Output

The labeling system generates two types of output files:

1. **CSV File**
   - **Location**: Saved in the specified output directory 
   - **Structure**:
     - Each row represents a query-entity pair.
     - Columns:
       - `Query`: The search query.
       - `Entity`: The name of the entity (e.g., restaurant, hotel).
       - `Relevance`: Boolean (`True`/`False`) indicating whether the entity is relevant to the query.
   - **Notes**:
     - If the LLM provides any response other than "True" (case insensitive), the system defaults to "False".

2. **Ground Truth JSON File**
   - **Location**: Automatically generated in the same directory as the CSV file with the filename `ground_truth.json`.
   - **Trigger**: Created only when all query-entity pairs have been labeled (i.e., the labeling process is complete).
   - **Structure**:
     - A JSON dictionary where each key is a query and the value is a list of relevant entities (those marked as `True` in the CSV).
     - Example:
       ```json
       {
         "Where can I find cuisine rooted in ancient cooking methods?": ["Restaurant A", "Restaurant B"],
         "Where can I experience a cultural journey through the tasting menu?": ["Restaurant C"]
       }
       ```

### Notes
- The system ensures all outputs are saved in UTF-8 encoding.
- The CSV file is updated incrementally as labeling progresses, while the JSON file is generated only once at the end.

## Extending the System

To add support for new LLM providers:
1. Create a new client class in `llm_clients.py`
2. Update the `get_llm_client` function to support the new client
3. Add the new client to the configuration options
