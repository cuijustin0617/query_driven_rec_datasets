# Yelp Restaurants IR Dataset

## Per Document Labeling Pipeline
This pipeline provides a more sophisticated approach to generating relevance judgments by processing documents through a multi-step pipeline using an LLM.


### Pipeline Components

1. **DocumentSummarizer**: Given a query, generates query-specific summaries for each restaurant document

2. **RelevanceJudge**: Determines which restaurants are relevant to a specific query

3. **RelevancePipeline**: Orchestrates the entire process from data loading to ground truth generation

### Setup

1. **Install the package in development mode**:

```bash
pip install -e .
```

2. **Set up environment variables**:

Create a `.env` file in the root directory with the required API keys:
```
# GEMINI_API_KEY=your_gemini_api_key_here
# OPENAI_API_KEY=your_openai_key_here
# DEEPSEEK_API_KEY=your_deepseek_key_here
```
Include whichever API key corresponds to the LLM provider you've configured in the `config.py` file.


3. **Configure the settings**:

Edit the `per_query_labeling/config.py` file to set up your:

- File paths for queries and documents

- Output directories

- LLM provider and model settings

### Running the Pipeline

Run the main script to execute the pipeline:
```python -m per_query_labeling.main```


This pipeline:

- Reads queries and restaurant documents

- Generates restaurant summaries with respect to each query

- Determines relevant items for each query based on all summaries at once

- Creates and saves a ground truth JSON file


### Configuration

Edit the `per_query_labeling/config.py` file to customize:

- File paths for queries, documents, and output directories

- LLM provider and model settings

- API keys and retry settings

The output will be saved to the configured output directory, with separate folders for summaries, relevance judgments, and a final ground truth JSON file.


## Per Query Labeling Pipeline

### 1. Generate Relevance Labels

**Prerequisites**:
   - Restaurant documents in a folder: e.g.`Philadelphia/docs_2k/*.txt`
   - Query file at `queries.txt` -- one query per row
   - Output directory created at a folder e.g. `Philadelphia/judgements_multi_2k/`

Run the label generation script to create relevance judgments for each restaurant:

```bash
python generate_all_labels.py
```

This script performs the following operations:
- Processes each restaurant document against all queries
- Uses GPT-4o mini to determine query relevance
- Generates CSV files containing True/False judgments
- Skips already processed restaurants
- Stores results in the specified output folder (e.g., 'Philadelphia/judgements_multi_2k/')

### 2. Generate Ground Truth JSON

After generating all relevance labels, create the consolidated ground truth JSON file:

```bash
python generate_json.py
```

This script will:
- Process all CSV files from the judgments folder
- Create a JSON file mapping queries to relevant restaurants
- Output the results to a specified JSON file (e.g., 'Philadelphia/ground_truth_cuisine_2k.json')

## Output Format

The final ground truth JSON file will have the following structure:

```json
{
    "query1": ["restaurant1", "restaurant2", ...],
    "query2": ["restaurant3", "restaurant4", ...],
    ...
}
```
