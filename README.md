# Yelp Restaurants IR Dataset


## Label Generation Process

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
