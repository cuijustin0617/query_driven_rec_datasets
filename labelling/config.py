"""
Configuration settings for the labeling pipeline.
"""
import os
from pathlib import Path
from typing import Dict, List

#############################################
###    USER CONFIGURATION SETTINGS       ###
#############################################

# Change these settings as needed for your labeling project
DEFAULT_DOMAIN = "city"                                          # Options: "city", "restaurant", "hotel"
DEFAULT_CORPUS_DIR = "datasets/Traveldest/corpus/cities"         # Directory containing your corpus text files
DEFAULT_QUERIES_FILE = "datasets/Traveldest/queries_travel.txt"  # File with your queries

# Output configuration - customize these settings for your output files
DEFAULT_OUTPUT_DIR = "labelling/output/traveldest"               # Where to save results (folder will be created if it doesn't exist)
DEFAULT_OUTPUT_FILENAME = "gemini_labels.csv"                    # Name of the output CSV file

# LLM configuration
DEFAULT_LLM_CLIENT = "gemini"                                    # Currently only "gemini" is supported
DEFAULT_MODEL_NAME = "gemini-2.0-flash"                          # Model to use

#############################################
### END OF USER CONFIGURATION SETTINGS    ###
#############################################

# Define domain-specific language mappings
DOMAIN_MAPPINGS = {
    "city": {
        "singular": "city",
        "plural": "cities",
        "context": "travel destination",
        "person": "traveler",
        "csv_entity_header": "City",
        "description_term": "City Description",
        "entity_intro": "City Info",
    },
    "restaurant": {
        "singular": "restaurant",
        "plural": "restaurants",
        "context": "dining option",
        "person": "diner",
        "csv_entity_header": "Restaurant",
        "description_term": "Restaurant Description",
        "entity_intro": "Restaurant Info",
    },
    "hotel": {
        "singular": "hotel",
        "plural": "hotels",
        "context": "accommodation",
        "person": "guest",
        "csv_entity_header": "Hotel",
        "description_term": "Hotel Description",
        "entity_intro": "Hotel Info",
    }
}

class LabelingConfig:
    """Configuration for the labeling pipeline."""
    
    def __init__(
        self,
        domain: str = DEFAULT_DOMAIN,
        corpus_dir: str = DEFAULT_CORPUS_DIR,
        queries_file: str = DEFAULT_QUERIES_FILE,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        output_filename: str = DEFAULT_OUTPUT_FILENAME,
        llm_client: str = DEFAULT_LLM_CLIENT,
        model_name: str = DEFAULT_MODEL_NAME
    ):
        # Validate domain
        if domain not in DOMAIN_MAPPINGS:
            raise ValueError(f"Domain must be one of: {list(DOMAIN_MAPPINGS.keys())}")
        
        self.domain = domain
        self.domain_mappings = DOMAIN_MAPPINGS[domain]
        
        # Paths
        self.corpus_dir = corpus_dir
        self.queries_file = queries_file
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.output_path = os.path.join(output_dir, output_filename)
        
        # LLM settings
        self.llm_client = llm_client  # Type of LLM client to use
        self.model_name = model_name  # Model name for the LLM client
    
    def get_csv_headers(self) -> List[str]:
        """Get CSV headers based on the domain."""
        return ['Query', self.domain_mappings['csv_entity_header'], 'Relevance']

def get_default_config() -> LabelingConfig:
    """Returns the default configuration."""
    return LabelingConfig() 