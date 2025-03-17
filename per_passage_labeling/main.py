"""Main entry point for the passage-based relevance pipeline."""
import time
from pathlib import Path
import shutil
import json

from per_passage_labeling.models.llm_client import LLMClient
from .pipeline.pipeline import PassageRelevancePipeline
from .config import OUTPUT_DIR, GROUND_TRUTH_PATH

def save_config_to_output_dir():
    """Save the current config file to the output directory for reference."""
    output_config_path = Path(OUTPUT_DIR) / "config.py"
    current_config_path = Path(__file__).parent / "config.py"
    
    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Copy config file to output directory
    shutil.copy(current_config_path, output_config_path)

def main():
    """Run the passage-based relevance pipeline."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Save config file to output directory
    save_config_to_output_dir()

    # Start time
    start_time = time.time()
    
    # Initialize LLM client and pipeline
    llm_client = LLMClient()
    pipeline = PassageRelevancePipeline(llm_client)
    ground_truth = pipeline.run()
    
    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print results
    print(f"\nPipeline completed in {elapsed_time:.2f} seconds")
    print(f"Ground truth saved to {GROUND_TRUTH_PATH}")
    print(f"Processed {len(ground_truth)} queries")
    
    
if __name__ == "__main__":
    main()
