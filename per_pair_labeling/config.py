import os
from pathlib import Path
from typing import Dict, Any, Optional, List

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

class PipelineConfig:
    """Configuration for the labeling pipeline."""
    
    def __init__(
        self,
        domain: str = "city",
        input_json_path: str = "data/dense_results/travel_dest/dense_result_apr8_part2.json",
        output_dir: str = "per_pair_labeling/datasets/travel_dest",
        output_filename: str = "gemini_labels_apr8_part2.csv",
        disabled_queries_filename: str = "disabled_queries.json"
    ):
        # Validate domain
        if domain not in DOMAIN_MAPPINGS:
            raise ValueError(f"Domain must be one of: {list(DOMAIN_MAPPINGS.keys())}")
        
        self.domain = domain
        self.domain_mappings = DOMAIN_MAPPINGS[domain]
        
        # Set input path
        self.input_json_path = input_json_path
        
        # Set output path
        os.makedirs(output_dir, exist_ok=True)
        self.output_csv_path = os.path.join(output_dir, output_filename)
        
        # Set disabled queries path
        self.disabled_queries_path = os.path.join(output_dir, disabled_queries_filename)
        
        # Parameters for query disabling - too many high scores (easy query)
        self.max_labels_threshold = 200  # Number of labels to check before disabling a query
        self.max_high_score_threshold = 110  # Max number of '3' scores before disabling a query
        
        # Parameters for query disabling - too few high scores (difficult query)
        self.min_labels_threshold = 100  # Check for too few high scores at this threshold
        self.min_high_score_threshold = 3  # Min number of '3' scores required at min_labels_threshold
    
    def get_csv_headers(self) -> list:
        """Get CSV headers based on the domain."""
        return ['Query', self.domain_mappings['csv_entity_header'], 'Relevance Score']
    
    def format_prompts(self) -> Dict[str, str]:
        """Generate domain-specific prompts."""
        
        umbrella_prompt_3 = f'''
You are tasked with evaluating how suitable **{{entity_name}}** is as a {self.domain_mappings['context']} for a {self.domain_mappings['person']} based on the provided query and {self.domain_mappings['singular']} description. Assign a score from **0 to 3** primarily based on your internal knowledge of the {self.domain_mappings['singular']} and the supportive evidence from the provided text.

### **Scoring Guidelines:**
- **0** = The {self.domain_mappings['singular']} is irrelevant to the query or contradicts the user's intent.  
- **1** = The {self.domain_mappings['singular']} is loosely related to the query but provides little value or relevance for the {self.domain_mappings['person']}.  
- **2** = The {self.domain_mappings['singular']} has some relevant features to the query, but it is not an ideal fit for the {self.domain_mappings['person']}'s intent.  
- **3** = The {self.domain_mappings['singular']} clearly matches the query goal and is highly suitable as a {self.domain_mappings['context']}.  

### **Input:**
- **Query:** {{query}}  
- **{self.domain_mappings['csv_entity_header']}:** {{entity_name}}  
- **{self.domain_mappings['description_term']}:** {{document}}  

### **Evaluation Steps:**
1. Identify the type of {self.domain_mappings['context']} or experience the {self.domain_mappings['person']} seeks based on the query.  
2. Assess the {self.domain_mappings['singular']}'s overall strength as a {self.domain_mappings['context']} (e.g., general popularity and appeal).  
3. Evaluate how well the {self.domain_mappings['singular']} matches the query's intent based on your internal knowledge.  
4. Cross-check the provided {self.domain_mappings['singular']} description for supporting details.  
5. Assign a single final score.  

### **Additional Instructions:**
- Be strict in your rating — only top-tier {self.domain_mappings['plural']} should receive a score of 3. Most {self.domain_mappings['plural']} should be rated 0 or 1 unless they have clear relevance and strong alignment with the query.  
- Consider the overall popularity of the {self.domain_mappings['singular']} as a {self.domain_mappings['context']} — some {self.domain_mappings['plural']} have a general popularity advantage over others for most queries.  
- The final score should reflect both the {self.domain_mappings['singular']}'s specific relevance to the query and its general strength as a {self.domain_mappings['context']}.

Your response must be a single integer between 0-3. Output only the score number and nothing else.
'''
        
        return {
            "UMBRELA_PROMPT_3": umbrella_prompt_3
        }

def get_default_config() -> PipelineConfig:
    """Returns the default configuration."""
    return PipelineConfig()
