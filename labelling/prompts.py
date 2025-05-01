"""
Prompts for binary relevance labeling across different domains.
"""

def get_binary_umbrella_prompt(domain_mappings):
    """
    Returns a binary (True/False) relevance prompt that works across different domains.
    
    Args:
        domain_mappings: Dictionary containing domain-specific terminology
        
    Returns:
        String prompt template with placeholders for query, entity_name, and document
    """
    
    binary_umbrella_prompt = f'''
You are tasked with evaluating whether **{{entity_name}}** is a suitable {domain_mappings['context']} for a {domain_mappings['person']} based on the provided query and {domain_mappings['singular']} description. Determine if the {domain_mappings['singular']} is relevant (**True**) or not relevant (**False**) to the query.

### **Scoring Guidelines:**
- **False** = The {domain_mappings['singular']} is irrelevant to the query, contradicts the user's intent, or provides little value for the {domain_mappings['person']}.
- **True** = The {domain_mappings['singular']} matches the query goal and is suitable as a {domain_mappings['context']} for the {domain_mappings['person']}'s intent.

### **Input:**
- **Query:** {{query}}  
- **{domain_mappings['csv_entity_header']}:** {{entity_name}}  
- **{domain_mappings['description_term']}:** {{document}}  

### **Evaluation Steps:**
1. Identify the type of {domain_mappings['context']} or experience the {domain_mappings['person']} seeks based on the query.  
2. Assess the {domain_mappings['singular']}'s overall strength as a {domain_mappings['context']} (e.g., general popularity and appeal).  
3. Evaluate how well the {domain_mappings['singular']} matches the query's intent based on your internal knowledge.  
4. Cross-check the provided {domain_mappings['singular']} description for supporting details.  
5. Make a binary decision (True/False).  

### **Additional Instructions:**
- Be strict in your rating — only {domain_mappings['plural']} with clear relevance and strong alignment with the query should receive True.  
- Consider the overall popularity of the {domain_mappings['singular']} as a {domain_mappings['context']} — some {domain_mappings['plural']} have a general popularity advantage over others for most queries.  
- The final decision should reflect both the {domain_mappings['singular']}'s specific relevance to the query and its general strength as a {domain_mappings['context']}.

Your response must be a single True or False. Output only the decision and nothing else.
'''
    
    return binary_umbrella_prompt 