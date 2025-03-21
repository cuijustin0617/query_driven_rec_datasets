import os
import csv
from openai import OpenAI
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Union
import json

# Define the schema for the response
class Score(BaseModel):
    score: int

response_format = {
    "type": "json_schema",
    "json_schema":{
    "name": "output_schema",
    "schema": Score.model_json_schema()
    }
}

### API Key
API_KEY = os.getenv("OPENAI_API_KEY")

class ChatGPT:
    def __init__(self, model_name: str = "gpt-4o", api_key: str = API_KEY):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
    def generate(self, message: list[dict], temperature: float = 0.0, response_format: Optional[dict] = None) -> Union[str, None]:
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=message,
                temperature=temperature,
                response_format=response_format
            )
        except Exception as e:
            print(e)
            return None
        return response.choices[0].message.content


UMBRELA_PROMPT = '''
Given a query and a document about a city, provide a score from 0 to 3, where higher scores mean the city is more relevant and helpful for a traveler with this query.

Scoring Guidelines:
0 = The city has little to do with the query.
1 = The city is loosely related to the query, but offers little or no useful information for a traveler.
2 = The city has some relevant features, but they are unclear, not emphasized, or mixed with unrelated details.
3 = The city clearly meets or supports the query goal in a practical or helpful way.

Instructions:
- Assign 3 if the city fulfills the query in any meaningful way, even with extra info.
- Assign 2 if the city shows potential to satisfy the query but lacks key details, or the relevance is partial, vague, or not well-supported.
- Assign 1 if the city is not very helpful for the query.
- Assign 0 only if the city has little relevance to the query.

Query: {query}
City Info: {document}

Split this problem into steps:
Consider the underlying intent of the search.
Measure how well the content matches a likely intent of the query (M).
Measure how trustworthy the passage is (T).
Consider the aspects above and the relative importance of each, and decide on a final score (O). Final score must be an integer value only.

Do not provide any code in result. Provide each score in the format of: ##final score: score without providing any reasoning.
'''

UMBRELA_PROMPT_2 ='''
Given a query and a description of a city, assign a score from 0 to 3 based on how suitable this city is as a travel destination for someone with that interest.
Scoring:
0 = The city is completely unrelated to the query.
1 = The city is weakly related, but is clearly not a good match for what the traveler is looking for.
2 = The city has some relevant aspects, but overall is not a strong or clear match.
3 = The city is a reasonably good or strong match for what the traveler is looking for — even if it's only partially relevant.
Important Instructions:
- Assign a score of 3 if the city has any practical or relevant qualities that could make it a suitable destination for the query.
- Assign 1 only if the city clearly does not fit what the traveler likely wants.
- Assign 0 only if the city has nothing to do with the query.
Query: {query}
City Description: {document}
Steps:
- Consider what kind of destination a traveler is seeking from the query.
- Judge how well the city fits that intent or interest.
- Make a final decision.
Final score must be a single integer only.
Output format: ##final score: [score]
'''

UMBRELA_PROMPT_3 = '''
You are tasked with evaluating how suitable **{city}** is as a travel destination for a traveler based on the provided query and city description. Assign a score from **0 to 3** primarily based on your internal knowledge of the city and the supportive evidence from the provided text.

### **Scoring Guidelines:**
- **0** = The city is irrelevant to the query or contradicts the user’s intent.  
- **1** = The city is loosely related to the query but provides little value or relevance for the traveler.  
- **2** = The city has some relevant features to the query, but it is not an ideal fit for the traveler’s intent.  
- **3** = The city clearly matches the query goal and is highly suitable as a travel destination.  

### **Input:**
- **Query:** {query}  
- **City:** {city}  
- **City Description:** {document}  

### **Evaluation Steps:**
1. Identify the type of destination or experience the traveler seeks based on the query.  
2. Assess the city’s overall strength as a travel destination (e.g., general popularity and tourist appeal).  
3. Evaluate how well the city matches the query’s intent based on your internal knowledge.  
4. Cross-check the provided city description for supporting details.  
5. Assign a single final score.  

### **Additional Instructions:**
- Be strict in your rating — only top-tier cities should receive a score of 3. Most cities should be rated 0 or 1 unless they have clear relevance and strong alignment with the query.  
- Consider the overall popularity of the city as a travel destination — some cities have a general popularity advantage over others for most queries.  
- The final score should reflect both the city’s specific relevance to the query and its general strength as a travel destination.  
'''


def process_json(input_json_path, output_csv_path, gpt_client):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    total_queries = len(data)
    query_counter = 0
    

    for query, city_data in data.items():
        city_counter = 0
        query_counter += 1
        print(f"\nProcessing query {query_counter}/{total_queries}: {query}")
        for city, documents in city_data.items():
            city_counter += 1
            print(f"  - Processing city: {city} with {len(documents)} documents")
            #concatenate list of passages in a document into a string
            document = "\n".join(documents)
            
            prompt = UMBRELA_PROMPT_3.format(query=query, document=document, city=city)  ## change prompt

            messages = [
                {"role": "system", "content": "You are a helpful assistant that evaluates relevance scores for travel-related query-document pairs."},
                {"role": "user", "content": prompt},
            ]

            response_text = gpt_client.generate(messages, temperature=0.0, response_format=response_format)
            print(f"    Response: {response_text}")

            if response_text is None:
                print(f"    Failed to get response for city: {city}")
                final_score = "Error"
            else:
                try:
                    scores_obj = Score.model_validate_json(response_text)
                    final_score = scores_obj.score
                except Exception as e:
                    print(f"    Parsing error for city {city}: {e}")
                    final_score = "ParsingError"
            
            results.append([query, city, final_score])
        #     if city_counter == 20:
        #         break
        # if query_counter == 2:
        #     break

    # Write results to CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Query', 'City', 'Relevance Score'])
        writer.writerows(results)

    print(f"\nProcessing completed. Results saved to {output_csv_path}")


def main():
    gpt_client = ChatGPT(api_key=API_KEY)

    input_json_path = "scripts/labeling/gpt-labelling/sample_5_dense_result.json"
    output_csv_path = "scripts/labeling/gpt-labelling/labeled_documents.csv"

    process_json(input_json_path, output_csv_path, gpt_client)

if __name__ == "__main__":
    main()