import os
import csv
from openai import OpenAI
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Union
import json

# Define the schema for the response
class Scores(BaseModel):
    scores: dict[str, int]

response_format = {
    "type": "json_schema",
    "json_schema":{
    "name": "output_schema",
    "schema": Scores.model_json_schema()
    }
}

### API Key
API_KEY = os.getenv("OPENAI_API_KEY")

class ChatGPT:
    def __init__(self, model_name: str = "gpt-4o-2024-11-20", api_key: str = API_KEY):
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
            
            prompt = UMBRELA_PROMPT_2.format(query=query, document=document)  ## change prompt

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
                    scores_obj = Scores.model_validate_json(response_text)
                    final_score = scores_obj.scores.get("final score") or scores_obj.scores.get("final_score", "Missing")
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