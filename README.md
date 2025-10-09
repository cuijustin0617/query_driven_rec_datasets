<div align="center">
  
# Query-Driven Recommendation Datasets

[![arXiv](https://img.shields.io/badge/arXiv-B31B1B?logo=arxiv&logoColor=fff&style=flat)](https://www.arxiv.org/pdf/2510.02656) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-FFD400?logo=huggingface&logoColor=fff&style=flat)](https://huggingface.co/datasets/cuijustin0617/NLRec)

</div>

We provide three query-driven recommendation datasets designed to evaluate systems under challenging conditions where 
  
1. user intent is implicitly expressed through broad or indirect queries 
  
2. items are described through multiple diverse textual sources. 
  
Each dataset contains 100 natural language queries, ground truth relevance labels, and original corpus of items for reference.



## Datasets

| Dataset | Cities/Categories | Corpus Size | Queries |
|---------|------------------|-------------|---------|
| Yelp_Restaurant | New Orleans (nor), Philadelphia (phi) | 1,152 restaurants (nor: 515, phi: 637) | 100 |
| TripAdvisor_Hotel | New York City, Chicago, London, Montreal | 586 hotels (nyc: 182, chicago: 74, london: 266, montreal: 64) | 100 |
| Traveldest | /| 775 cities | 100 |

### Yelp_Restaurant
A dataset of restaurant recommendations based on Yelp reviews:
- 100 natural language queries in `queries_restaurant.txt`
- Ground truth files:
  - New Orleans: `gt_nor.json`
  - Philadelphia: `gt_phi.json`
- Corpus of 1,152 restaurants organized by city (nor, phi)

### TripAdvisor_Hotel
A dataset of hotel recommendations based on TripAdvisor reviews:
- 100 natural language queries in `queries_hotel.txt`
- Ground truth files for each city:
  - New York City: `gt_nyc.json`
  - Chicago: `gt_chicago.json`
  - London: `gt_london.json`
  - Montreal: `gt_montreal.json`
- Corpus of 586 hotels organized by city (nyc, chicago, london, montreal)

### Traveldest
A dataset of travel destination recommendations based on WikiVoyage pages:
- 100 natural language queries in `queries_travel.txt`
- Ground truth for various cities in `gt_cities.json`
- Corpus of 775 cities with detailed descriptions from WikiVoyage

## Labeling Pipeline

The repository also includes the complete labeling pipeline used to create these datasets. The pipeline uses Large Language Models to generate binary relevance labels for query-document pairs.

For detailed instructions on how to use the labeling pipeline, including configuration options, input formats, and output explanations, please **refer to the dedicated `README.md`** in the `labelling` folder. 
