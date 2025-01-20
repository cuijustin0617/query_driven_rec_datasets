import csv
import json

# First, create a set of Philadelphia business IDs
philly_business_ids = set()

# Read the Philadelphia restaurants CSV
with open('philadelphia_restaurants.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        philly_business_ids.add(row['business_id'])

# Read the review.json file and filter for Philadelphia businesses
philly_reviews = []

with open('yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
    for line in f:
        review = json.loads(line.strip())
        if review['business_id'] in philly_business_ids:
            philly_reviews.append(review)

# Write the filtered reviews to a new JSON file
with open('philadelphia_reviews.json', 'w', encoding='utf-8') as f:
    for review in philly_reviews:
        json.dump(review, f)
        f.write('\n')

print(f"Filtered {len(philly_reviews)} reviews for Philadelphia restaurants")
