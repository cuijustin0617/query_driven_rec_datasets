import csv
import json

# First, create a set of new orleans business IDs
business_ids = set()

# Read the new orleans restaurants CSV
with open('data/new_orleans/new_orleans_restaurants.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        business_ids.add(row['business_id'])

# Read the review.json file and filter for new orleans businesses
reviews = []

with open('original/yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
    for line in f:
        review = json.loads(line.strip())
        if review['business_id'] in business_ids:
            reviews.append(review)

# Write the filtered reviews to a new JSON file
with open('new_orleans_reviews.json', 'w', encoding='utf-8') as f:
    for review in reviews:
        json.dump(review, f)
        f.write('\n')

print(f"Filtered {len(reviews)} reviews for New Orleans restaurants")
