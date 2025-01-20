import pandas as pd
import json
import os

# Read the CSV and JSON files
restaurants_df = pd.read_csv('data/30_phi_restaurants_with_over_50.csv')
reviews_df = pd.read_json('data/philadelphia_reviews.json', lines=True)
reviews = reviews_df.to_dict('records')

# Create docs directory if it doesn't exist
if not os.path.exists('docs'):
    os.makedirs('docs')

# Create a dictionary mapping business_ids to restaurant names for faster lookup
restaurant_names = dict(zip(restaurants_df['business_id'], restaurants_df['name']))

# Dictionary to keep track of review counts per restaurant
review_counts = {business_id: 0 for business_id in restaurant_names}
completed_restaurants = 0

# Process each review
for review in reviews:
    business_id = review['business_id']
    
    # Check if this review is for one of our target restaurants
    if business_id in restaurant_names and review_counts[business_id] < 50:
        restaurant_name = restaurant_names[business_id]
        # Create safe filename by replacing problematic characters
        safe_filename = "docs/" + restaurant_name.replace('/', '_').replace(chr(92), '_') + ".txt"
        
        # Append review text to the restaurant's file
        with open(safe_filename, 'a', encoding='utf-8') as f:
            # Strip any extra whitespace and add a single newline
            f.write(review['text'].strip())
        
        # Update counts
        review_counts[business_id] += 1
        if review_counts[business_id] == 50:
            completed_restaurants += 1
            print(f"Completed {completed_restaurants}/30 restaurants (Just finished: {restaurant_name})")
        
        # Break if all restaurants have 50 reviews
        if completed_restaurants == 30:
            print("All restaurants have reached 50 reviews!")
            break
