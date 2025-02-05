import pandas as pd

# Read the data (adjust the file path as needed)
df = pd.read_json('original/yelp_academic_dataset_business.json', lines=True)

# Filter for New Orleans restaurants
restaurants = df[
    (df['city'].str.lower() == 'new orleans') & 
    (df['categories'].str.contains('Restaurants', case=False, na=False))
]

# Get the count
total_count = len(restaurants)

# Print results
print(f"Total New Orleans restaurants found: {total_count}")

# Optionally save the filtered data
restaurants.to_csv('new_orleans_restaurants.csv', index=False)

