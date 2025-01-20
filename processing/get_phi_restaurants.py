import pandas as pd

# Read the data (adjust the file path as needed)
df = pd.read_json('yelp_academic_dataset_business.json', lines=True)

# Filter for Philadelphia restaurants
philly_restaurants = df[
    (df['city'].str.lower() == 'philadelphia') & 
    (df['categories'].str.contains('Restaurants', case=False, na=False))
]

# Get the count
total_count = len(philly_restaurants)

# Print results
print(f"Total Philadelphia restaurants found: {total_count}")

# Optionally save the filtered data
philly_restaurants.to_csv('philadelphia_restaurants.csv', index=False)
