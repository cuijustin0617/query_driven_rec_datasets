import json
from collections import defaultdict

# Dictionary to store city -> restaurant count
city_restaurants = defaultdict(int)

# Read the file line by line
with open('yelp_academic_dataset_business.json', 'r', encoding='utf-8') as f:
    for line in f:
        # Parse JSON object
        business = json.loads(line.strip())
        
        # Check if 'Restaurants' is in the categories
        if business['categories'] and 'Restaurants' in business['categories'].split(', '):
            city_restaurants[business['city']] += 1

# Print results and write top 15 to file
with open('top_15_restaurant_cities.txt', 'w', encoding='utf-8') as outfile:
    for i, (city, count) in enumerate(sorted(city_restaurants.items(), key=lambda x: (-x[1], x[0])), 1):
        print(f"{city}: {count} restaurants")
        
        # Write only top 15 to file
        if i <= 15:
            outfile.write(f"{city}: {count} restaurants\n")