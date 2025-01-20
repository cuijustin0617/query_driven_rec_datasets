import csv

def filter_high_reviews(input_file, output_file, min_reviews=50):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Filter and write rows
        for row in reader:
            if int(row['review_count']) > min_reviews:
                writer.writerow(row)

# Usage
filter_high_reviews('philadelphia_restaurants.csv', 'philadelphia_restaurants_filtered.csv')
