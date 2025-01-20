import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_restaurant_reviews():
    # Read the CSV file
    df = pd.read_csv('philadelphia_restaurants.csv')
    
    # Create a figure with larger size
    plt.figure(figsize=(12, 6))
    
    # Create histogram of review counts with adjusted bin width
    sns.histplot(data=df, x='review_count', bins=200, binwidth=50)
    
    # Customize the plot
    plt.title('Distribution of Restaurant Review Counts in Philadelphia')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Number of Restaurants')
    
    # Set x-axis ticks at intervals of 50
    max_reviews = df['review_count'].max()
    plt.xticks(range(0, int(max_reviews) + 50, 50))
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Calculate and print key statistics
    print("\nReview Count Statistics:")
    print(f"Total restaurants: {len(df)}")
    print(f"Average reviews per restaurant: {df['review_count'].mean():.1f}")
    print(f"Median reviews per restaurant: {df['review_count'].median():.1f}")
    print(f"Maximum reviews: {df['review_count'].max()}")
    print(f"Minimum reviews: {df['review_count'].min()}")
    
    # Get top 10 restaurants by review count
    print("\nTop 10 Most Reviewed Restaurants:")
    top_10 = df.nlargest(10, 'review_count')[['name', 'review_count']]
    for idx, row in top_10.iterrows():
        print(f"{row['name']}: {row['review_count']} reviews")
    
    # Save the plot
    plt.savefig('restaurant_reviews_distribution.png')
    plt.close()

if __name__ == "__main__":
    analyze_restaurant_reviews()
