import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def calculate_average_scores(directory, query_list):
    """Calculate average scores for specific metrics in a method directory,
    only for the queries in the provided query list."""
    metrics = {}
    
    # Track queries found in the metric files
    found_queries = set()
    
    # Only include these specific metrics
    allowed_metrics = ['recall_at10', 'recall_at30', 'map_at10', 'map_at30', 'rprecision']
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            metric_name = filename.split('.')[0]
            # Only process allowed metrics
            if metric_name in allowed_metrics:
                with open(os.path.join(directory, filename)) as f:
                    data = json.load(f)
                    
                    # Filter data to only include queries from the query_list
                    filtered_data = {k: v for k, v in data.items() if k in query_list}
                    
                    # Add found queries to our tracking set
                    found_queries.update(filtered_data.keys())
                    
                    # Calculate average if we have any matching queries
                    if filtered_data:
                        avg_score = sum(filtered_data.values()) / len(filtered_data)
                        metrics[metric_name] = round(avg_score, 3)  # Round to 3 decimal places
                    else:
                        metrics[metric_name] = 0.0  # No matching queries
    
    # Return both the metrics and the set of found queries
    return metrics, found_queries

def highlight_max(s):
    """Highlight the maximum in a Series."""
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]

def save_as_image(df, output_name="metric_averages"):
    """Save the styled DataFrame as an image with winners highlighted."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare cell colors: highlight max in each column
    cell_colors = []
    for col in df.columns:
        max_val = df[col].max()
        colors = ['lightgreen' if val == max_val else 'white' for val in df[col]]
        cell_colors.append(colors)
    cell_colors = np.array(cell_colors).T  # Transpose to match cellText shape
    
    # Round values to 3 decimal places for display
    cell_text = df.round(3).values
    
    table = ax.table(cellText=cell_text,
                     colLabels=df.columns,
                     rowLabels=df.index,
                     cellLoc='center',
                     loc='center',
                     cellColours=cell_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.savefig(f"{output_name}.png", bbox_inches='tight', dpi=300)
    plt.close()

def read_queries_from_file(file_path):
    """Read queries from a text file, each query on a separate line."""
    with open(file_path, 'r') as f:
        # Strip whitespace from each line and filter out empty lines
        queries = [line.strip() for line in f if line.strip()]
    return queries

def main():
    parser = argparse.ArgumentParser(description='Calculate metric averages for specific queries')
    parser.add_argument('--domain', type=str, default='travel', choices=['travel', 'hotel', 'restaurant'],
                        help='Domain to analyze (travel, hotel, or restaurant)')
    parser.add_argument('--model_type', type=str, default='e5', choices=['e5', 'minilm'],
                        help='Model type to use (e5 or minilm)')
    parser.add_argument('--query_file', type=str, required=True,
                        help='Path to a text file containing queries (one per line)')
    args = parser.parse_args()
    
    domain = args.domain
    model_type = args.model_type
    query_file = args.query_file
    results_dir = f"final_{model_type}"
    
    # Read queries from the file
    try:
        queries = read_queries_from_file(query_file)
        print(f"Loaded {len(queries)} queries from {query_file}")
    except FileNotFoundError:
        print(f"Error: Query file '{query_file}' not found")
        return
    
    methods = ['eqr', 'q2d', 'q2e', 'none']
    output_name = f"{model_type}_{domain}_query_filtered_metric_averages"
     
    results = {}
    # Keep track of all queries found across all methods/cities
    all_found_queries = set()
    
    if domain == 'restaurant':
        # Handle restaurant domain which has two cities (nor and phi)
        cities = ['nor', 'phi']
        for method in methods:
            # Store metrics for each city
            city_metrics = {}
            city_found_queries = set()
            
            for city in cities:
                # Match the actual directory structure
                method_dir = os.path.join("pattern_analyzing", results_dir, domain, city, f"{city}_{method}")
                try:
                    city_metrics[city], found_queries = calculate_average_scores(method_dir, queries)
                    city_found_queries.update(found_queries)
                except FileNotFoundError:
                    print(f"Warning: Directory not found: {method_dir}")
                    city_metrics[city] = {}
            
            # Update the set of all found queries
            all_found_queries.update(city_found_queries)
            
            # Average the metrics across cities
            combined_metrics = {}
            # Get all unique metric names from both cities
            all_metrics = set()
            for city_data in city_metrics.values():
                all_metrics.update(city_data.keys())
                
            # Calculate average for each metric across cities
            for metric in all_metrics:
                values = [city_data.get(metric, 0) for city_data in city_metrics.values() 
                          if metric in city_data]
                if values:
                    combined_metrics[metric] = round(sum(values) / len(values), 3)
            
            results[method] = combined_metrics
    elif domain == 'hotel':
        # Handle hotel domain which has four cities (chicago, london, montreal, nyc)
        cities = ['chicago', 'london', 'montreal', 'nyc']
        for method in methods:
            # Store metrics for each city
            city_metrics = {}
            city_found_queries = set()
            
            for city in cities:
                # Match the actual directory structure
                method_dir = os.path.join("pattern_analyzing", results_dir, domain, city, f"{city}_{method}")
                try:
                    city_metrics[city], found_queries = calculate_average_scores(method_dir, queries)
                    city_found_queries.update(found_queries)
                except FileNotFoundError:
                    print(f"Warning: Directory not found: {method_dir}")
                    city_metrics[city] = {}
            
            # Update the set of all found queries
            all_found_queries.update(city_found_queries)
            
            # Average the metrics across cities
            combined_metrics = {}
            # Get all unique metric names from all cities
            all_metrics = set()
            for city_data in city_metrics.values():
                all_metrics.update(city_data.keys())
                
            # Calculate average for each metric across cities
            for metric in all_metrics:
                values = [city_data.get(metric, 0) for city_data in city_metrics.values() 
                          if metric in city_data]
                if values:
                    combined_metrics[metric] = round(sum(values) / len(values), 3)
            
            results[method] = combined_metrics
    else:
        # Handle travel domain
        for method in methods:
            method_dir = os.path.join("pattern_analyzing", results_dir, domain, f"{domain}_{method}")
            try:
                results[method], found_queries = calculate_average_scores(method_dir, queries)
                all_found_queries.update(found_queries)
            except FileNotFoundError:
                print(f"Warning: Directory not found: {method_dir}")
                results[method] = {}
    
    # Verify that all queries were found
    missing_queries = set(queries) - all_found_queries
    if missing_queries:
        print(f"WARNING: {len(missing_queries)} queries from the file were not found in any metric files:")
        for query in missing_queries:
            print(f"  - {query}")
    else:
        print("All queries from the file were found in the metric results.")
    
    # Create DataFrame and transpose to have methods as rows and metrics as columns
    df = pd.DataFrame(results).T
    
    # Round all values to 3 decimal places
    df = df.round(3)
    
    # Apply styling to highlight the max in each column
    styled_df = df.style.apply(highlight_max)
    
    # Save as image with highlighted winners
    save_as_image(df, output_name)
    
    # Print to console with winners marked and rounded values
    print(f"Results saved to '{output_name}.png'")
    print(f"\nAverage Scores for {domain} domain with {model_type} model (filtered to specified queries, winners highlighted):")
    print(styled_df.to_string())

if __name__ == "__main__":
    main()
