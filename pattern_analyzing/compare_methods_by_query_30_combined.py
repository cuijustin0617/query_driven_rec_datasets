import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import defaultdict

def load_query_metrics(directory):
    """Load all metric values for each query in a method directory."""
    query_metrics = defaultdict(dict)
    allowed_metrics = ['map_at10', 'map_at30', 'recall_at10', 'recall_at30', 'rprecision']
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            metric_name = filename.split('.')[0]
            if metric_name not in allowed_metrics:
                continue
            with open(os.path.join(directory, filename)) as f:
                data = json.load(f)
                for query_id, score in data.items():
                    query_metrics[query_id][metric_name] = score
    return query_metrics

def load_city_averaged_metrics(domain, method, embedding_model, cities=None):
    """Load and average metrics across cities for a domain/method combination for a specific embedding model."""
    if domain not in ['restaurant', 'hotel'] or cities is None:
        # For domains without cities or if no cities specified
        method_dir = os.path.join(f"pattern_analyzing/final_{embedding_model}", domain, f"{domain}_{method}")
        try:
            return load_query_metrics(method_dir)
        except FileNotFoundError as e:
            print(f"Warning: Directory not found: {e}")
            return defaultdict(dict)
    
    # For domains with cities, average across cities
    query_metrics = defaultdict(lambda: defaultdict(list))
    
    for city in cities:
        method_dir = os.path.join(f"pattern_analyzing/final_{embedding_model}", domain, city, f"{city}_{method}")
        try:
            city_metrics = load_query_metrics(method_dir)
            
            # Collect metrics for each query/metric across cities
            for query_id, metrics in city_metrics.items():
                for metric_name, value in metrics.items():
                    query_metrics[query_id][metric_name].append(value)
        except FileNotFoundError as e:
            print(f"Warning: Directory not found: {e}")
    
    # Average the metrics across cities
    result = defaultdict(dict)
    for query_id, metrics in query_metrics.items():
        for metric_name, values in metrics.items():
            if values:  # Only if we have values
                result[query_id][metric_name] = sum(values) / len(values)
    
    return result

def load_combined_metrics(domain, method, cities=None):
    """Load and average metrics from both embedding models (E5 and MiniLM)."""
    # Load metrics from both embedding models
    minilm_data = load_city_averaged_metrics(domain, method, "minilm", cities)
    e5_data = load_city_averaged_metrics(domain, method, "e5", cities)
    
    # Combine and average metrics from both embedding models
    combined_data = defaultdict(dict)
    
    # Get all query IDs from both datasets
    all_query_ids = set(minilm_data.keys()) | set(e5_data.keys())
    
    for query_id in all_query_ids:
        minilm_metrics = minilm_data.get(query_id, {})
        e5_metrics = e5_data.get(query_id, {})
        
        # Get all metrics present in either dataset
        all_metrics = set(minilm_metrics.keys()) | set(e5_metrics.keys())
        
        for metric in all_metrics:
            # Get metric values from both sources, defaulting to 0 if not present
            minilm_value = minilm_metrics.get(metric, 0)
            e5_value = e5_metrics.get(metric, 0)
            
            # Only include in combined result if the metric exists in both datasets
            if metric in minilm_metrics and metric in e5_metrics:
                # Calculate the average
                combined_data[query_id][metric] = (minilm_value + e5_value) / 2
    
    return combined_data, minilm_data, e5_data

def calculate_method_differences(method1_data, method2_data, selected_metrics=None):
    """Calculate average metric difference between two methods for each query."""
    diff_by_query = {}
    
    # Find queries that exist in both methods
    common_queries = set(method1_data.keys()) & set(method2_data.keys())
    
    for query_id in common_queries:
        # Get metrics that exist for this query in both methods
        method1_metrics = method1_data[query_id]
        method2_metrics = method2_data[query_id]
        
        common_metrics = set(method1_metrics.keys()) & set(method2_metrics.keys())
        
        # If selected_metrics is provided, only use those that are common
        if selected_metrics:
            common_metrics = common_metrics & set(selected_metrics)
        
        if not common_metrics:
            continue
            
        # Calculate difference for each metric
        diffs = [method1_metrics[metric] - method2_metrics[metric] for metric in common_metrics]
        # Average difference across all metrics
        avg_diff = sum(diffs) / len(diffs)
        diff_by_query[query_id] = avg_diff
    
    return diff_by_query

def get_top_queries(diff_by_query, top_n=100):
    """Get the top N queries where method1 outperforms method2 by the largest margin."""
    # Sort queries by difference (largest positive difference first)
    # This selects queries where method1 beats method2 the most
    sorted_queries = sorted(diff_by_query.items(), key=lambda x: x[1], reverse=True)
    # Take top N
    top_queries = [query_id for query_id, diff in sorted_queries[:top_n]]
    return top_queries

def filter_metrics_by_queries(directory, selected_queries, selected_metrics=None):
    """Calculate average scores for metrics but only using selected queries."""
    metrics = {}
    allowed_metrics = ['map_at10', 'map_at30', 'recall_at10', 'recall_at30', 'rprecision']
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            metric_name = filename.split('.')[0]
            
            # Skip metrics not in allowed list
            if metric_name not in allowed_metrics:
                continue
                
            # Skip metrics not in selected_metrics if it's provided
            if selected_metrics and metric_name not in selected_metrics:
                continue
                
            with open(os.path.join(directory, filename)) as f:
                data = json.load(f)
                # Filter data to only include selected queries
                filtered_data = {q: v for q, v in data.items() if q in selected_queries}
                if filtered_data:
                    avg_score = sum(filtered_data.values()) / len(filtered_data)
                    metrics[metric_name] = round(avg_score, 3)
                else:
                    metrics[metric_name] = 0
    return metrics

def filter_and_average_metrics_by_queries(domain, method, selected_queries, embedding_model, cities=None, selected_metrics=None):
    """Calculate average scores for metrics on selected queries for a specific embedding model, averaging across cities if needed."""
    if domain not in ['restaurant', 'hotel'] or cities is None:
        # For domains without cities or if no cities specified
        method_dir = os.path.join(f"pattern_analyzing/final_{embedding_model}", domain, f"{domain}_{method}")
        try:
            return filter_metrics_by_queries(method_dir, selected_queries, selected_metrics)
        except FileNotFoundError:
            print(f"Warning: Directory not found: {method_dir}")
            return {}
    
    # For domains with multiple cities
    city_metrics = {}
    
    for city in cities:
        method_dir = os.path.join(f"pattern_analyzing/final_{embedding_model}", domain, city, f"{city}_{method}")
        try:
            city_metrics[city] = filter_metrics_by_queries(method_dir, selected_queries, selected_metrics)
        except FileNotFoundError:
            print(f"Warning: Directory not found: {method_dir}")
            city_metrics[city] = {}
    
    # Average metrics across cities
    combined_metrics = {}
    all_metrics = set()
    for city_data in city_metrics.values():
        all_metrics.update(city_data.keys())
    
    # Filter all_metrics to only include our allowed metrics
    allowed_metrics = ['map_at10', 'map_at30', 'recall_at10', 'recall_at30', 'rprecision']
    all_metrics = all_metrics & set(allowed_metrics)
    
    for metric in all_metrics:
        values = [city_data.get(metric, 0) for city_data in city_metrics.values() 
                 if metric in city_data]
        if values:
            combined_metrics[metric] = round(sum(values) / len(values), 3)
    
    return combined_metrics

def highlight_max(s):
    """Highlight the maximum in a Series."""
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]

def save_as_image(minilm_df, e5_df, output_name="method_comparison_top_queries"):
    """Save two styled DataFrames as an image with winners highlighted."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Set titles
    ax1.set_title("MiniLM Results", fontsize=14)
    ax2.set_title("E5 Results", fontsize=14)
    
    # Turn off axes
    ax1.axis('tight')
    ax1.axis('off')
    ax2.axis('tight')
    ax2.axis('off')
    
    # Create tables for MiniLM
    cell_colors1 = []
    for col in minilm_df.columns:
        max_val = minilm_df[col].max()
        colors = ['lightgreen' if val == max_val else 'white' for val in minilm_df[col]]
        cell_colors1.append(colors)
    cell_colors1 = np.array(cell_colors1).T
    
    cell_text1 = minilm_df.round(3).values
    
    table1 = ax1.table(cellText=cell_text1,
                      colLabels=minilm_df.columns,
                      rowLabels=minilm_df.index,
                      cellLoc='center',
                      loc='center',
                      cellColours=cell_colors1)
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1.2, 1.2)
    
    # Create tables for E5
    cell_colors2 = []
    for col in e5_df.columns:
        max_val = e5_df[col].max()
        colors = ['lightgreen' if val == max_val else 'white' for val in e5_df[col]]
        cell_colors2.append(colors)
    cell_colors2 = np.array(cell_colors2).T
    
    cell_text2 = e5_df.round(3).values
    
    table2 = ax2.table(cellText=cell_text2,
                      colLabels=e5_df.columns,
                      rowLabels=e5_df.index,
                      cellLoc='center',
                      loc='center',
                      cellColours=cell_colors2)
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 1.2)
    
    plt.tight_layout()
    plt.savefig(f"{output_name}.png", bbox_inches='tight', dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare methods and calculate metrics for top queries with largest differences')
    parser.add_argument('--domain', type=str, default='travel', choices=['travel', 'hotel', 'restaurant'],
                        help='Domain to analyze (travel, hotel, or restaurant)')
    parser.add_argument('--method1', type=str, default='eqr', choices=['eqr', 'q2d', 'q2e', 'none'],
                        help='First method to compare')
    parser.add_argument('--method2', type=str, default='q2e', choices=['eqr', 'q2d', 'q2e', 'none'],
                        help='Second method to compare')
    parser.add_argument('--top_n', type=int, default=100, 
                        help='Number of top queries to select based on difference')
    # Define default metrics
    default_metrics = ['map_at10', 'map_at30', 'recall_at10', 'recall_at30', 'rprecision']
    parser.add_argument('--metrics', type=str, nargs='+', default=default_metrics,
                        help='Specific metrics to include when calculating differences for ranking queries')
    args = parser.parse_args()
    
    domain = args.domain
    method1 = args.method1
    method2 = args.method2
    top_n = args.top_n
    selected_metrics = args.metrics
    
    # Ensure selected metrics are only from our allowed set
    allowed_metrics = ['map_at10', 'map_at30', 'recall_at10', 'recall_at30', 'rprecision']
    selected_metrics = [m for m in selected_metrics if m in allowed_metrics]
    
    methods = ['eqr', 'q2d', 'q2e', 'none']
    output_name = f"30_{domain}_{method1}_vs_{method2}_top{top_n}"
    
    # If specific metrics were selected, add them to the output filename
    if selected_metrics:
        metrics_str = '_'.join(selected_metrics)
        output_name = f"{output_name}_sortby_{metrics_str}"
        print(f"Using selected metrics for sorting queries: {', '.join(selected_metrics)}")
    
    # Define cities for domains with city-based structure
    cities = None
    if domain == 'restaurant':
        cities = ['nor', 'phi']
    elif domain == 'hotel':
        cities = ['chicago', 'london', 'montreal', 'nyc']
    
    # Load data for method1 and method2, averaging across cities and embedding models
    method1_combined, method1_minilm, method1_e5 = load_combined_metrics(domain, method1, cities)
    method2_combined, method2_minilm, method2_e5 = load_combined_metrics(domain, method2, cities)
    
    # Calculate differences between methods for each query using only selected metrics
    # Use the combined (averaged) data for selecting top queries
    diff_by_query = calculate_method_differences(method1_combined, method2_combined, selected_metrics)
    
    if not diff_by_query:
        print(f"No common queries found between {method1} and {method2} for {domain} domain")
        return
    
    # Get top N queries with largest differences
    top_queries = get_top_queries(diff_by_query, top_n)
    
    print(f"Top {len(top_queries)} queries where {method1} outperforms {method2} by the largest margin:")
    for i, query_id in enumerate(top_queries):
        print(f"{query_id}")
    
    # Calculate metrics for all methods using only the top queries
    minilm_results = {}
    e5_results = {}
    
    for method in methods:
        # Calculate metrics for MiniLM embedding model
        minilm_results[method] = filter_and_average_metrics_by_queries(
            domain, method, top_queries, "minilm", cities
        )
        
        # Calculate metrics for E5 embedding model
        e5_results[method] = filter_and_average_metrics_by_queries(
            domain, method, top_queries, "e5", cities
        )
    
    # Create DataFrames and transpose to have methods as rows and metrics as columns
    minilm_df = pd.DataFrame(minilm_results).T
    e5_df = pd.DataFrame(e5_results).T
    
    # Round all values to 3 decimal places
    minilm_df = minilm_df.round(3)
    e5_df = e5_df.round(3)
    
    # Apply styling to highlight the max in each column
    minilm_styled_df = minilm_df.style.apply(highlight_max)
    e5_styled_df = e5_df.style.apply(highlight_max)
    
    # Save as image with highlighted winners
    save_as_image(minilm_df, e5_df, output_name)
    
    metric_desc = f" (queries sorted using {', '.join(selected_metrics)})" if selected_metrics else ""
    print(f"\nResults saved to '{output_name}.png'")
    print(f"\nMiniLM Scores for {domain} domain using top {len(top_queries)} queries where {method1} outperforms {method2}{metric_desc} (winners highlighted):")
    print(minilm_styled_df.to_string())
    print(f"\nE5 Scores for {domain} domain using top {len(top_queries)} queries where {method1} outperforms {method2}{metric_desc} (winners highlighted):")
    print(e5_styled_df.to_string())

if __name__ == "__main__":
    main() 

'''
# Examples:
# Use all metrics:
python pattern_analyzing/compare_methods_by_query_30_combined.py --domain travel --method1 eqr --method2 q2d --top_n 100
python pattern_analyzing/compare_methods_by_query_30_combined.py --domain hotel --method1 eqr --method2 q2e --top_n 100
python pattern_analyzing/compare_methods_by_query_30_combined.py --domain restaurant --method1 eqr --method2 q2e --top_n 100

# Use only specific metrics:
python pattern_analyzing/compare_methods_by_query_30_combined.py --domain restaurant --method1 eqr --method2 q2e --top_n 100 --metrics map_at10 recall_at10 rprecision
python pattern_analyzing/compare_methods_by_query_30_combined.py --domain travel --method1 eqr --method2 q2d --top_n 90 --metrics map_at10 recall_at10 map_at30
python pattern_analyzing/compare_methods_by_query_30_combined.py --domain hotel --method1 eqr --method2 q2e --top_n 100 --metrics map_at10 recall_at10 map_at30

'''