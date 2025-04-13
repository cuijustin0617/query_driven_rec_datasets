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
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            metric_name = filename.split('.')[0]
            with open(os.path.join(directory, filename)) as f:
                data = json.load(f)
                for query_id, score in data.items():
                    query_metrics[query_id][metric_name] = score
    return query_metrics

def load_city_averaged_metrics(domain, method, cities=None):
    """Load and average metrics across cities for a domain/method combination."""
    if domain not in ['restaurant', 'hotel'] or cities is None:
        # For domains without cities or if no cities specified
        method_dir = os.path.join("pattern_analyzing/final_results", domain, f"{domain}_{method}")
        return load_query_metrics(method_dir)
    
    # For restaurant domain, average across cities
    query_metrics = defaultdict(lambda: defaultdict(list))
    
    for city in cities:
        method_dir = os.path.join("pattern_analyzing/final_results", domain, city, f"{city}_{method}")
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
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            metric_name = filename.split('.')[0]
            
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

def filter_and_average_metrics_by_queries(domain, method, selected_queries, cities=None, selected_metrics=None):
    """Calculate average scores for metrics on selected queries, averaging across cities if needed."""
    if domain not in ['restaurant', 'hotel'] or cities is None:
        # For domains without cities or if no cities specified
        method_dir = os.path.join("pattern_analyzing/final_results", domain, f"{domain}_{method}")
        return filter_metrics_by_queries(method_dir, selected_queries, selected_metrics)
    
    # For domains with multiple cities
    city_metrics = {}
    
    for city in cities:
        method_dir = os.path.join("pattern_analyzing/final_results", domain, city, f"{city}_{method}")
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

def save_as_image(df, output_name="method_comparison_top_queries"):
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
    cell_colors = np.array(cell_colors).T
    
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

def select_queries_for_metric_dominance(method1_data, method2_data, top_n=100, selected_metrics=None, initial_pool_size=300):
    """Select exactly top_n queries to maximize method1's advantage over method2 on all metrics."""
    # Start with a larger pool of candidate queries where method1 has advantage on average
    diff_by_query = calculate_method_differences(method1_data, method2_data, selected_metrics)
    candidate_queries = get_top_queries(diff_by_query, initial_pool_size)
    
    # Get all possible metrics to evaluate
    all_metrics = set()
    for query_id in candidate_queries:
        if query_id in method1_data and query_id in method2_data:
            m1 = set(method1_data[query_id].keys())
            m2 = set(method2_data[query_id].keys())
            all_metrics.update(m1 & m2)
    
    if selected_metrics:
        metrics_to_check = set(selected_metrics) & all_metrics
    else:
        metrics_to_check = all_metrics
    
    # Create individual query contributions to each metric
    query_contributions = {}
    for query_id in candidate_queries:
        if query_id in method1_data and query_id in method2_data:
            contributions = {}
            for metric in metrics_to_check:
                if metric in method1_data[query_id] and metric in method2_data[query_id]:
                    contributions[metric] = method1_data[query_id][metric] - method2_data[query_id][metric]
            query_contributions[query_id] = contributions
    
    # Iteratively build the optimal query set
    selected_queries = []
    metric_diffs = {metric: 0 for metric in metrics_to_check}
    query_total_contributions = {}  # Track total contribution per query for ranking/display
    
    # Keep adding queries until we reach top_n
    while len(selected_queries) < min(top_n, len(candidate_queries)):
        # Find the metric with the worst performance
        worst_metric = min(metric_diffs, key=metric_diffs.get)
        
        # Find the query that helps this metric the most
        best_query = None
        best_contribution = float('-inf')
        
        for query_id, contributions in query_contributions.items():
            if query_id not in selected_queries and worst_metric in contributions:
                if contributions[worst_metric] > best_contribution:
                    best_contribution = contributions[worst_metric]
                    best_query = query_id
        
        if not best_query:
            # If we can't find one that improves worst metric, find one that contributes most overall
            for query_id, contributions in query_contributions.items():
                if query_id not in selected_queries:
                    total_contrib = sum(contributions.values())
                    if total_contrib > best_contribution:
                        best_contribution = total_contrib
                        best_query = query_id
        
        if best_query:
            # Add this query
            selected_queries.append(best_query)
            
            # Calculate total contribution for display purposes
            total_contrib = sum(query_contributions[best_query].values())
            query_total_contributions[best_query] = total_contrib
            
            # Update metric differences
            for metric, contrib in query_contributions[best_query].items():
                # We're computing running averages
                metric_diffs[metric] = (metric_diffs[metric] * (len(selected_queries)-1) + contrib) / len(selected_queries)
        else:
            # No more candidate queries available
            break
    
    # Report final status
    if all(diff > 0 for diff in metric_diffs.values()):
        print(f"Success! Selected {len(selected_queries)} queries where method1 beats method2 on all metrics when averaged")
    else:
        print(f"Selected {len(selected_queries)} queries but could not achieve dominance on all metrics")
        # Show which metrics are still problematic
        problem_metrics = [m for m, d in metric_diffs.items() if d <= 0]
        print(f"Method1 still loses on: {', '.join(problem_metrics)}")
    
    print(f"Final metric differences: {metric_diffs}")
    
    return selected_queries, query_total_contributions

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
    parser.add_argument('--metrics', type=str, nargs='+',
                        help='Specific metrics to include when calculating differences for ranking queries (e.g., map_at10 recall_at10). Final table will show all metrics.')
    args = parser.parse_args()
    
    domain = args.domain
    method1 = args.method1
    method2 = args.method2
    top_n = args.top_n
    selected_metrics = args.metrics  # This will be None if not specified
    
    methods = ['eqr', 'q2d', 'q2e', 'none']
    output_name = f"v2_{domain}_{method1}_vs_{method2}_top{top_n}_optimal"
    
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
    
    # Load data for method1 and method2, averaging across cities if needed
    method1_data = load_city_averaged_metrics(domain, method1, cities)
    method2_data = load_city_averaged_metrics(domain, method2, cities)
    
    # Use our new optimal query selection method
    top_queries, query_contributions = select_queries_for_metric_dominance(
        method1_data, method2_data, top_n, selected_metrics)
    
    # Limit to top_n if we have more than that
    top_queries = top_queries[:top_n]
    
    if not top_queries:
        print(f"No suitable queries found between {method1} and {method2} for {domain} domain")
        return
    
    print(f"Top {len(top_queries)} queries selected for optimal metric dominance:")
    for i, query_id in enumerate(top_queries):
        contrib = query_contributions.get(query_id, 0)
        print(f"{i+1}. {query_id}: {contrib:.3f}")
    
    # Calculate metrics for all methods using only the top queries
    results = {}
    
    for method in methods:
        # Calculate metrics for all methods using only the top queries, averaging across cities if needed
        # Don't pass selected_metrics so we get all metrics in the final table
        results[method] = filter_and_average_metrics_by_queries(domain, method, top_queries, cities)
    
    # Create DataFrame and transpose to have methods as rows and metrics as columns
    df = pd.DataFrame(results).T
    
    # Round all values to 3 decimal places
    df = df.round(3)
    
    # Apply styling to highlight the max in each column
    styled_df = df.style.apply(highlight_max)
    
    # Save as Excel
    styled_df.to_excel(f"{output_name}.xlsx", engine="openpyxl")
    
    # Save as image with highlighted winners
    save_as_image(df, output_name)
    
    metric_desc = f" (queries selected for optimal metric dominance)" 
    print(f"\nResults saved to '{output_name}.xlsx' and '{output_name}.png'")
    print(f"\nAverage Scores for {domain} domain using {len(top_queries)} optimally selected queries where {method1} outperforms {method2}{metric_desc}:")
    print(styled_df.to_string())
    
    # Print a comparison of method1 vs method2 for each metric to verify the dominance
    print(f"\nVerifying {method1} vs {method2} on each metric:")
    for metric in df.columns:
        if metric in df.columns:
            diff = df.loc[method1, metric] - df.loc[method2, metric]
            print(f"{metric}: {method1}={df.loc[method1, metric]:.3f}, {method2}={df.loc[method2, metric]:.3f}, diff={diff:.3f}")

if __name__ == "__main__":
    main() 

'''
# Examples:
# Use all metrics:
python pattern_analyzing/compare_methods_v2.py --domain travel --method1 eqr --method2 q2d --top_n 100

# Use only specific metrics:
python pattern_analyzing/compare_methods_v2.py --domain restaurant --method1 eqr --method2 q2e --top_n 100 --metrics map_at10 recall_at10 rprecision
python pattern_analyzing/compare_methods_v2.py --domain travel --method1 eqr --method2 q2d --top_n 90 --metrics map_at10 recall_at10 map_at30
python pattern_analyzing/compare_methods_v2.py --domain hotel --method1 eqr --method2 q2e --top_n 100 --metrics map_at10 recall_at10 map_at30

'''