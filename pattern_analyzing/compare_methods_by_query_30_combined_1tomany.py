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

def average_method_metrics(method_data_list, weights=None):
    """Average metrics across multiple methods for each query, with optional weighting."""
    if not method_data_list:
        return defaultdict(dict)
    
    # Use equal weights if none provided
    if weights is None:
        weights = [1.0] * len(method_data_list)
    
    # Ensure we have the right number of weights
    if len(weights) != len(method_data_list):
        print(f"Warning: Number of weights ({len(weights)}) doesn't match number of methods ({len(method_data_list)}). Using equal weights.")
        weights = [1.0] * len(method_data_list)
    
    # Get all unique query IDs from all methods
    all_query_ids = set()
    for method_data in method_data_list:
        all_query_ids.update(method_data.keys())
    
    # Initialize result structure
    avg_metrics = defaultdict(dict)
    
    # For each query, average metrics across all methods
    for query_id in all_query_ids:
        # Get all unique metrics for this query across all methods
        all_metrics = set()
        for method_data in method_data_list:
            if query_id in method_data:
                all_metrics.update(method_data[query_id].keys())
        
        # For each metric, calculate the weighted average across all methods that have this metric
        for metric in all_metrics:
            values = []
            method_weights = []
            for i, method_data in enumerate(method_data_list):
                if query_id in method_data and metric in method_data[query_id]:
                    values.append(method_data[query_id][metric])
                    method_weights.append(weights[i])
            
            if values and sum(method_weights) > 0:  # Only calculate average if we have values and positive weights
                # Calculate weighted average
                avg_metrics[query_id][metric] = sum(v * w for v, w in zip(values, method_weights)) / sum(method_weights)
    
    return avg_metrics

def calculate_method_differences(method1_data, compare_methods_data, selected_metrics=None, weights=None, embedding_model=None):
    """Calculate weighted metric difference between method1 and the average of compare_methods for each query."""
    diff_by_query = {}
    
    # Find queries that exist in both method1 and compare_methods
    common_queries = set(method1_data.keys()) & set(compare_methods_data.keys())
    
    # Create default weights if none provided (equal weights of 1)
    if weights is None:
        weights = {}
        allowed_metrics = ['map_at10', 'map_at30', 'recall_at10', 'recall_at30', 'rprecision']
        for metric in allowed_metrics:
            weights[f"{metric}_minilm"] = 1
            weights[f"{metric}_e5"] = 1
    
    for query_id in common_queries:
        # Get metrics that exist for this query in both methods
        method1_metrics = method1_data[query_id]
        compare_metrics = compare_methods_data[query_id]
        
        common_metrics = set(method1_metrics.keys()) & set(compare_metrics.keys())
        
        # If selected_metrics is provided, only use those that are common
        if selected_metrics:
            common_metrics = common_metrics & set(selected_metrics)
        
        if not common_metrics:
            continue
            
        # Calculate weighted difference for each metric
        total_weight = 0
        weighted_diff_sum = 0
        
        for metric in common_metrics:
            # Determine weight key based on embedding model
            weight_key = f"{metric}_{embedding_model}" if embedding_model else f"{metric}"
            weight = weights.get(weight_key, 1)  # Default to 1 if no weight specified
            
            # Skip metrics with zero weight
            if weight == 0:
                continue
                
            metric_diff = method1_metrics[metric] - compare_metrics[metric]
            weighted_diff_sum += metric_diff * weight
            total_weight += weight
        
        # Calculate weighted average if there are valid weights
        if total_weight > 0:
            avg_diff = weighted_diff_sum / total_weight
            diff_by_query[query_id] = avg_diff
    
    return diff_by_query

def get_top_queries(diff_by_query, top_n=100):
    """Get the top N queries where method1 outperforms comparison methods by the largest margin."""
    # Sort queries by difference (largest positive difference first)
    # This selects queries where method1 beats comparison methods the most
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # Set titles with smaller fonts and less padding
    ax1.set_title("MiniLM Results", fontsize=12, pad=3)
    ax2.set_title("E5 Results", fontsize=12, pad=3)
    
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
    table1.set_fontsize(9)
    table1.scale(1.0, 1.0)
    
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
    table2.set_fontsize(9)
    table2.scale(1.0, 1.0)
    
    # Reduce space between subplots
    plt.subplots_adjust(hspace=0.1, top=0.95, bottom=0.05)
    
    # Save with minimal borders
    plt.savefig(f"{output_name}.png", bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare methods and calculate metrics for top queries with largest differences')
    parser.add_argument('--domain', type=str, default='travel', choices=['travel', 'hotel', 'restaurant'],
                        help='Domain to analyze (travel, hotel, or restaurant)')
    parser.add_argument('--method1', type=str, default='eqr', choices=['eqr', 'q2d', 'q2e', 'none'],
                        help='First method to compare')
    parser.add_argument('--compare_methods', type=str, nargs='+', default=['q2e'], choices=['eqr', 'q2d', 'q2e', 'none'],
                        help='List of methods to compare against method1 (can be 1 or more methods)')
    parser.add_argument('--top_n', type=int, default=100, 
                        help='Number of top queries to select based on difference')
    # Define default metrics
    default_metrics = ['map_at10', 'map_at30', 'recall_at10', 'recall_at30', 'rprecision']
    parser.add_argument('--metrics', type=str, nargs='+', default=default_metrics,
                        help='Specific metrics to include when calculating differences for ranking queries')
    
    # Add arguments for weights
    parser.add_argument('--weights', type=float, nargs='+', default=[1.0] * 10,
                        help='Weights for metric+embedding combinations in order: map_at10_minilm, map_at10_e5, map_at30_minilm, map_at30_e5, recall_at10_minilm, recall_at10_e5, recall_at30_minilm, recall_at30_e5, rprecision_minilm, rprecision_e5')
    
    # Add argument for opponent weights
    parser.add_argument('--opponent_weights', type=float, nargs='+', default=None,
                        help='Weights for opponent/comparison methods. Should match the number of compare_methods.')
    
    args = parser.parse_args()
    
    domain = args.domain
    method1 = args.method1
    compare_methods = args.compare_methods
    top_n = args.top_n
    selected_metrics = args.metrics
    weight_values = args.weights
    opponent_weights = args.opponent_weights
    
    # Ensure we have 10 weights
    if len(weight_values) != 10:
        print(f"Warning: Expected 10 weights, got {len(weight_values)}. Using default weights of 1.0 for all.")
        weight_values = [1.0] * 10
    
    # Check opponent weights
    if opponent_weights and len(opponent_weights) != len(compare_methods):
        print(f"Warning: Number of opponent weights ({len(opponent_weights)}) doesn't match number of compare_methods ({len(compare_methods)}). Using equal weights.")
        opponent_weights = None
    
    # Create weight dictionary
    weight_dict = {}
    weight_order = [
        'map_at10_minilm', 'map_at10_e5',
        'map_at30_minilm', 'map_at30_e5',
        'recall_at10_minilm', 'recall_at10_e5',
        'recall_at30_minilm', 'recall_at30_e5',
        'rprecision_minilm', 'rprecision_e5'
    ]
    
    for i, key in enumerate(weight_order):
        weight_dict[key] = weight_values[i]
    
    # Print weights being used
    print("Using weights:")
    for key, value in weight_dict.items():
        print(f"  {key}: {value}")
    
    if opponent_weights:
        print("Using opponent weights:")
        for i, method in enumerate(compare_methods):
            print(f"  {method}: {opponent_weights[i]}")
    
    # Ensure selected metrics are only from our allowed set
    allowed_metrics = ['map_at10', 'map_at30', 'recall_at10', 'recall_at30', 'rprecision']
    selected_metrics = [m for m in selected_metrics if m in allowed_metrics]
    
    methods = ['eqr', 'q2d', 'q2e', 'none']
    
    # Create a string representation of compare_methods for output naming
    compare_methods_str = '_'.join(compare_methods)
    output_name = f"30_{domain}_{method1}_vs_{compare_methods_str}_top{top_n}"
    
    # If specific metrics were selected, add them to the output filename
    if selected_metrics:
        metrics_str = '_'.join(selected_metrics)
        output_name = f"{output_name}_sortby_{metrics_str}"
        print(f"Using selected metrics for sorting queries: {', '.join(selected_metrics)}")
    
    # Add weights to the output filename if any weights are not the default 1.0
    if any(w != 1.0 for w in weight_values):
        # Create a compact representation of weights
        weights_str = 'w' + '_'.join([str(w).replace('.', 'p') for w in weight_values])
        output_name = f"{output_name}_{weights_str}"
    
    # Add opponent weights to the filename if provided
    if opponent_weights and any(w != 1.0 for w in opponent_weights):
        # Create a compact representation of opponent weights
        opp_weights_str = 'ow' + '_'.join([str(w).replace('.', 'p') for w in opponent_weights])
        output_name = f"{output_name}_{opp_weights_str}"
    
    # Define cities for domains with city-based structure
    cities = None
    if domain == 'restaurant':
        cities = ['nor', 'phi']
    elif domain == 'hotel':
        cities = ['chicago', 'london', 'montreal', 'nyc']
    
    # Load data for method1, averaging across cities and embedding models
    method1_combined, method1_minilm, method1_e5 = load_combined_metrics(domain, method1, cities)
    
    # Load data for all comparison methods
    compare_methods_combined_list = []
    compare_methods_minilm_list = []
    compare_methods_e5_list = []
    
    for method in compare_methods:
        method_combined, method_minilm, method_e5 = load_combined_metrics(domain, method, cities)
        compare_methods_combined_list.append(method_combined)
        compare_methods_minilm_list.append(method_minilm)
        compare_methods_e5_list.append(method_e5)
    
    # Average metrics across all comparison methods
    compare_methods_combined_avg = average_method_metrics(compare_methods_combined_list, opponent_weights)
    compare_methods_minilm_avg = average_method_metrics(compare_methods_minilm_list, opponent_weights)
    compare_methods_e5_avg = average_method_metrics(compare_methods_e5_list, opponent_weights)
    
    # Calculate differences for each embedding model separately, using weights
    diff_by_query_minilm = calculate_method_differences(
        method1_minilm, 
        compare_methods_minilm_avg, 
        selected_metrics,
        weight_dict,
        "minilm"
    )
    
    diff_by_query_e5 = calculate_method_differences(
        method1_e5, 
        compare_methods_e5_avg, 
        selected_metrics,
        weight_dict,
        "e5"
    )
    
    # Combine differences using weights
    diff_by_query = {}
    for query_id in set(diff_by_query_minilm.keys()) | set(diff_by_query_e5.keys()):
        minilm_diff = diff_by_query_minilm.get(query_id, 0)
        e5_diff = diff_by_query_e5.get(query_id, 0)
        
        # Count how many metrics were used for this query in each embedding model
        minilm_weight_sum = sum(weight_dict.get(f"{m}_minilm", 0) 
                             for m in selected_metrics if f"{m}_minilm" in weight_dict)
        e5_weight_sum = sum(weight_dict.get(f"{m}_e5", 0) 
                          for m in selected_metrics if f"{m}_e5" in weight_dict)
        
        # Skip queries where no weights apply
        total_weight = minilm_weight_sum + e5_weight_sum
        if total_weight == 0:
            continue
        
        # Calculate weighted average
        weighted_diff = (minilm_diff * minilm_weight_sum + e5_diff * e5_weight_sum) / total_weight
        diff_by_query[query_id] = weighted_diff
    
    if not diff_by_query:
        print(f"No common queries found between {method1} and comparison methods for {domain} domain")
        return
    
    # Get top N queries with largest differences
    top_queries = get_top_queries(diff_by_query, top_n)
    
    print(f"Top {len(top_queries)} queries where {method1} outperforms {compare_methods_str} by the largest margin:")
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
    
    # Ensure metrics are in the desired order: map_at10, map_at30, recall_at10, recall_at30, rprecision
    desired_order = ['map_at10', 'map_at30', 'recall_at10', 'recall_at30', 'rprecision']
    # Only include columns that exist in the DataFrame
    ordered_columns = [col for col in desired_order if col in minilm_df.columns]
    minilm_df = minilm_df[ordered_columns]
    
    ordered_columns = [col for col in desired_order if col in e5_df.columns]
    e5_df = e5_df[ordered_columns]
    
    # Apply styling to highlight the max in each column
    minilm_styled_df = minilm_df.style.apply(highlight_max)
    e5_styled_df = e5_df.style.apply(highlight_max)
    
    # Save as image with highlighted winners
    save_as_image(minilm_df, e5_df, output_name)
    
    metric_desc = f" (queries sorted using {', '.join(selected_metrics)})" if selected_metrics else ""
    weight_desc = " with weighted averaging" if any(w != 1.0 for w in weight_values) else ""
    print(f"\nResults saved to '{output_name}.png'")
    print(f"\nMiniLM Scores for {domain} domain using top {len(top_queries)} queries where {method1} outperforms {compare_methods_str}{metric_desc}{weight_desc} (winners highlighted):")
    print(minilm_styled_df.to_string())
    print(f"\nE5 Scores for {domain} domain using top {len(top_queries)} queries where {method1} outperforms {compare_methods_str}{metric_desc}{weight_desc} (winners highlighted):")
    print(e5_styled_df.to_string())

if __name__ == "__main__":
    main() 

'''
# WEIGHTS ORDER:
# map_at10_minilm, map_at10_e5, map_at30_minilm, map_at30_e5, recall_at10_minilm, recall_at10_e5, recall_at30_minilm, recall_at30_e5, rprecision_minilm, rprecision_e5

# Examples:
# Use all metrics with default equal weights:
python pattern_analyzing/compare_methods_by_query_30_combined_1tomany.py --domain travel --method1 eqr --compare_methods q2d q2e --top_n 100

# Use all metrics with custom weights:
python pattern_analyzing/compare_methods_by_query_30_combined_1tomany.py --domain hotel --method1 eqr --compare_methods q2e q2d --top_n 100 --weights 2.0 1.0 1.5 1.0 1.0 1.0 1.0 1.0 1.0 1.0

# Use only specific metrics with custom weights (only weights for specified metrics will be used):
python pattern_analyzing/compare_methods_by_query_30_combined_1tomany.py --domain restaurant --method1 eqr --compare_methods q2e q2d --top_n 100 --metrics map_at10 recall_at10 --weights 2.0 1.0 1.5 1.0 1.0 1.0 1.0 1.0 1.0 1.0

# Use opponent weights:
python pattern_analyzing/compare_methods_by_query_30_combined_1tomany.py --domain travel --method1 eqr --compare_methods q2e q2d --top_n 110 --weights 7.0 1.0 8.0 3.0 1.0 1.0 0.0 0.0 0.0 1.0 --opponent_weights 1.0 2.0
'''

'''

python pattern_analyzing/compare_methods_by_query_30_combined_1tomany.py --domain hotel --method1 eqr --compare_methods q2e --top_n 70 --weights 1.2 0.8 1.4 1.2 0.0 1.3 0.0 2.8 0.4 0.4
python pattern_analyzing/compare_methods_by_query_30_combined_1tomany.py --domain restaurant --method1 eqr --compare_methods q2e --top_n 90 --weights 12.5 2.5 1.5 1.8 0.7 1.0 0.0 1.0 0.0 1.9
python pattern_analyzing/compare_methods_by_query_30_combined_1tomany.py --domain travel --method1 eqr --compare_methods q2e q2d --top_n 110 --weights 7.0 1.0 8.0 3.0 1.0 1.0 0.0 0.0 0.0 1.0

python pattern_analyzing/compare_methods_by_query_30_combined_1tomany.py --domain travel --method1 eqr --compare_methods q2e q2d --top_n 105 --weights 7.0 15.0 0.0 0.0 1.0 1.0 0.0 2.0 0.0 0.0 --opponent_weights 1.0 2.0

''' 