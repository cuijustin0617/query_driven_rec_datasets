import os
import json
import argparse
from typing import List, Dict, Tuple
import numpy as np

def load_metric_data(metric_type: str, evaluation_metric: str, city: str) -> Dict[str, float]:
    """Load metric data for a specific metric type, evaluation metric, and city."""
    path = f"pattern_analyzing/hotel/{city}/{city}_{metric_type}/{evaluation_metric}.json"
    with open(path, 'r') as f:
        return json.load(f)

def get_query_metric_diff(metric1: str, metric2: str, evaluation_metrics: List[str], 
                         cities: List[str]) -> List[Tuple[str, float]]:
    """
    Calculate the average difference between two metrics for each query across cities.
    If one metric is 'none', rank queries based on the other metric's scores.
    
    Args:
        metric1: First metric (e.g., 'eqr', 'none')
        metric2: Second metric (e.g., 'q2e', 'none')
        evaluation_metrics: List of evaluation metrics (e.g., ['map_at10', 'recall_at10'])
        cities: List of cities to consider
        
    Returns:
        List of tuples (query, avg_diff) sorted by avg_diff in descending order
    """
    # Dictionary to store query scores or differences
    query_values = {}
    
    # Single metric ranking case
    if metric1 == 'none' or metric2 == 'none':
        active_metric = metric2 if metric1 == 'none' else metric1
        
        # Get all unique queries across all cities and metrics
        all_queries = set()
        for city in cities:
            for eval_metric in evaluation_metrics:
                try:
                    data = load_metric_data(active_metric, eval_metric, city)
                    all_queries.update(data.keys())
                except FileNotFoundError:
                    print(f"Warning: Missing data for {active_metric}, {eval_metric}, {city}")
        
        # Calculate average scores for each query
        for query in all_queries:
            scores = []
            for city in cities:
                city_scores = []
                for eval_metric in evaluation_metrics:
                    try:
                        data = load_metric_data(active_metric, eval_metric, city)
                        if query in data:
                            city_scores.append(data[query])
                    except FileNotFoundError:
                        continue
                
                if city_scores:
                    # Average score across evaluation metrics for this city
                    scores.append(np.mean(city_scores))
            
            if scores:
                # Average score across cities
                query_values[query] = np.mean(scores)
    
    # Difference between two metrics
    else:
        # Get all unique queries across all cities and metrics
        all_queries = set()
        for city in cities:
            for eval_metric in evaluation_metrics:
                try:
                    data1 = load_metric_data(metric1, eval_metric, city)
                    all_queries.update(data1.keys())
                except FileNotFoundError:
                    print(f"Warning: Missing data for {metric1}, {eval_metric}, {city}")
                    
                try:
                    data2 = load_metric_data(metric2, eval_metric, city)
                    all_queries.update(data2.keys())
                except FileNotFoundError:
                    print(f"Warning: Missing data for {metric2}, {eval_metric}, {city}")
        
        # Calculate differences for each query
        for query in all_queries:
            diffs = []
            for city in cities:
                city_diffs = []
                for eval_metric in evaluation_metrics:
                    try:
                        data1 = load_metric_data(metric1, eval_metric, city)
                        data2 = load_metric_data(metric2, eval_metric, city)
                        
                        if query in data1 and query in data2:
                            diff = data2[query] - data1[query]
                            city_diffs.append(diff)
                    except FileNotFoundError:
                        continue
                
                if city_diffs:
                    # Average difference across evaluation metrics for this city
                    diffs.append(np.mean(city_diffs))
            
            if diffs:
                # Average difference across cities
                query_values[query] = np.mean(diffs)
    
    # Sort queries by value (descending)
    sorted_queries = sorted(query_values.items(), key=lambda x: x[1], reverse=True)
    return sorted_queries

def main():
    parser = argparse.ArgumentParser(description='Rank queries based on metric differences')
    parser.add_argument('--metric1', type=str, required=True, choices=['eqr', 'q2e', 'q2d', 'none'],
                        help='First metric')
    parser.add_argument('--metric2', type=str, required=True, choices=['eqr', 'q2e', 'q2d', 'none'],
                        help='Second metric')
    parser.add_argument('--eval_metrics', type=str, nargs='+', required=True,
                        help='Evaluation metrics (e.g., map_at10 recall_at10 rprecision)')
    parser.add_argument('--output', type=str, default='query_ranks.json',
                        help='Output file path')
    
    args = parser.parse_args()
    
    # Validate metrics
    if args.metric1 == args.metric2 and args.metric1 != 'none':
        print("Error: metric1 and metric2 must be different unless one is 'none'")
        return
    
    if args.metric1 == 'none' and args.metric2 == 'none':
        print("Error: both metrics cannot be 'none'")
        return
    
    # Cities to consider
    cities = ['london', 'chicago', 'montreal', 'nyc']
    
    # Get sorted queries
    sorted_queries = get_query_metric_diff(args.metric1, args.metric2, args.eval_metrics, cities)
    
    # Output results
    if args.metric1 == 'none':
        print(f"\nTop 10 queries ranked by {args.metric2} performance:")
        for query, score in sorted_queries[:10]:
            print(f"{score:.4f}: {query}")
        
        print(f"\nBottom 10 queries ranked by {args.metric2} performance:")
        for query, score in sorted_queries[-10:]:
            print(f"{score:.4f}: {query}")
    elif args.metric2 == 'none':
        print(f"\nTop 10 queries ranked by {args.metric1} performance:")
        for query, score in sorted_queries[:10]:
            print(f"{score:.4f}: {query}")
        
        print(f"\nBottom 10 queries ranked by {args.metric1} performance:")
        for query, score in sorted_queries[-10:]:
            print(f"{score:.4f}: {query}")
    else:
        print(f"\nTop 10 queries where {args.metric2} performs better than {args.metric1}:")
        for query, diff in sorted_queries[:10]:
            print(f"{diff:.4f}: {query}")
        
        print(f"\nBottom 10 queries where {args.metric1} performs better than {args.metric2}:")
        for query, diff in sorted_queries[-10:]:
            print(f"{diff:.4f}: {query}")
    
    # Save all results to file
    results = {
        "metric1": args.metric1,
        "metric2": args.metric2,
        "eval_metrics": args.eval_metrics,
        "query_ranks": dict(sorted_queries)
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFull results saved to {args.output}")

if __name__ == "__main__":
    main()
