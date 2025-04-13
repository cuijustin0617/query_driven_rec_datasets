import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def calculate_average_scores(directory):
    """Calculate average scores for specific metrics in a method directory."""
    metrics = {}
    # Only include these specific metrics
    allowed_metrics = ['recall_at10', 'recall_at30', 'map_at10', 'map_at30', 'rprecision']
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            metric_name = filename.split('.')[0]
            # Only process allowed metrics
            if metric_name in allowed_metrics:
                with open(os.path.join(directory, filename)) as f:
                    data = json.load(f)
                    avg_score = sum(data.values()) / len(data)
                    metrics[metric_name] = round(avg_score, 3)  # Round to 3 decimal places
    return metrics

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

def main():
    parser = argparse.ArgumentParser(description='Calculate metric averages for different domains')
    parser.add_argument('--domain', type=str, default='travel', choices=['travel', 'hotel', 'restaurant'],
                        help='Domain to analyze (travel, hotel, or restaurant)')
    args = parser.parse_args()
    
    domain = args.domain
    methods = ['eqr', 'q2d', 'q2e', 'none']
    output_name = f"{domain}_metric_averages"      ############################ TO CHANGE #########################################
     
    results = {}
    
    if domain == 'restaurant':
        # Handle restaurant domain which has two cities (nor and phi)
        cities = ['nor', 'phi']
        for method in methods:
            # Store metrics for each city
            city_metrics = {}
            
            for city in cities:
                # Match the actual directory structure
                method_dir = os.path.join("pattern_analyzing/final_results", domain, city, f"{city}_{method}")
                try:
                    city_metrics[city] = calculate_average_scores(method_dir)
                except FileNotFoundError:
                    print(f"Warning: Directory not found: {method_dir}")
                    city_metrics[city] = {}
            
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
            
            for city in cities:
                # Match the actual directory structure
                method_dir = os.path.join("pattern_analyzing/final_results", domain, city, f"{city}_{method}")
                try:
                    city_metrics[city] = calculate_average_scores(method_dir)
                except FileNotFoundError:
                    print(f"Warning: Directory not found: {method_dir}")
                    city_metrics[city] = {}
            
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
            method_dir = os.path.join("pattern_analyzing/final_results", domain, f"{domain}_{method}")
            try:
                results[method] = calculate_average_scores(method_dir)
            except FileNotFoundError:
                print(f"Warning: Directory not found: {method_dir}")
                results[method] = {}
    
    # Create DataFrame and transpose to have methods as rows and metrics as columns
    df = pd.DataFrame(results).T
    
    # Round all values to 3 decimal places
    df = df.round(3)
    
    # Apply styling to highlight the max in each column
    styled_df = df.style.apply(highlight_max)
    
    # Save as Excel
    # styled_df.to_excel(f"{output_name}.xlsx", engine="openpyxl")
    
    # Save as image with highlighted winners
    save_as_image(df, output_name)
    
    # Print to console with winners marked and rounded values
    print(f"Results saved to '{output_name}.xlsx' and '{output_name}.png'")
    print(f"\nAverage Scores for {domain} domain (winners highlighted, rounded to 3 decimal places):")
    print(styled_df.to_string())

if __name__ == "__main__":
    main() 