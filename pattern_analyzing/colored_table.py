import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def load_metric_data(metric_type: str, evaluation_metric: str, city: str, domain: str):
    """Load metric data for a specific metric type, evaluation metric, and city."""
    if domain == 'travel':
        # Special case for travel domain - different directory structure
        path = f"pattern_analyzing/final_results/{domain}/{city}_{metric_type}/{evaluation_metric}.json"
    else:
        path = f"pattern_analyzing/final_results/{domain}/{city}/{city}_{metric_type}/{evaluation_metric}.json"
    with open(path, 'r') as f:
        return json.load(f)

def get_all_queries(metric_types, evaluation_metrics, cities, domain):
    """Get all unique queries across all metrics, evaluation metrics, and cities."""
    all_queries = set()
    for metric in metric_types:
        for city in cities:
            for eval_metric in evaluation_metrics:
                try:
                    data = load_metric_data(metric, eval_metric, city, domain)
                    all_queries.update(data.keys())
                except FileNotFoundError:
                    print(f"Warning: Missing data for {metric}, {eval_metric}, {city}")
    return sorted(all_queries)

def calculate_average_scores(queries, metric_types, evaluation_metrics, cities, domain):
    """Calculate average scores for each query across cities and evaluation metrics."""
    # Initialize the DataFrame
    columns = metric_types
    df = pd.DataFrame(index=queries, columns=columns)
    
    # Calculate scores
    for query in queries:
        for metric in metric_types:
            all_scores = []
            for city in cities:
                city_scores = []
                for eval_metric in evaluation_metrics:
                    try:
                        data = load_metric_data(metric, eval_metric, city, domain)
                        if query in data:
                            city_scores.append(data[query])
                    except FileNotFoundError:
                        continue
                
                if city_scores:
                    # Average score across evaluation metrics for this city
                    all_scores.append(np.mean(city_scores))
            
            if all_scores:
                # Average score across cities
                df.at[query, metric] = np.mean(all_scores)
            else:
                df.at[query, metric] = np.nan
    
    # Add average row
    df.loc['AVERAGE'] = df.mean()
    
    # Count times each method is worst
    worst_counts = {metric: 0 for metric in metric_types}
    for query in queries:
        # Get scores for this query across all methods
        scores = df.loc[query].dropna()
        if len(scores) > 0:
            # Find method with lowest score
            worst_method = scores.idxmin()
            worst_counts[worst_method] += 1
    
    # Add worst count row
    df.loc['TIMES_WORST'] = pd.Series(worst_counts)
    
    return df

def visualize_metrics_table(df, output_file):
    """Create a color-coded table visualization of the metrics."""
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.4)))
    
    # Hide the axes
    ax.axis('tight')
    ax.axis('off')
    
    # Define a custom colormap (gradient from red to yellow to green)
    cmap = LinearSegmentedColormap.from_list('rg', ["#ff9999", "#ffff99", "#99ff99"], N=256)
    
    # Convert DataFrame to numeric values (this will convert non-numeric to NaN)
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    values = df_numeric.values
    
    # Handle NaN values
    cell_text = np.empty_like(values, dtype=object)
    for i in range(df_numeric.shape[0]):
        for j in range(df_numeric.shape[1]):
            val = df_numeric.iloc[i, j]
            cell_text[i, j] = f"{val:.4f}" if pd.notnull(val) else ""
    
    # Global normalization across the entire table
    # Find min/max across all values in the table
    mask = ~np.isnan(values)
    if np.any(mask):
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        
        # Create normalized array for colors
        norm_values = np.zeros_like(values)
        
        if vmax > vmin:
            # Normalize all values
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    if not np.isnan(values[i, j]):
                        norm_values[i, j] = (values[i, j] - vmin) / (vmax - vmin)
                    else:
                        norm_values[i, j] = 0.5  # Default color for NaN
        else:
            # All values are the same, use middle color
            norm_values[mask] = 0.5
            norm_values[~mask] = 0.5
    else:
        # No valid values, use default coloring
        norm_values = np.full_like(values, 0.5)
        vmin, vmax = 0, 1  # Default range for colorbar
    
    # Create the table with colored cells based on values
    table = ax.table(
        cellText=cell_text,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        cellColours=plt.colormaps.get_cmap(cmap)(norm_values)
    )
    
    # Adjust table style
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Highlight the average row with bold text
    for (i, j), cell in table.get_celld().items():
        if i == len(df):  # The last row (Average)
            cell.set_text_props(fontweight='bold')
    
    # Add a title and colorbar explanation
    plt.title('Average Scores by Query and Method', fontsize=14, pad=20)
    
    # Add a simple colorbar as legend
    ax_legend = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                       cax=ax_legend, orientation='horizontal')
    cbar.set_label('Score Value')
    
    # Save the figure
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Visualization saved to {output_file}")
    
    # Also save as CSV for reference
    csv_file = output_file.replace('.png', '.csv')
    df.to_csv(csv_file)
    print(f"Data saved to {csv_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize metrics across methods')
    parser.add_argument('--eval_metrics', type=str, nargs='+', required=True,
                        help='Evaluation metrics (e.g., map_at10 recall_at10 rprecision)')
    parser.add_argument('--output', type=str, default='metrics_comparison.png',
                        help='Output file path for visualization')
    parser.add_argument('--top_n', type=int, default=150,
                        help='Number of top queries to include (default: 50, use 0 for all)')
    parser.add_argument('--domain', type=str, choices=['hotel', 'restaurant', 'restaurant_jst', 'travel'], default='hotel',
                        help='Domain to analyze (hotel, restaurant, or travel)')
    
    args = parser.parse_args()
    
    # Methods to consider
    metric_types = ['eqr', 'q2e', 'q2d', 'none']
    
    # Cities to consider based on domain
    if args.domain == 'hotel':
        cities = ['london', 'chicago', 'montreal', 'nyc']
    elif args.domain == 'restaurant':
        cities = ['nor', 'phi']
    elif args.domain == 'restaurant_jst':
        cities = ['nor', 'phi']
    else:  # travel
        cities = ['travel']
    
    print(f"Analyzing {args.domain} domain with cities: {', '.join(cities)}")
    
    # Get all unique queries
    all_queries = get_all_queries(metric_types, args.eval_metrics, cities, args.domain)
    
    # Calculate average scores
    df = calculate_average_scores(all_queries, metric_types, args.eval_metrics, cities, args.domain)
    
    # Extract worst counts before removing the row
    if 'TIMES_WORST' in df.index:
        worst_counts = df.loc['TIMES_WORST'].to_dict()
        df = df.drop('TIMES_WORST')
    else:
        worst_counts = {}
    
    # Sort by the average of all metrics (descending)
    df_sorted = df.loc[df.index != 'AVERAGE'].mean(axis=1).sort_values(ascending=False)
    top_queries = df_sorted.index[:args.top_n] if args.top_n > 0 else df_sorted.index
    
    # Create a new dataframe with only the top queries
    df_viz = df.loc[list(top_queries) + ['AVERAGE']]
    
    # Create visualization
    visualize_metrics_table(df_viz, args.output)
    
    # Print out the 'TIMES_WORST' information separately
    if worst_counts:
        print("\nTIMES WORST (out of", len(all_queries), "queries):")
        for method, count in worst_counts.items():
            print(f"  {method}: {int(count)}")
    
    print(f"\nVisualization created with {'all' if args.top_n == 0 else args.top_n} queries and metrics: {', '.join(args.eval_metrics)}")

if __name__ == "__main__":
    main() 


'''
python pattern_analyzing/colored_table.py --eval_metrics map_at10 map_at30 map_at50 recall_at10 recall_at30 recall_at50 rprecision --domain travel --output travel_metrics_comparison.png

python pattern_analyzing/colored_table.py --eval_metrics map_at10  --domain hotel --output hotel_map10_comparison.png

python pattern_analyzing/colored_table.py --eval_metrics map_at10 map_at30 map_at50 recall_at10 recall_at30 recall_at50 rprecision --domain restaurant --output restaurant_metrics_comparison.png




'''