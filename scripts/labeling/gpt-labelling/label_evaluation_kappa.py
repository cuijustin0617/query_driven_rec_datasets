import csv
import json
from collections import defaultdict, Counter
from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def load_gpt_scores(csv_path):
    gpt_data_by_query = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row['Query'].strip()
            city = row['City'].strip()
            try:
                score = int(row['Relevance Score'])
            except ValueError:
                continue
            gpt_data_by_query[query].append((city, score))
    return gpt_data_by_query

def load_ground_truth(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_score_distribution(gpt_data_by_query, sample_queries):
    for query in sample_queries:
        city_scores = gpt_data_by_query.get(query, [])
        score_counts = Counter(score for _, score in city_scores)

        print(f"\nQuery: {query}")
        for score in range(4):
            count = score_counts.get(score, 0)
            print(f"Score {score}: {count} occurrences")

        plt.figure(figsize=(6, 4))
        scores = [score for _, score in city_scores]
        plt.hist(scores, bins=[-0.5, 0.5, 1.5, 2.5, 3.5], edgecolor='black', rwidth=0.8)
        plt.title(f"Score Distribution for Query:\n{query}", wrap=True)
        plt.xticks([0, 1, 2, 3])
        plt.xlabel("Relevance Score")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('scripts/labeling/gpt-labelling/bar_plots/' + query + '.png')


def evaluate_metrics_per_query(gpt_data_by_query, ground_truth):
    results = {}

    for query, city_scores in gpt_data_by_query.items():
        gpt_labels = []
        truth_labels = []

        for city, gpt_score in city_scores:
            binarized_gpt = 1 if gpt_score == 3 else 0
            truth_cities = ground_truth.get(query, [])
            binarized_truth = 1 if city in truth_cities else 0

            gpt_labels.append(binarized_gpt)
            truth_labels.append(binarized_truth)

        if len(gpt_labels) >= 2:
            kappa = cohen_kappa_score(gpt_labels, truth_labels)
            accuracy = accuracy_score(truth_labels, gpt_labels)
            precision = precision_score(truth_labels, gpt_labels, zero_division=0)
            recall = recall_score(truth_labels, gpt_labels, zero_division=0)
            f1 = f1_score(truth_labels, gpt_labels, zero_division=0)

            results[query] = {
                'kappa': round(kappa, 4),
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4)
            }
        else:
            results[query] = {
                'kappa': None,
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1_score': None
            }
    return results


def main():
    csv_path = 'scripts/labeling/gpt-labelling/labeled_documents.csv'
    json_path = 'scripts/labeling/gpt-labelling/ground_truth.json'

    gpt_data_by_query = load_gpt_scores(csv_path)
    ground_truth = load_ground_truth(json_path)
    ground_truth = {query: cities for query, cities in ground_truth.items() if query in gpt_data_by_query} # Filter ground truth

    # Plot score distribution for first 5 sample queries
    sample_queries = list(gpt_data_by_query.keys())[:5]
    plot_score_distribution(gpt_data_by_query, sample_queries)

    # Compute metrics
    results = evaluate_metrics_per_query(gpt_data_by_query, ground_truth)
    print("\nPer-Query Evaluation Metrics:")
    for query, metrics in results.items():
        print(f"\nQuery: {query}")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value}")

if __name__ == "__main__":
    main()
