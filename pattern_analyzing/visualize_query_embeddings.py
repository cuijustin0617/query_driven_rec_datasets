import json
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors

# Load the JSON file and extract queries and scores
with open('pattern_analyzing/q2d_q2e_map_10.json', 'r') as f:
    data = json.load(f)
    
queries = list(data['query_ranks'].keys())
scores = list(data['query_ranks'].values())

# Load MiniLM model and generate embeddings
print("Loading model and generating embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(queries)

# Apply t-SNE dimensionality reduction
print("Applying t-SNE dimensionality reduction...")
tsne = TSNE(n_components=2, random_state=42, perplexity=10)
embeddings_2d = tsne.fit_transform(embeddings)

# Create visualization with color gradient based on scores
plt.figure(figsize=(14, 10))

# Create a colormap normalized to the score range
norm = mcolors.Normalize(vmin=min(scores), vmax=max(scores))
scatter = plt.scatter(
    embeddings_2d[:, 0], 
    embeddings_2d[:, 1], 
    c=scores, 
    cmap='viridis', 
    s=100, 
    alpha=0.8,
    norm=norm
)

# Add colorbar with clear labels
cbar = plt.colorbar(scatter)
cbar.set_label('Score (Higher is Better)', rotation=270, labelpad=20, fontsize=12)

# Add title and labels
plt.title('t-SNE Visualization of Query Embeddings Colored by Score', fontsize=16)
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)

# Label some points (avoid labeling all to prevent clutter)
# Sort by absolute score value to prioritize extreme scores for labeling
labeled_indices = sorted(range(len(scores)), key=lambda i: abs(scores[i]), reverse=True)[:15]

for i in labeled_indices:
    plt.annotate(
        text=queries[i][:30] + "..." if len(queries[i]) > 30 else queries[i],
        xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
    )

# Save the visualization
plt.tight_layout()
plt.savefig('query_embedding_visualization.png', dpi=300)
print("Visualization saved as 'query_embedding_visualization.png'")
plt.show() 