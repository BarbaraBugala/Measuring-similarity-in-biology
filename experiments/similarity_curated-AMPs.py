import torch
import torch.nn.functional as F
from itertools import combinations
from pathlib import Path

# -----------------------------
# LOAD EMBEDDINGS
# -----------------------------
EMBEDDINGS_PATH = "data/embeddings_esm2_curated-AMPs/all_embeddings.pt"

sequence_embeddings = torch.load(EMBEDDINGS_PATH)

labels = list(sequence_embeddings.keys())
embeddings = torch.stack([sequence_embeddings[l] for l in labels])

print(f"Loaded {len(labels)} embeddings")
print(f"Embedding dimension: {embeddings.shape[1]}")

# Normalize embeddings for cosine similarity
normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

# Cosine similarity matrix
cosine_sim_matrix = torch.mm(normalized_embeddings, normalized_embeddings.T)

print("Cosine similarity matrix shape:", cosine_sim_matrix.shape)

# Efficient Euclidean distance matrix
euclidean_dist_matrix = torch.cdist(embeddings, embeddings, p=2)

print("Euclidean distance matrix shape:", euclidean_dist_matrix.shape)
