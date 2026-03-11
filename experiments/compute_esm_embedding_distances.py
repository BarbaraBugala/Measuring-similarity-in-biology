#!/usr/bin/env python3
"""
compute_esm_embedding_distances_and_kernels.py

Loads ESM embeddings for individual proteins, computes:
- Cosine similarity and distance
- Euclidean distance
- RBF kernel from Euclidean distance

Saves all results as CSV for downstream analysis (CKA-ready).

Usage:
    python experiments/compute_esm_embedding_distances_and_kernels.py
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
from config import EMBEDDINGS_FILE, DISTANCE_DIR

embeddings_dir = EMBEDDINGS_FILE
output_dir = DISTANCE_DIR 
output_dir.mkdir(parents=True, exist_ok=True)

# Distance computation parameters
BATCH_SIZE = 500  # for Euclidean distance computation

# -----------------------------
# LOAD INDIVIDUAL EMBEDDINGS
# -----------------------------
print("Loading individual embedding files...")
labels = []
embedding_list = []

for file_path in embeddings_dir.glob("*.pt"):
    if file_path.name == "all_embeddings.pt":
        continue
    emb = torch.load(file_path)           # embedding: 1D tensor [D]
    embedding_list.append(emb)
    labels.append(file_path.stem)

if not embedding_list:
    raise ValueError("No .pt files found in the directory!")

embeddings = torch.stack(embedding_list)  # NxD
N, D = embeddings.shape
print(f"Loaded {N} embeddings with dimension {D}")

# -----------------------------
# COSINE SIMILARITY & DISTANCE
# -----------------------------
print("Computing Cosine Similarity and Distance...")
norm_emb = F.normalize(embeddings, p=2, dim=1)
cos_sim = torch.mm(norm_emb, norm_emb.T)
cos_kernel = (cos_sim + 1.0) / 2.0  # scale to [0,1]
cos_distance = 1.0 - cos_sim         # distance = 1 - similarity

# Save CSV
pd.DataFrame(cos_kernel.cpu().numpy(), index=labels, columns=labels).to_csv(output_dir / "esm2_cosine_kernel.csv")
pd.DataFrame(cos_distance.cpu().numpy(), index=labels, columns=labels).to_csv(output_dir / "esm2_cosine_distance.csv")
print("Cosine kernel and distance saved.")

# -----------------------------
# EUCLIDEAN DISTANCE
# -----------------------------
print("Computing Euclidean Distance...")
dist_mat = torch.zeros(N, N)

for i in range(0, N, BATCH_SIZE):
    end_i = min(i + BATCH_SIZE, N)
    dist_mat[i:end_i] = torch.cdist(embeddings[i:end_i], embeddings, p=2)
    print(f"Processed rows {i} to {end_i} / {N}")

df_euc = pd.DataFrame(dist_mat.cpu().numpy(), index=labels, columns=labels)
df_euc.to_csv(output_dir / "esm2_euclidean_distance.csv")
print("Euclidean distance saved.")

# -----------------------------
# RBF KERNEL from Euclidean distance
# -----------------------------
gamma = 1.0 / D  # heuristic: 1 / embedding dimension
rbf_kernel = torch.exp(-gamma * dist_mat**2)
pd.DataFrame(rbf_kernel.cpu().numpy(), index=labels, columns=labels).to_csv(output_dir / "esm2_euclidean_kernel.csv")
print("RBF kernel from Euclidean distance saved.")

print("All four matrices computed successfully!")