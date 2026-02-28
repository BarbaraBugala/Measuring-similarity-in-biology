#!/usr/bin/env python3
"""
compute_esm_embedding_distances.py

Loads ESM embeddings for individual proteins, computes cosine similarity
and Euclidean distance matrices, and saves the results as CSV for downstream analysis.

Usage:
    python scripts/compute_esm_embedding_distances.py
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
EMBEDDINGS_DIR = Path("data/embeddings_esm2_curated-AMPs")
OUTPUT_DIR = Path("results/distance_matrices")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Distance computation parameters
BATCH_SIZE = 500  # for Euclidean distance computation
COMPUTE_COSINE = True
COMPUTE_EUCLIDEAN = True

# -----------------------------
# LOAD INDIVIDUAL EMBEDDINGS
# -----------------------------
print("Loading individual embedding files...")
labels = []
embedding_list = []

for file_path in EMBEDDINGS_DIR.glob("*.pt"):
    if file_path.name == "all_embeddings.pt":
        continue
    emb = torch.load(file_path)
    embedding_list.append(emb)
    labels.append(file_path.stem)

if not embedding_list:
    raise ValueError("No .pt files found in the directory!")

embeddings = torch.stack(embedding_list)
print(f"Loaded {len(labels)} embeddings with dimension {embeddings.shape[1]}")

# -----------------------------
# COSINE SIMILARITY
# -----------------------------
if COMPUTE_COSINE:
    print("Computing Cosine Similarity...")
    norm_emb = F.normalize(embeddings, p=2, dim=1)
    cos_sim = torch.mm(norm_emb, norm_emb.T)
    # Convert to DataFrame and save
    df_cos = pd.DataFrame(cos_sim.cpu().numpy(), index=labels, columns=labels)
    cos_path = OUTPUT_DIR / "esm2_cosine_similarity.csv"
    df_cos.to_csv(cos_path)
    print(f"Cosine similarity matrix saved to: {cos_path}")

# -----------------------------
# EUCLIDEAN DISTANCE
# -----------------------------
if COMPUTE_EUCLIDEAN:
    print("Computing Euclidean Distance with batching...")
    N = len(embeddings)
    dist_mat = torch.zeros(N, N)

    for i in range(0, N, BATCH_SIZE):
        end_i = min(i + BATCH_SIZE, N)
        dist_mat[i:end_i] = torch.cdist(embeddings[i:end_i], embeddings, p=2)
        print(f"Processed rows {i} to {end_i} / {N}")

    # Convert to DataFrame and save
    df_euc = pd.DataFrame(dist_mat.cpu().numpy(), index=labels, columns=labels)
    euc_path = OUTPUT_DIR / "esm2_euclidean_distance.csv"
    df_euc.to_csv(euc_path)
    print(f"Euclidean distance matrix saved to: {euc_path}")

print("Done!")