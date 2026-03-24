#!/usr/bin/env python3
"""
compute_pairwise_mknn.py

Computes mutual k-nearest neighbor overlap between 
distance matrices of different representations with stable tie-handling.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config import RESULTS_DIR, DISTANCE_FILES, EMB_METHODS, SEQ_METHODS, MODEL_TYPE

# -----------------------------
# CONFIG
# -----------------------------
K_VALUES = [5, 10, 20]
OUTPUT_FILE = RESULTS_DIR / "similarities" / MODEL_TYPE / "mknn_scores.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# STABLE KNN FUNCTIONS
# -----------------------------

def get_stable_knn_mask(D, k):
    """
    Returns a boolean mask where mask[i, j] is True if j is a k-NN of i.
    Handles ties by including ALL neighbors at the threshold distance.
    """
    n = D.shape[0]
    mask = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        dist_row = D[i, :].copy()
        
        # We ignore the diagonal (self-distance) to find true neighbors
        dist_row[i] = np.inf 
        
        # Find the distance of the k-th closest neighbor
        # partition is O(n), faster than a full sort O(n log n)
        kth_dist = np.partition(dist_row, k-1)[k-1]
        
        # Include everyone who is at most kth_dist away
        mask[i, :] = (dist_row <= kth_dist)
        
    return mask

def mutual_knn_graph(D, k):
    """
    Compute mutual kNN adjacency matrix.
    A and B are mutual neighbors if A is in B's kNN AND B is in A's kNN.
    """
    knn_mask = get_stable_knn_mask(D, k)
    
    # Mutual kNN is the intersection of the mask and its transpose
    # (i -> j) AND (j -> i)
    return np.logical_and(knn_mask, knn_mask.T)

def mknn_overlap(D1, D2, k):
    """
    Jaccard overlap between mutual kNN graphs of two different distance metrics.
    """
    G1 = mutual_knn_graph(D1, k)
    G2 = mutual_knn_graph(D2, k)

    intersection = np.logical_and(G1, G2).sum()
    union = np.logical_or(G1, G2).sum()

    if union == 0:
        return 0.0

    return intersection / union

# -----------------------------
# LOAD MATRICES
# -----------------------------
print("Loading distance matrices...")
distances = {}
ids = {}

for name, path in DISTANCE_FILES.items():
    if not Path(path).exists():
        print(f"Warning: {path} not found. Skipping {name}.")
        continue
    df = pd.read_csv(path, index_col=0)
    distances[name] = df
    ids[name] = set(df.index)

print("All matrices loaded.")

# -----------------------------
# COMPUTE mKNN OVERLAP
# -----------------------------
results = []

for esm in EMB_METHODS:
    if esm not in distances: continue
    
    for seq in SEQ_METHODS:
        if seq not in distances: continue

        print(f"Computing mKNN: {esm} vs {seq}")

        # Ensure we are comparing the exact same set of biological entities
        common_ids = sorted(list(ids[esm].intersection(ids[seq])))
        
        if len(common_ids) < max(K_VALUES):
            print(f"Skipping {esm}/{seq}: too few common IDs ({len(common_ids)})")
            continue

        # Slice matrices to common IDs
        D1 = distances[esm].loc[common_ids, common_ids].values
        D2 = distances[seq].loc[common_ids, common_ids].values

        for k in K_VALUES:
            score = mknn_overlap(D1, D2, k)

            results.append({
                "esm_method": esm,
                "sequence_method": seq,
                "k": k,
                "mknn_overlap": score,
                "n_samples": len(common_ids)
            })

# -----------------------------
# SAVE RESULTS
# -----------------------------
if results:
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_FILE, index=False)
    print("\nSaved mKNN scores to:", OUTPUT_FILE)
    print(df_results)
else:
    print("No results generated. Check your EMB_METHODS and SEQ_METHODS lists.")