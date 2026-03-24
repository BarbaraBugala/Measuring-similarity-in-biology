#!/usr/bin/env python3
"""
compute_mknn_baseline.py

Computes mutual k-nearest neighbor overlap between distance matrices 
WITHOUT name-matching. This is used for calculating baseline scores 
after matrices have been permuted.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config import RESULTS_DIR, BASELINE_DISTANCE_FILES, BASELINE_EMB_METHODS, BASELINE_SEQ_METHODS, MODEL_TYPE

# -----------------------------
# CONFIG
# -----------------------------
K_VALUES = [5, 10, 20]
# Saving to a distinct file to avoid overwriting your real results
OUTPUT_FILE = RESULTS_DIR / "similarities" / MODEL_TYPE / "mknn_scores_baseline.csv"
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
        
        # Ignore self-distance
        dist_row[i] = np.inf 
        
        # Find distance of the k-th closest neighbor
        kth_dist = np.partition(dist_row, k-1)[k-1]
        
        # Include everyone who is at most kth_dist away
        mask[i, :] = (dist_row <= kth_dist)
        
    return mask

def mutual_knn_graph(D, k):
    """
    Compute mutual kNN adjacency matrix via logical AND of mask and transpose.
    """
    knn_mask = get_stable_knn_mask(D, k)
    return np.logical_and(knn_mask, knn_mask.T)

def mknn_overlap(D1, D2, k):
    """
    Jaccard overlap between mutual kNN graphs (Positional comparison).
    """
    G1 = mutual_knn_graph(D1, k)
    G2 = mutual_knn_graph(D2, k)

    intersection = np.logical_and(G1, G2).sum()
    union = np.logical_or(G1, G2).sum()

    return (intersection / union) if union != 0 else 0.0

# -----------------------------
# LOAD MATRICES (Raw Values)
# -----------------------------
print("Loading distance matrices as raw arrays...")
matrices = {}

for name, path in BASELINE_DISTANCE_FILES.items():
    if not Path(path).exists():
        print(f"  [Warning] {path} not found. Skipping {name}.")
        continue
    
    # We load the CSV and immediately discard the index/header names
    df = pd.read_csv(path, index_col=0)
    matrices[name] = df.values 
    print(f"  Loaded {name}: {df.shape}")

# -----------------------------
# COMPUTE mKNN OVERLAP (No Matching)
# -----------------------------
results = []

for esm in BASELINE_EMB_METHODS:
    if esm not in matrices: continue
    D_esm = matrices[esm]
    
    for seq in BASELINE_SEQ_METHODS:
        if seq not in matrices: continue
        D_seq = matrices[seq]

        # Ensure we can compare them
        if D_esm.shape != D_seq.shape:
            print(f"  [Skip] Shape mismatch: {esm} vs {seq}")
            continue

        print(f"Computing mKNN Baseline: {esm} vs {seq}")

        for k in K_VALUES:
            # This compares Row i to Row i directly
            score = mknn_overlap(D_esm, D_seq, k)

            results.append({
                "esm_method": esm,
                "sequence_method": seq,
                "k": k,
                "mknn_overlap": score,
                "n_samples": D_esm.shape[0]
            })

# -----------------------------
# SAVE RESULTS
# -----------------------------
if results:
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_FILE, index=False)
    print(f"\nBaseline results saved to: {OUTPUT_FILE}")
    print(df_results)
else:
    print("No results were generated. Check your config and file paths.")