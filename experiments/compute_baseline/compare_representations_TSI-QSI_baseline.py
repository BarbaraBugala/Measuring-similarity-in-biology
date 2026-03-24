#!/usr/bin/env python3
"""
compute_pairwise_QSI_TSI_baseline.py

Computes QSI and TSI similarity between ESM embeddings and 
sequence-based kernels WITHOUT name matching (positional baseline).
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.data import RepresentationPair
from src.qsi import EfficientQSI, EfficientApproxQSI
from src.tsi import EfficientTSI, EfficientApproxTSI

from config import (
    RESULTS_DIR, 
    BASELINE_KERNEL_FILES, 
    BASELINE_EMB_METHODS, 
    BASELINE_SEQ_METHODS, 
    MODEL_TYPE
)

# -----------------------------
# CONFIG
# -----------------------------
# Saving to a baseline-specific file
OUTPUT_FILE = RESULTS_DIR / "similarities" / MODEL_TYPE / "qsi_tsi_scores_baseline.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

USE_APPROX = True

# -----------------------------
# DISTANCE FUNCTION WRAPPER
# -----------------------------
def make_distance_function(matrix):
    # This closure maps indices directly to the matrix values
    def d(i, j):
        return matrix[i, j]
    return d

# -----------------------------
# LOAD MATRICES (Raw Values)
# -----------------------------
print("Loading matrices as raw arrays...")

matrices = {}

for name, path in BASELINE_KERNEL_FILES.items():
    if not Path(path).exists():
        print(f"  [Warning] {path} not found. Skipping.")
        continue
    
    # Load and immediately convert to .values to strip name awareness
    df = pd.read_csv(path, index_col=0)
    matrices[name] = df.values
    print(f"  Loaded {name}: {df.shape}")

# -----------------------------
# INITIALIZE METRICS
# -----------------------------
if USE_APPROX:
    # batch_size should be <= total number of samples
    qsi_metric = EfficientApproxQSI(euclidean=False, batch_size=500, no_batches=10)
    tsi_metric = EfficientApproxTSI(euclidean=False, batch_size=500, no_batches=10)
else:
    qsi_metric = EfficientQSI(euclidean=False)
    tsi_metric = EfficientTSI(euclidean=False)

# -----------------------------
# COMPUTE SCORES (Positional)
# -----------------------------
results = []

for emb in BASELINE_EMB_METHODS:
    if emb not in matrices: continue
    D1 = matrices[emb]
    
    for seq in BASELINE_SEQ_METHODS:
        if seq not in matrices: continue
        D2 = matrices[seq]

        # Ensure matrices are the same size for positional comparison
        if D1.shape != D2.shape:
            print(f"  [Skip] Shape mismatch: {emb} vs {seq}")
            continue

        print(f"Computing QSI/TSI Baseline: {emb} vs {seq}")

        n = D1.shape[0]

        # Use raw integer indices for the representation pair
        X = np.arange(n)
        Y = np.arange(n)

        representations = RepresentationPair(
            X=X,
            Y=Y,
            d_x=make_distance_function(D1),
            d_y=make_distance_function(D2),
        )

        # Compute metrics
        qsi_score = qsi_metric(representations)
        tsi_score = tsi_metric(representations)

        results.append({
            "embedding_method": emb,
            "sequence_method": seq,
            "qsi": qsi_score,
            "tsi": tsi_score,
            "n_samples": n
        })

# -----------------------------
# SAVE RESULTS
# -----------------------------
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_FILE, index=False)

print("\nSaved QSI/TSI baseline scores to:", OUTPUT_FILE)
print(df_results)