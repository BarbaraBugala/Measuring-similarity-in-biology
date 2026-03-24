#!/usr/bin/env python3
"""
compute_pairwise_QSI_TSI.py

Computes QSI and TSI similarity between ESM embeddings and
sequence-based kernels and saves results.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.data import RepresentationPair

# QSI
from src.qsi import EfficientQSI, EfficientApproxQSI

# TSI
from src.tsi import EfficientTSI, EfficientApproxTSI

from config import RESULTS_DIR, KERNEL_FILES, EMB_METHODS, SEQ_METHODS, MODEL_TYPE


# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_FILE = RESULTS_DIR / "similarities" / MODEL_TYPE/ "qsi_tsi_scores.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

USE_APPROX = True


# -----------------------------
# DISTANCE FUNCTION WRAPPER
# -----------------------------
def make_distance_function(matrix):
    def d(i, j):
        return matrix[i, j]
    return d


# -----------------------------
# LOAD MATRICES
# -----------------------------
print("Loading distance matrices...")

matrices = {}
ids = {}

for name, path in KERNEL_FILES.items():
    df = pd.read_csv(path, index_col=0)
    matrices[name] = df
    ids[name] = set(df.index)

print("All matrices loaded.")


# -----------------------------
# INITIALIZE METRICS
# -----------------------------
if USE_APPROX:
    qsi_metric = EfficientApproxQSI(euclidean=False, batch_size=500, no_batches=10)
    tsi_metric = EfficientApproxTSI(euclidean=False, batch_size=500, no_batches=10)
else:
    qsi_metric = EfficientQSI(euclidean=False)
    tsi_metric = EfficientTSI(euclidean=False)


# -----------------------------
# COMPUTE SCORES
# -----------------------------
results = []

for emb in EMB_METHODS:
    for seq in SEQ_METHODS:

        print(f"Computing QSI/TSI: {emb} vs {seq}")

        # align matrices
        common_ids = sorted(ids[emb].intersection(ids[seq]))

        D1 = matrices[emb].loc[common_ids, common_ids].values
        D2 = matrices[seq].loc[common_ids, common_ids].values

        n = D1.shape[0]

        X = np.arange(n)
        Y = np.arange(n)

        representations = RepresentationPair(
            X=X,
            Y=Y,
            d_x=make_distance_function(D1),
            d_y=make_distance_function(D2),
        )

        qsi_score = qsi_metric(representations)
        tsi_score = tsi_metric(representations)

        results.append({
            "embedding_method": emb,
            "sequence_method": seq,
            "qsi": qsi_score,
            "tsi": tsi_score
        })


# -----------------------------
# SAVE RESULTS
# -----------------------------
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_FILE, index=False)

print("\nSaved QSI/TSI scores to:", OUTPUT_FILE)
print(df_results)