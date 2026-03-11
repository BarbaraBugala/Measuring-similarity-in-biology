#!/usr/bin/env python3
"""
compute_pairwise_CKA.py

Computes CKA similarity between ESM2 embeddings (cosine and euclidean)
and sequence-based kernels (Blosum, Hamming, Blast, MSA) and saves results.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config import DISTANCE_DIR

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_FILE = DISTANCE_DIR / "similarities" / "cka_scores.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

KERNEL_FILES = {
    "hamming": DISTANCE_DIR / "kernels" / "sequence_hamming_kernel.csv",
    "blosum": DISTANCE_DIR / "kernels" / "sequence_blosum_kernel.csv",
    "blast": DISTANCE_DIR / "kernels" / "sequence_blast_kernel.csv",
    "msa_p": DISTANCE_DIR / "kernels" / "msa_p_kernel.csv",
    "esm2_cosine": DISTANCE_DIR / "kernels" / "esm2_cosine_kernel.csv",
    "esm2_euclidean": DISTANCE_DIR / "kernels" / "esm2_euclidean_kernel.csv",
}

ESM_METHODS = ["esm2_cosine", "esm2_euclidean"]
SEQ_METHODS = ["hamming", "blosum", "blast", "msa_p"]

# -----------------------------
# CKA IMPLEMENTATION
# -----------------------------
def center_kernel(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def compute_cka(K, L):
    Kc = center_kernel(K)
    Lc = center_kernel(L)
    hsic = np.trace(Kc @ Lc)
    norm_k = np.trace(Kc @ Kc)
    norm_l = np.trace(Lc @ Lc)
    return hsic / np.sqrt(norm_k * norm_l)



# -----------------------------
# LOAD ALL KERNELS
# -----------------------------
print("Loading kernel matrices...")
kernels = {}
ids = {}

for name, path in KERNEL_FILES.items():
    df = pd.read_csv(path, index_col=0)
    kernels[name] = df
    ids[name] = set(df.index)

print("All kernels loaded.")

# -----------------------------
# COMPUTE CKA FOR ESM vs SEQ KERNELS
# -----------------------------
results = []

for esm in ESM_METHODS:
    for seq in SEQ_METHODS:
        print(f"Computing CKA: {esm} vs {seq}")
        # Align on common IDs
        common_ids = sorted(ids[esm].intersection(ids[seq]))
        K = kernels[esm].loc[common_ids, common_ids].values
        L = kernels[seq].loc[common_ids, common_ids].values
        score = compute_cka(K, L)
        results.append({
            "esm_method": esm,
            "sequence_method": seq,
            "cka": score
        })

# -----------------------------
# SAVE RESULTS
# -----------------------------
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_FILE, index=False)
print("\nSaved CKA scores to:", OUTPUT_FILE)
print(df_results)