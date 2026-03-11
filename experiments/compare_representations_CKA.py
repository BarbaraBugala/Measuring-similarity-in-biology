#!/usr/bin/env python3
"""
compare_representations_CKA.py

Loads two kernel matrices (already converted from distance matrices),
centers them, and computes CKA similarity.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config import DISTANCES_CKA_1, DISTANCES_CKA_2, COMPARISON_OF_CKA

# -----------------------------
# Load kernel matrices
# -----------------------------
print("Loading kernel matrices...")

K1 = pd.read_csv(DISTANCES_CKA_1, index_col=0)
K2 = pd.read_csv(DISTANCES_CKA_2, index_col=0)


# -----------------------------
# Normalize sequence IDs to match
# -----------------------------
def normalize_ids(ids):
    """
    Convert IDs from ESM style (tr_X_Y_Z) to Hamming style (tr|X|Y_Z)
    """
    normalized = []
    for lab in ids:
        parts = lab.split('_', 2)  # split into at most 3 parts
        if len(parts) == 3:
            new_lab = f"{parts[0]}|{parts[1]}|{parts[2]}"
        else:
            new_lab = lab
        normalized.append(new_lab)
    return pd.Index(normalized)

K1.index = normalize_ids(K1.index)
K1.columns = normalize_ids(K1.columns)
K2.index = normalize_ids(K2.index)
K2.columns = normalize_ids(K2.columns)


# Ensure same sequences and ordering

common_seqs = K1.index.intersection(K2.index)
K1_aligned = K1.loc[common_seqs, common_seqs].values
K2_aligned = K2.loc[common_seqs, common_seqs].values

N = K1_aligned.shape[0]
print(f"Matrices loaded and aligned: {N} x {N}")

print("K1 diagonal:", np.diag(K1_aligned)[:10])
print("K2 diagonal:", np.diag(K2_aligned)[:10])

# -----------------------------
# Centering function
# -----------------------------
def center_kernel(K):
    """Double-center a kernel matrix for CKA."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

print("Centering kernels...")
K1c = center_kernel(K1_aligned)
K2c = center_kernel(K2_aligned)

# -----------------------------
# Compute CKA
# -----------------------------
def compute_cka(K, L):
    """Linear or kernel CKA."""
    hsic = np.trace(K @ L)
    norm_k = np.trace(K @ K)
    norm_l = np.trace(L @ L)
    return hsic / np.sqrt(norm_k * norm_l)

print("Computing CKA...")
cka_value = compute_cka(K1c, K2c)

print("\n==============================")
print(f"CKA similarity for {COMPARISON_OF_CKA}: {cka_value:.6f}")
print("==============================")