#!/usr/bin/env python3
"""
compute_pairwise_CKA_no_matching.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config import RESULTS_DIR, BASELINE_KERNEL_FILES, BASELINE_EMB_METHODS, BASELINE_SEQ_METHODS, MODEL_TYPE

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_FILE = RESULTS_DIR / "similarities" / MODEL_TYPE / "cka_scores_baseline.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# CKA IMPLEMENTATION
# -----------------------------
def center_kernel(K):
    """
    Centers the kernel matrix: K_centered = (I - 1/n) K (I - 1/n)
    """
    n = K.shape[0]
    # Using a more memory-efficient centering approach for large n
    unit = np.ones((n, n)) / n
    return K - unit @ K - K @ unit + unit @ K @ unit

def compute_cka(K, L):
    """
    Computes Centered Kernel Alignment (CKA).
    """
    Kc = center_kernel(K)
    Lc = center_kernel(L)
    
    # HSIC is the Frobenius inner product of centered kernels
    hsic = np.sum(Kc * Lc) 
    # Norms for scaling
    norm_k = np.linalg.norm(Kc)
    norm_l = np.linalg.norm(Lc)
    
    return hsic / (norm_k * norm_l)

# -----------------------------
# LOAD ALL KERNELS
# -----------------------------
print("Loading kernel matrices...")
kernels = {}

for name, path in BASELINE_KERNEL_FILES.items():
    # Use .values immediately to drop name awareness
    df = pd.read_csv(path, index_col=0)
    kernels[name] = df.values 
    print(f"  Loaded {name}: {df.shape}")

# -----------------------------
# COMPUTE CKA (Index-to-Index)
# -----------------------------
results = []

for esm in BASELINE_EMB_METHODS:
    if esm not in kernels: continue
    K = kernels[esm]
    
    for seq in BASELINE_SEQ_METHODS:
        if seq not in kernels: continue
        L = kernels[seq]
        
        # Check if dimensions match
        if K.shape != L.shape:
            print(f"  [Warning] Shapes mismatch: {esm} {K.shape} vs {seq} {L.shape}. Skipping.")
            continue

        print(f"Computing CKA (Positional): {esm} vs {seq}")
        
        # We compare them exactly as they are ordered in the files
        score = compute_cka(K, L)
        
        results.append({
            "esm_method": esm,
            "sequence_method": seq,
            "cka": score
        })

# -----------------------------
# SAVE
# -----------------------------
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_FILE, index=False)
print("\nResults saved to:", OUTPUT_FILE)