#!/usr/bin/env python3
"""
compute_pam_distance.py

Compute true PAM evolutionary distances from an MSA of proteins.
Requires aligned sequences (gaps will be ignored).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Align import substitution_matrices
from itertools import combinations
from math import log

# -----------------------------
# CONFIG
# -----------------------------
from config import RESULTS_DIR

alignment_path = RESULTS_DIR / "msa" / "aligned_mafft.fasta"
output_dir = RESULTS_DIR
output_dir.mkdir(parents=True, exist_ok=True)

distance_paths = output_dir / "distances"
distance_paths.mkdir(parents=True, exist_ok=True)

kernel_paths = output_dir / "kernels"
kernel_paths.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD PAM MATRIX AS PROBABILITIES
# -----------------------------
# PAM250 in Biopython is in scores; convert to approximate probabilities
# Formula: p ~ 2^(score/2) / Z (rough approximation)
pam_matrix = substitution_matrices.load("PAM250")
# Normalize each row to sum ~1
letters = pam_matrix.alphabet
pam_prob = {aa: {bb: 2**(pam_matrix[aa, bb]/2) for bb in letters} for aa in letters}
for aa in letters:
    s = sum(pam_prob[aa].values())
    for bb in letters:
        pam_prob[aa][bb] /= s

# -----------------------------
# LOAD MSA
# -----------------------------
records = list(SeqIO.parse(alignment_path, "fasta"))
labels = [r.id for r in records]
sequences = [str(r.seq) for r in records]
N = len(sequences)
print(f"Loaded {N} sequences.")

# -----------------------------
# COMPUTE PAM DISTANCES
# -----------------------------
dist_mat = np.zeros((N, N))

for i, j in combinations(range(N), 2):
    seq1, seq2 = sequences[i], sequences[j]
    assert len(seq1) == len(seq2), "Sequences must be aligned (MSA)."
    
    # Only consider positions with no gaps
    probs = []
    for a, b in zip(seq1, seq2):
        if a in letters and b in letters:
            probs.append(pam_prob[a][b])
    
    # Compute PAM distance
    if probs:
        avg_prob = np.mean(probs)
        distance = -log(avg_prob)
    else:
        distance = 10.0  # arbitrary large distance if no valid positions
    dist_mat[i, j] = distance
    dist_mat[j, i] = distance

# -----------------------------
# SAVE DISTANCE MATRIX
# -----------------------------
df = pd.DataFrame(dist_mat, index=labels, columns=labels)
distance_path = distance_paths / "msa_pam_distance.csv"
df.to_csv(distance_path)
print(f"PAM distance matrix saved to: {distance_path}")

# -----------------------------
# CONVERT TO MEDIAN-SCALED RBF KERNEL FOR CKA
# -----------------------------
print("Converting PAM distances to RBF kernel with median-based gamma for CKA...")

# Compute median squared distance
median_sq = np.median(dist_mat**2)
gamma = 1.0 / median_sq if median_sq > 0 else 1.0
print(f"Using gamma = {gamma:.5f}")

# Compute RBF kernel
rbf_kernel = np.exp(-gamma * dist_mat**2)

# Save kernel matrix
df_kernel = pd.DataFrame(rbf_kernel, index=labels, columns=labels)
kernel_path = kernel_paths / "msa_pam_kernel.csv"
df_kernel.to_csv(kernel_path)
print(f"RBF kernel matrix saved to: {kernel_path}")