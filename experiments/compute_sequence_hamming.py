#!/usr/bin/env python3
"""
compute_sequence_distances_hamming.py

Computes pairwise sequence distances Hamming
for short protein sequences in a FASTA file,
and also converts distances to an RBF kernel for CKA.
"""

from pathlib import Path
from Bio import SeqIO
import numpy as np
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
from config import FASTA_FILE, DISTANCE_DIR

fasta = FASTA_FILE
output_dir = DISTANCE_DIR 
output_dir.mkdir(parents=True, exist_ok=True)

distance_paths = output_dir / "distances"
distance_paths.mkdir(parents=True, exist_ok=True)

kernel_paths = output_dir / "kernels"
kernel_paths.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD SEQUENCES
# -----------------------------
sequences = []
labels = []

for record in SeqIO.parse(FASTA_FILE, "fasta"):
    sequences.append(str(record.seq))
    labels.append(record.id)

N = len(sequences)
print(f"Loaded {N} sequences.")

# -----------------------------
# COMPUTE HAMMING DISTANCE MATRIX
# -----------------------------
print("Computing pairwise Hamming distances...")
dist_mat = np.zeros((N, N))

for i in range(N):
    seq_i = sequences[i]
    for j in range(i, N):
        seq_j = sequences[j]
        # Hamming distance / normalized
        diff = sum(a != b for a, b in zip(seq_i, seq_j)) / len(seq_i)
        dist_mat[i, j] = diff
        dist_mat[j, i] = diff  # symmetric

# -----------------------------
# SAVE DISTANCE MATRIX AS CSV
# -----------------------------
df = pd.DataFrame(dist_mat, index=labels, columns=labels)
distance_path = distance_paths / "sequence_hamming_distance.csv"
df.to_csv(distance_path)
print(f"Distance matrix saved to: {distance_path}")

# -----------------------------
# CONVERT TO RBF KERNEL
# -----------------------------
print("Converting Hamming distances to RBF kernel for CKA...")
# Heuristic gamma: 1 / median squared distance
median_sq = np.median(dist_mat**2)
gamma = 1.0 / median_sq if median_sq > 0 else 1.0

rbf_kernel = np.exp(-gamma * dist_mat**2)
df_kernel = pd.DataFrame(rbf_kernel, index=labels, columns=labels)
kernel_path = kernel_paths / "sequence_hamming_kernel.csv"
df_kernel.to_csv(kernel_path)
print(f"RBF kernel matrix saved to: {kernel_path}")