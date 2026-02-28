#!/usr/bin/env python3
"""
compute_sequence_distances.py

Computes pairwise sequence distances (Hamming or optional BLOSUM-based) 
for short protein sequences in a FASTA file.
"""

from pathlib import Path
from Bio import SeqIO
import numpy as np
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
FASTA_FILE = Path("data/curated-AMPs.fasta")
OUTPUT_DIR = Path("results/distance_matrices_hamming")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
# SAVE AS CSV
# -----------------------------
df = pd.DataFrame(dist_mat, index=labels, columns=labels)
output_path = OUTPUT_DIR / "sequence_hamming_distance.csv"
df.to_csv(output_path)
print(f"Distance matrix saved to: {output_path}")