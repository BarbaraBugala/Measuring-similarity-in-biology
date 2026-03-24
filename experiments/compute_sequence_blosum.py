#!/usr/bin/env python3
"""
compute_sequence_blosum.py

Computes pairwise distances between sequences using Biopython 
PairwiseAligner and multiprocessing for speed, and also produces
a kernel matrix suitable for CKA.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO, Align
from Bio.Align import substitution_matrices
from multiprocessing import Pool, cpu_count
from functools import partial

# -----------------------------
# CONFIG
# -----------------------------
from config import FASTA_FILE, RESULTS_DIR

fasta = FASTA_FILE
output_dir = RESULTS_DIR 
output_dir.mkdir(parents=True, exist_ok=True)

distance_paths = output_dir / "distances"
distance_paths.mkdir(parents=True, exist_ok=True)

kernel_paths = output_dir / "kernels"
kernel_paths.mkdir(parents=True, exist_ok=True)

def get_aligner():
    """Needleman Wunsch algo"""
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    aligner.mode = 'global'
    return aligner

def compute_row(i, sequences, self_scores, n_seqs):
    """Computes a single row of the distance matrix."""
    aligner = get_aligner()
    row = np.zeros(n_seqs)
    seq_i = sequences[i]
    
    for j in range(i + 1, n_seqs):
        score = aligner.score(seq_i, sequences[j])
        max_self = max(self_scores[i], self_scores[j])
        distance = 1 - (score / max_self) if max_self > 0 else 1.0
        row[j] = max(0.0, distance)
    
    return i, row

def main():
    # -----------------------------
    # LOAD SEQUENCES
    # -----------------------------
    records = list(SeqIO.parse(FASTA_FILE, "fasta"))
    sequences = [str(r.seq) for r in records]
    labels = [r.id for r in records]
    N = len(sequences)
    print(f"Loaded {N} sequences.")

    # -----------------------------
    # PRECOMPUTE SELF-SCORES
    # -----------------------------
    print("Precomputing self-alignment scores...")
    aligner = get_aligner()
    self_scores = [aligner.score(s, s) for s in sequences]

    # -----------------------------
    # PARALLEL COMPUTATION
    # -----------------------------
    print(f"Computing distance matrix using {cpu_count()} cores...")
    dist_mat = np.zeros((N, N))
    
    with Pool(processes=cpu_count()) as pool:
        func = partial(compute_row, sequences=sequences, self_scores=self_scores, n_seqs=N)
        results = pool.map(func, range(N))

    # Assemble symmetric matrix
    for i, row in results:
        dist_mat[i, i:] = row[i:]
        for j in range(i + 1, N):
            dist_mat[j, i] = dist_mat[i, j]

    # -----------------------------
    # SAVE DISTANCE MATRIX
    # -----------------------------
    df = pd.DataFrame(dist_mat, index=labels, columns=labels)
    distance_path = distance_paths / "sequence_blosum_distance.csv"
    df.to_csv(distance_path)
    print(f"Distance matrix saved to: {distance_path}")

    # -----------------------------
    # CONVERT TO KERNEL FOR CKA
    # -----------------------------
    print("Converting BLOSUM distances to 1-D kernel for CKA...")
    kernel = 1.0 - dist_mat  # diagonal = 1, off-diagonal = similarity
    df_kernel = pd.DataFrame(kernel, index=labels, columns=labels)
    kernel_path = kernel_paths / "sequence_blosum_kernel.csv"
    df_kernel.to_csv(kernel_path)
    print(f"Kernel matrix saved to: {kernel_path}")

if __name__ == "__main__":
    main()