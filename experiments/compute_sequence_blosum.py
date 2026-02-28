#!/usr/bin/env python3
"""
compute_sequence_blosum.py

Computes pairwise distances between sequences using the modern Biopython 
PairwiseAligner and multiprocessing for speed.
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
FASTA_FILE = Path("data/curated-AMPs.fasta")
OUTPUT_DIR = Path("results/distance_matrices_blosum")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_aligner():
    """Initializes a fast C-based aligner."""
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
        
        # Normalization: 1 - (score / max possible score for these two)
        max_self = max(self_scores[i], self_scores[j])
        
        if max_self > 0:
            distance = 1 - (score / max_self)
        else:
            distance = 1.0
            
        # Ensure distance is non-negative (handles very dissimilar sequences)
        row[j] = max(0.0, distance)
    
    return i, row

def main():
    # -----------------------------
    # LOAD SEQUENCES
    # -----------------------------
    if not FASTA_FILE.exists():
        print(f"Error: {FASTA_FILE} not found.")
        return

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
    
    # Use a pool of workers to compute rows in parallel
    with Pool(processes=cpu_count()) as pool:
        func = partial(compute_row, sequences=sequences, self_scores=self_scores, n_seqs=N)
        results = pool.map(func, range(N))

    # Assemble the symmetric matrix
    for i, row in results:
        dist_mat[i, i:] = row[i:]
        # Mirror the upper triangle to the lower triangle
        for j in range(i + 1, N):
            dist_mat[j, i] = dist_mat[i, j]

    # -----------------------------
    # SAVE AS CSV
    # -----------------------------
    df = pd.DataFrame(dist_mat, index=labels, columns=labels)
    output_path = OUTPUT_DIR / "sequence_blosum_distance.csv"
    df.to_csv(output_path)
    print(f"Success! Matrix saved to: {output_path}")

if __name__ == "__main__":
    main()