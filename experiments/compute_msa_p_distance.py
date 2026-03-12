#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO

def p_distance(seq1, seq2):
    mismatches = 0
    valid_sites = 0
    for a, b in zip(seq1, seq2):
        if a == "-" or b == "-":
            continue
        valid_sites += 1
        if a != b:
            mismatches += 1
    if valid_sites == 0:
        return 0.0
    return mismatches / valid_sites

def main():
    from config import FASTA_FILE, DISTANCE_DIR

    alignment_path = DISTANCE_DIR / "msa" / "aligned_mafft.fasta"
    output_dir = DISTANCE_DIR 
    output_dir.mkdir(parents=True, exist_ok=True)

    distance_paths = output_dir / "distances"
    distance_paths.mkdir(parents=True, exist_ok=True)

    kernel_paths = output_dir / "kernels"
    kernel_paths.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # LOAD ALIGNED SEQUENCES
    # -----------------------------
    sequences = list(SeqIO.parse(alignment_path, "fasta"))
    ids = [record.id for record in sequences]
    n = len(sequences)
    print(f"Loaded {n} aligned sequences.")

    # -----------------------------
    # COMPUTE P-DISTANCE MATRIX
    # -----------------------------
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = p_distance(str(sequences[i].seq), str(sequences[j].seq))
            matrix[i, j] = d
            matrix[j, i] = d

    df = pd.DataFrame(matrix, index=ids, columns=ids)
    distance_file = distance_paths / "msa_p_distance.csv"
    df.to_csv(distance_file)
    print(f"Saved p-distance matrix to {distance_file}")

    # -----------------------------
    # CONVERT TO KERNEL FOR CKA
    # -----------------------------
    print("Converting p-distance to 1-D kernel for CKA...")
    kernel = 1.0 - matrix  # diagonal = 1, off-diagonal = similarity
    df_kernel = pd.DataFrame(kernel, index=ids, columns=ids)
    kernel_file = kernel_paths / "msa_p_kernel.csv"
    df_kernel.to_csv(kernel_file)
    print(f"Saved kernel matrix to {kernel_file}")

if __name__ == "__main__":
    main()