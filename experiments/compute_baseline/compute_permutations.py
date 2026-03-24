#!/usr/bin/env python3
"""
permute_benchmarks.py
Symmetrically permutes 4 distance and 4 kernel matrices using a hardcoded seed.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import RESULTS_DIR, MODEL_TYPE

# -----------------------------
# SET YOUR SEED HERE
# -----------------------------
CHOSEN_SEED = 42 

# -----------------------------
# CONFIG
# -----------------------------
DIST_DIR = RESULTS_DIR / "distances"
KERN_DIR = RESULTS_DIR / "kernels"

BASE_NAMES = [
    f"{MODEL_TYPE}_cosine",
    f"{MODEL_TYPE}_euclidean",
]

def main():
    print(f"--- Executing Permutation (Seed: {CHOSEN_SEED}) ---")
    
    # Initialize RNG once to ensure all 8 files use the EXACT same shuffle mapping
    rng = np.random.default_rng(CHOSEN_SEED)
    perm_indices = None

    for base in BASE_NAMES:
        # Define paths for both the distance and kernel versions
        tasks = [
            DIST_DIR / f"{base}_distance.csv",
            KERN_DIR / f"{base}_kernel.csv"
        ]

        for file_path in tasks:
            if not file_path.exists():
                print(f"  [Skipping] {file_path.name} (Not found)")
                continue

            # Load original matrix
            df = pd.read_csv(file_path, index_col=0)
            n = len(df)

            # Generate the shuffle order only once based on the first file's size
            if perm_indices is None:
                perm_indices = rng.permutation(n)
                print(f"  Permutation mapping created for {n} rows/columns.")
            
            # Apply symmetric permutation: reorder rows, then reorder columns
            df_permuted = df.iloc[perm_indices, perm_indices]
            
            # Update the labels to reflect the new shuffled order
            shuffled_labels = df.index.values[perm_indices]
            df_permuted.index = shuffled_labels
            df_permuted.columns = shuffled_labels

            # Save with the PERM suffix in the original directory
            output_name = f"{file_path.stem}_PERM.csv"
            output_path = file_path.parent / output_name
            
            df_permuted.to_csv(output_path)
            print(f"  [Saved] {file_path.parent.name}/{output_name}")

    print(f"\nFinished. Permuted matrices are ready for baseline testing.")

if __name__ == "__main__":
    main()