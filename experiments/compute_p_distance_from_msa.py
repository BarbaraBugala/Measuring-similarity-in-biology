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
    BASE_DIR = Path(__file__).resolve().parent.parent
    alignment_path = BASE_DIR / "results" / "msa" / "aligned_mafft.fasta"
    output_dir = BASE_DIR / "results" / "distance_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = list(SeqIO.parse(alignment_path, "fasta"))
    ids = [record.id for record in sequences]
    n = len(sequences)

    print(f"Loaded {n} aligned sequences.")

    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = p_distance(str(sequences[i].seq), str(sequences[j].seq))
            matrix[i, j] = d
            matrix[j, i] = d

    df = pd.DataFrame(matrix, index=ids, columns=ids)
    output_file = output_dir / "msa_p_distance.csv"
    df.to_csv(output_file)

    print(f"Saved p-distance matrix to {output_file}")


if __name__ == "__main__":
    main()