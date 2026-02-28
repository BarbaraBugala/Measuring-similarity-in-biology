#!/usr/bin/env python3
"""
compute_sequence_blast.py

Performs an all-vs-all BLASTP for short peptide sequences and computes distance matrix.
Distance is computed as: distance = 1 - percent_identity / 100

Dependencies:
    - NCBI BLAST+ command-line tools installed (blastp, makeblastdb)
"""

from pathlib import Path
import pandas as pd
import subprocess

# -----------------------------
# CONFIG
# -----------------------------
FASTA_FILE = Path("data/curated-AMPs.fasta")
OUTPUT_DIR = Path("results/distance_matrices")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DB_NAME = "curated_db"
BLAST_OUTPUT = OUTPUT_DIR / "blast_results.tsv"
DISTANCE_CSV = OUTPUT_DIR / "sequence_blast_distance.csv"

# -----------------------------
# MAKE BLAST DATABASE
# -----------------------------
print("Creating BLAST database...")
subprocess.run([
    "makeblastdb",
    "-in", str(FASTA_FILE),
    "-dbtype", "prot",
    "-out", DB_NAME
], check=True)

# -----------------------------
# RUN ALL-VS-ALL BLAST
# -----------------------------
print("Running all-vs-all BLASTP...")
subprocess.run([
    "blastp",
    "-query", str(FASTA_FILE),
    "-db", DB_NAME,
    "-out", str(BLAST_OUTPUT),
    "-outfmt", "6 qseqid sseqid pident",
    "-word_size", "2",          # short sequences
    "-matrix", "BLOSUM80"       # better for short peptides
], check=True)

# -----------------------------
# PARSE RESULTS AND BUILD DISTANCE MATRIX
# -----------------------------
print("Parsing BLAST results and building distance matrix...")
# Load sequences to get labels
labels = [record.id for record in SeqIO.parse(FASTA_FILE, "fasta")]
N = len(labels)
label_to_idx = {label: i for i, label in enumerate(labels)}

# Initialize distance matrix
import numpy as np
dist_mat = np.ones((N, N))  # default 1 for max distance

# Read BLAST output
with open(BLAST_OUTPUT) as f:
    for line in f:
        qid, sid, pident = line.strip().split()
        i = label_to_idx[qid]
        j = label_to_idx[sid]
        distance = 1 - float(pident)/100
        dist_mat[i, j] = distance

# Symmetrize the matrix
for i in range(N):
    for j in range(i+1, N):
        dist_mat[j, i] = dist_mat[i, j]

# Save as CSV
df = pd.DataFrame(dist_mat, index=labels, columns=labels)
df.to_csv(DISTANCE_CSV)
print(f"BLAST distance matrix saved to: {DISTANCE_CSV}")