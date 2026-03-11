#!/usr/bin/env python3
"""
compute_sequence_blast.py

Performs an all-vs-all BLASTP for protein sequences, computes a distance matrix,
and produces a kernel matrix suitable for CKA.

Distance definition:
    distance = 1 - percent_identity / 100
"""

from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from Bio import SeqIO

# -----------------------------
# CONFIG
# -----------------------------
from config import FASTA_FILE, DISTANCE_DIR

fasta = FASTA_FILE
output_dir = DISTANCE_DIR 
output_dir.mkdir(parents=True, exist_ok=True)

db_name = output_dir / "blast" / "blast_db"
blast_output = output_dir / "blast_results.tsv"
distance_csv = output_dir / "sequence_blast_distance.csv"
kernel_csv = output_dir / "sequence_blast_kernel.csv"

# -----------------------------
# LOAD SEQUENCE LABELS
# -----------------------------
print("Loading sequence labels...")
labels = [record.id for record in SeqIO.parse(fasta, "fasta")]
N = len(labels)
label_to_idx = {label: i for i, label in enumerate(labels)}

# -----------------------------
# MAKE BLAST DATABASE
# -----------------------------
print("Creating BLAST database...")
subprocess.run([
    "makeblastdb",
    "-in", str(fasta),
    "-dbtype", "prot",
    "-out", str(db_name)
], check=True)

# -----------------------------
# RUN ALL-VS-ALL BLAST
# -----------------------------
print("Running all-vs-all BLASTP...")
subprocess.run([
    "blastp",
    "-query", str(fasta),
    "-db", str(db_name),
    "-out", str(blast_output),
    "-outfmt", "6 qseqid sseqid pident"
], check=True)

# -----------------------------
# PARSE RESULTS
# -----------------------------
print("Parsing BLAST results...")
best_identity = {}
with open(blast_output) as f:
    for line in f:
        qid, sid, pident = line.strip().split()
        pident = float(pident)
        key = (qid, sid)
        if key not in best_identity or pident > best_identity[key]:
            best_identity[key] = pident

# -----------------------------
# BUILD DISTANCE MATRIX
# -----------------------------
print("Building distance matrix...")
dist_mat = np.ones((N, N))
for (qid, sid), pident in best_identity.items():
    i = label_to_idx[qid]
    j = label_to_idx[sid]
    distance = 1 - pident / 100
    dist_mat[i, j] = distance

# Symmetrize
for i in range(N):
    for j in range(i + 1, N):
        dist_mat[j, i] = dist_mat[i, j]

# Self-distance = 0
np.fill_diagonal(dist_mat, 0)

# -----------------------------
# SAVE DISTANCE MATRIX
# -----------------------------
df = pd.DataFrame(dist_mat, index=labels, columns=labels)
df.to_csv(distance_csv)
print(f"BLAST distance matrix saved to: {distance_csv}")

# -----------------------------
# CONVERT TO KERNEL FOR CKA
# -----------------------------
print("Converting BLAST distances to 1-D kernel for CKA...")
kernel = 1.0 - dist_mat  # diagonal = 1, off-diagonal = similarity
df_kernel = pd.DataFrame(kernel, index=labels, columns=labels)
df_kernel.to_csv(kernel_csv)
print(f"BLAST kernel matrix saved to: {kernel_csv}")