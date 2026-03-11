#!/usr/bin/env python3
"""
compute_esm_embeddings_raw.py

Compute ESM-2 embeddings for a set of protein sequences from a FASTA file
and save all embeddings stacked in a single .pt file, preserving the FASTA order.
"""

import torch
import esm
from Bio import SeqIO
from pathlib import Path
import torch.nn.functional as F
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
from config import FASTA_FILE, EMBEDDINGS_FILE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 5
MAX_LEN = 1022  # ESM token limit

output_dir = Path(EMBEDDINGS_FILE)
output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]

# -----------------------------
# LOAD MODEL
# -----------------------------
print(f"Loading ESM-2 model on {DEVICE}...")
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
model = model.to(DEVICE)
model.eval()
batch_converter = alphabet.get_batch_converter()

# -----------------------------
# LOAD & FILTER SEQUENCES
# -----------------------------
all_records = list(SeqIO.parse(FASTA_FILE, "fasta"))
all_sequences = []
skipped = 0

for record in all_records:
    seq_str = str(record.seq)
    if len(seq_str) <= MAX_LEN:
        all_sequences.append((record.id, seq_str))
    else:
        skipped += 1

if skipped > 0:
    print(f"Skipped {skipped} sequences longer than {MAX_LEN} amino acids.")

if not all_sequences:
    raise ValueError("No valid sequences found!")

print(f"{len(all_sequences)} sequences will be processed.")

# -----------------------------
# COMPUTE & SAVE EMBEDDINGS
# -----------------------------
stacked_embeddings = []

with torch.no_grad():
    for batch_idx, batch_data in enumerate(batch_generator(all_sequences, BATCH_SIZE)):
        labels, strs, tokens = batch_converter(batch_data)
        tokens = tokens.to(DEVICE)

        results = model(tokens, repr_layers=[12], return_contacts=False)
        token_reps = results["representations"][12]

        for i, (label, seq) in enumerate(batch_data):
            seq_len = len(seq)
            embedding = token_reps[i, 1 : seq_len + 1].mean(0).cpu()
            stacked_embeddings.append(embedding)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        if (batch_idx + 1) % 5 == 0:
            print(f"Processed {(batch_idx + 1) * BATCH_SIZE} sequences...")

# -----------------------------
# SAVE STACKED EMBEDDINGS
# -----------------------------
stacked_tensor = torch.stack(stacked_embeddings)
stacked_path = output_dir / "all_embeddings.pt"
torch.save(stacked_tensor, stacked_path)
print(f"\nAll embeddings stacked and saved to {stacked_path}")