import torch
import esm
from Bio import SeqIO
from pathlib import Path
import re

# -----------------------------
# CONFIG
# -----------------------------
from config import FASTA_FILE, EMBEDDINGS_FILE


output_dir = EMBEDDINGS_FILE 
output_dir.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 5 
MAX_LEN = 1022 #limit of 1024 tokens, minus start and end, protein sequence can't exceed 1022 AAs

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?* ]', '_', name)

def batch_generator(data, batch_size):
    """Yield successive n-sized batches from data."""
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]

# -----------------------------
# LOAD MODEL
# -----------------------------
print(f"Loading ESM-2 model on {DEVICE}...")
# 35M params, 12 transformer layers
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
model = model.to(DEVICE)
model.eval()
batch_converter = alphabet.get_batch_converter()

# -----------------------------
# LOAD & FILTER SEQUENCES
# -----------------------------
# with skipping sequences longer than 1022 AAs
print("Reading FASTA file...")
all_sequences = []
skipped = 0 
for record in SeqIO.parse(FASTA_FILE, "fasta"):
    seq_str = str(record.seq)
    if len(seq_str) <= MAX_LEN:
        all_sequences.append((record.id, seq_str))
    else:
        skipped += 1

if skipped > 0:
    print(f"Skipped {skipped} sequences longer than {MAX_LEN} amino acids.")

if not all_sequences:
    raise ValueError("No valid sequences found!")

# -----------------------------
# CREATE OUTPUT DIRECTORY
# -----------------------------
Path(output_dir).mkdir(parents=True, exist_ok=True)

# -----------------------------
# COMPUTE & SAVE EMBEDDINGS (BATCHED)
# -----------------------------
print(f"Processing {len(all_sequences)} sequences in batches of {BATCH_SIZE}...")

with torch.no_grad():
    for batch_idx, batch_data in enumerate(batch_generator(all_sequences, BATCH_SIZE)):
        # Convert batch to tokens
        labels, strs, tokens = batch_converter(batch_data)
        tokens = tokens.to(DEVICE)

        # Forward pass
        # Layer 12 is the final layer for the 35M model
        results = model(tokens, repr_layers=[12], return_contacts=False)
        token_reps = results["representations"][12] # [batch_size, seq_len, 480] in this case

        # Extract and save each protein in the batch
        for i, (label, seq) in enumerate(batch_data):
            seq_len = len(seq)
            
            # ESM-2 adds a start token at [0], so the sequence is at [1 : seq_len+1]
            # .mean(0) average over amino acids (maybe its a good idea to add linear layer to transform it into vector)
            embedding = token_reps[i, 1 : seq_len + 1].mean(0).cpu()

            # Save individual file
            safe_label = sanitize_filename(label)
            torch.save(embedding, Path(output_dir) / f"{safe_label}.pt")

        # Memory Cleanup
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        if (batch_idx + 1) % 5 == 0:
            print(f"Processed {(batch_idx + 1) * BATCH_SIZE} sequences...")

print(f"\nSuccess! All individual embeddings saved in: {output_dir}")
print("Note: 'all_embeddings.pt' was skipped to save RAM. "
      "You can load individual .pt files as needed.")