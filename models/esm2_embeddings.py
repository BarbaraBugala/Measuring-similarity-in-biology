import torch
import esm
from Bio import SeqIO
from pathlib import Path
import re

# -----------------------------
# CONFIG
# -----------------------------
FASTA_FILE = "data/curated-AMPs.fasta"
OUTPUT_DIR = "data/embeddings_esm2_curated-AMPs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 5  # Reduce this to 1 if it still crashes; increase if you have high RAM
MAX_LEN = 1022  # ESM-2 limit (1024 minus start/stop tokens)

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
# Using the 35M parameter model as per your original code
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
model = model.to(DEVICE)
model.eval()
batch_converter = alphabet.get_batch_converter()

# -----------------------------
# LOAD & FILTER SEQUENCES
# -----------------------------
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
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

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
        token_reps = results["representations"][12]

        # Extract and save each protein in the batch
        for i, (label, seq) in enumerate(batch_data):
            seq_len = len(seq)
            
            # ESM-2 adds a start token at [0], so the sequence is at [1 : seq_len+1]
            # .mean(0) produces the "Per-Protein" embedding
            embedding = token_reps[i, 1 : seq_len + 1].mean(0).cpu()

            # Save individual file
            safe_label = sanitize_filename(label)
            torch.save(embedding, Path(OUTPUT_DIR) / f"{safe_label}.pt")

        # Memory Cleanup
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        if (batch_idx + 1) % 5 == 0:
            print(f"Processed {(batch_idx + 1) * BATCH_SIZE} sequences...")

print(f"\nSuccess! All individual embeddings saved in: {OUTPUT_DIR}")
print("Note: 'all_embeddings.pt' was skipped to save RAM. "
      "You can load individual .pt files as needed.")