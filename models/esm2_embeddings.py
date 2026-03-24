#!/usr/bin/env python3
import torch
import esm
from Bio import SeqIO
from pathlib import Path

# Import your prettier config
from config import FASTA_FILE, EMBEDDINGS_FILE, MODEL_TYPE

# -----------------------------
# MODEL MAPPING
# -----------------------------
# Maps your custom MODEL_TYPE strings to (Official Name, Representation Layer)
MODEL_MAP = {
    "esm2_8M":   ("esm2_t6_8M_UR50D", 6),
    "esm2_35M":  ("esm2_t12_35M_UR50D", 12),
    "esm2_150M": ("esm2_t30_150M_UR50D", 30),
    "esm2_650M": ("esm2_t33_650M_UR50D", 33),
    "esm2_3B":   ("esm2_t36_3B_UR50D", 36),
    "esm2_15B":  ("esm2_t48_15B_UR50D", 48),
}

if MODEL_TYPE not in MODEL_MAP:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}. Please use one of {list(MODEL_MAP.keys())}")

OFFICIAL_NAME, REPR_LAYER = MODEL_MAP[MODEL_TYPE]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 5
MAX_LEN = 1022 

# -----------------------------
# LOAD MODEL & DATA
# -----------------------------
print(f"Loading {MODEL_TYPE} ({OFFICIAL_NAME}) on {DEVICE}...")
model, alphabet = esm.pretrained.load_model_and_alphabet(OFFICIAL_NAME)
model = model.to(DEVICE)
model.eval()

batch_converter = alphabet.get_batch_converter()
output_dir = Path(EMBEDDINGS_FILE)
output_dir.mkdir(parents=True, exist_ok=True)

all_records = list(SeqIO.parse(FASTA_FILE, "fasta"))
# Truncate to MAX_LEN to avoid positional embedding errors
all_sequences = [(r.id, str(r.seq)[:MAX_LEN]) for r in all_records]

# -----------------------------
# INFERENCE LOOP
# -----------------------------
stacked_embeddings = []

with torch.no_grad():
    for i in range(0, len(all_sequences), BATCH_SIZE):
        batch_data = all_sequences[i : i + BATCH_SIZE]
        labels, strs, tokens = batch_converter(batch_data)
        tokens = tokens.to(DEVICE)

        results = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
        token_reps = results["representations"][REPR_LAYER]

        for j, (label, seq) in enumerate(batch_data):
            # Mean pooling: average over the actual sequence residues 
            # (index 0 is <cls>, index len+1 is <eos>)
            mean_emb = token_reps[j, 1 : len(seq) + 1].mean(0).cpu()
            stacked_embeddings.append(mean_emb)
        
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"Processed {i}/{len(all_sequences)} sequences...")

# -----------------------------
# SAVE OUTPUT
# -----------------------------
stacked_tensor = torch.stack(stacked_embeddings)
# Saving with your specific naming convention
save_path = output_dir / f"all_embeddings_{MODEL_TYPE}.pt"
torch.save(stacked_tensor, save_path)

print(f"\n✅ Done! Saved {stacked_tensor.shape} embeddings of size {stacked_tensor.shape}")
print(f"File location: {save_path}")