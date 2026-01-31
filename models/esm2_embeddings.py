import torch
import esm
from Bio import SeqIO  # pip install biopython
from pathlib import Path
import re  # for sanitizing filenames

# -----------------------------
# CONFIG
# -----------------------------
FASTA_FILE = "data/proteins.fasta"       # your dataset
OUTPUT_DIR = "data/embeddings_esm2"     # folder to save embeddings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# HELPER FUNCTION TO SANITIZE FILENAMES
# -----------------------------
def sanitize_filename(name):
    # Replace invalid Windows filename characters with underscore
    return re.sub(r'[<>:"/\\|?* ]', '_', name)

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading ESM-2 model...")
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
model = model.to(DEVICE)
model.eval()
batch_converter = alphabet.get_batch_converter()

# -----------------------------
# LOAD SEQUENCES
# -----------------------------
sequences = [(record.id, str(record.seq)) for record in SeqIO.parse(FASTA_FILE, "fasta")]
if not sequences:
    raise ValueError("No sequences found in the FASTA file!")

# -----------------------------
# CREATE OUTPUT DIRECTORY
# -----------------------------
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------------
# CONVERT TO TOKENS
# -----------------------------
batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
batch_tokens = batch_tokens.to(DEVICE)

# -----------------------------
# COMPUTE EMBEDDINGS
# -----------------------------
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[12], return_contacts=False)
    token_representations = results["representations"][12]

# -----------------------------
# SAVE EMBEDDINGS
# -----------------------------
sequence_embeddings = {}
for i, (label, seq) in enumerate(sequences):
    seq_len = len(seq)
    embedding = token_representations[i, 1:seq_len+1].mean(0).cpu()
    sequence_embeddings[label] = embedding

    # SANITIZE label before saving
    safe_label = sanitize_filename(label)
    torch.save(embedding, Path(OUTPUT_DIR) / f"{safe_label}.pt")

# Save all embeddings together in one file
torch.save(sequence_embeddings, Path(OUTPUT_DIR) / "all_embeddings.pt")

print(f"All embeddings saved in {OUTPUT_DIR}")
