import torch
import esm

# 1. Load pretrained ESM-2 model (CPU-friendly)
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D() 
model.eval()  # set to evaluation mode

# 2. Create a batch converter
batch_converter = alphabet.get_batch_converter()

# 3. Example protein sequences (list of tuples: (name, sequence))
sequences = [
    ("protein1", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"),
    ("protein2", "GATAYIAKQRQISFVKSHFSRQLEERLGLIEVQ")
]

batch_labels, batch_strs, batch_tokens = batch_converter(sequences)

# 4. Run the model (get embeddings)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[12], return_contacts=False)
    token_representations = results["representations"][12]

# 5. Get per-sequence embeddings by averaging over tokens (excluding padding)
sequence_embeddings = []
for i, (label, seq) in enumerate(sequences):
    # mask padding tokens
    seq_len = len(seq)
    embedding = token_representations[i, 1:seq_len+1].mean(0)
    sequence_embeddings.append((label, embedding))

# 6. Print embeddings
for label, emb in sequence_embeddings:
    print(f"{label} embedding shape: {emb.shape}")
    print(emb)
