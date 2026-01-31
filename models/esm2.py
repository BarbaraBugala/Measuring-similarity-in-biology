import torch
import esm

# Step 1: choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 2: load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.eval()
model = model.to(device)

# Step 3: prepare sequences
sequences = [
    ("protein1", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"),
    ("protein2", "GILFVGSGVSGE"),
]

batch_converter = alphabet.get_batch_converter()
_, _, tokens = batch_converter(sequences)
tokens = tokens.to(device)

# Step 4: run inference
with torch.no_grad():
    results = model(tokens, repr_layers=[33])

# Step 5: extract embeddings
representations = results["representations"][33]

for i, (_, seq) in enumerate(sequences):
    embedding = representations[i, 1 : len(seq) + 1].mean(0)
    print(f"{sequences[i][0]} embedding shape:", embedding.shape)
