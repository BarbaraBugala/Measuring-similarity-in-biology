import torch
import pandas as pd
from config import FASTA_FILE, EMBEDDINGS_FILE, DISTANCE_DIR
from Bio import SeqIO

# Load all embeddings (tensor NxD)
embeddings = torch.load(EMBEDDINGS_FILE / "all_embeddings.pt")  # [N, D]

# Load FASTA to get labels
records = list(SeqIO.parse(FASTA_FILE, "fasta"))

def normalize_id(label):
    parts = label.split("_", 2)
    return f"{parts[0]}|{parts[1]}|{parts[2]}" if len(parts) == 3 else label

labels = [normalize_id(r.id) for r in records if len(r.seq) <= 1022]

# sanity check
assert embeddings.shape[0] == len(labels)

# Cosine similarity kernel
norm_emb = torch.nn.functional.normalize(embeddings, dim=1)
cos_sim = (torch.mm(norm_emb, norm_emb.T) + 1) / 2  # scale to [0,1]
cos_kernel_df = pd.DataFrame(cos_sim.cpu().numpy(), index=labels, columns=labels)
cos_kernel_df.to_csv(DISTANCE_DIR / "kernels" / "esm2_cosine_kernel.csv")

# Euclidean distance and RBF kernel
dist_mat = torch.cdist(embeddings, embeddings, p=2)
euclid_kernel = torch.exp(-dist_mat**2 / embeddings.shape[1])
euclid_kernel_df = pd.DataFrame(euclid_kernel.cpu().numpy(), index=labels, columns=labels)
euclid_kernel_df.to_csv(DISTANCE_DIR / "kernels" / "esm2_euclidean_kernel.csv")