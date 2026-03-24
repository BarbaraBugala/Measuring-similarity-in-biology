import torch
import pandas as pd
from config import FASTA_FILE, EMBEDDINGS_FILE, RESULTS_DIR, MODEL_TYPE
from Bio import SeqIO
import torch.nn.functional as F

# Load embeddings
embeddings = torch.load(EMBEDDINGS_FILE / f"all_embeddings_{MODEL_TYPE}.pt")

# Load FASTA labels
records = list(SeqIO.parse(FASTA_FILE, "fasta"))

def normalize_id(label):
    parts = label.split("_", 2)
    return f"{parts[0]}|{parts[1]}|{parts[2]}" if len(parts) == 3 else label

labels = [normalize_id(r.id) for r in records if len(r.seq) <= 1022]

assert embeddings.shape[0] == len(labels)

# Create folders
(RESULTS_DIR / "kernels").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "distances").mkdir(parents=True, exist_ok=True)

# -----------------------------
# COSINE SIMILARITY
# -----------------------------
norm_emb = F.normalize(embeddings, dim=1)

cos_sim = torch.mm(norm_emb, norm_emb.T)
cos_distance = 1 - cos_sim
cos_kernel = (cos_sim + 1) / 2

pd.DataFrame(
    cos_distance.cpu().numpy(),
    index=labels,
    columns=labels
).to_csv(RESULTS_DIR / "distances" / f"{MODEL_TYPE}_cosine_distance.csv")

pd.DataFrame(
    cos_kernel.cpu().numpy(),
    index=labels,
    columns=labels
).to_csv(RESULTS_DIR / "kernels" / f"{MODEL_TYPE}_cosine_kernel.csv")


# -----------------------------
# EUCLIDEAN DISTANCE
# -----------------------------
dist_mat = torch.cdist(embeddings, embeddings, p=2)

pd.DataFrame(
    dist_mat.cpu().numpy(),
    index=labels,
    columns=labels
).to_csv(RESULTS_DIR / "distances" / f"{MODEL_TYPE}_euclidean_distance.csv")


# -----------------------------
# RBF KERNEL
# -----------------------------
gamma = 1 / embeddings.shape[1]

euclid_kernel = torch.exp(-gamma * dist_mat**2)

pd.DataFrame(
    euclid_kernel.cpu().numpy(),
    index=labels,
    columns=labels
).to_csv(RESULTS_DIR / "kernels" / f"{MODEL_TYPE}_euclidean_kernel.csv")