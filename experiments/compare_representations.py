#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path

from src.data import RepresentationPair
from src.qsi import EfficientQSI
from src.qsi import EfficientApproxQSI

# -----------------------------
# CONFIG
# -----------------------------
DIST_DIR = Path("results")

HAMMING = DIST_DIR / "distance_matrices_hamming/sequence_hamming_distance.csv"
BLOSUM = DIST_DIR / "distance_matrices_blosum/sequence_blosum_distance.csv"
ESM_COSINE = DIST_DIR / "esm_distance_matrices_cosine_euclidean/esm2_cosine_similarity.csv"
ESM_EUCLIDEAN = DIST_DIR / "esm_distance_matrices_cosine_euclidean/esm2_euclidean_distance.csv"

USE_APPROX = True  # safer for 4000 proteins


# -----------------------------
# LOAD MATRICES
# -----------------------------
print("Loading distance matrices...")

d_hamming = pd.read_csv(HAMMING, index_col=0).values
d_blosum = pd.read_csv(BLOSUM, index_col=0).values
d_esm_euclidean = pd.read_csv(ESM_EUCLIDEAN, index_col=0).values

# Convert cosine similarity → distance
cosine_sim = pd.read_csv(ESM_COSINE, index_col=0).values
d_esm_cosine = 1 - cosine_sim

n = d_hamming.shape[0]

print(f"Loaded matrices for {n} proteins.")


# -----------------------------
# DISTANCE FUNCTION WRAPPER
# -----------------------------
def make_distance_function(matrix):
    def d(i, j):
        return matrix[i, j]
    return d


# -----------------------------
# CREATE REPRESENTATION PAIR
# -----------------------------
X = np.arange(n)
Y = np.arange(n)

representations = RepresentationPair(
    X=X,
    Y=Y,
    d_x=make_distance_function(d_blosum),
    d_y=make_distance_function(d_esm_cosine),
)


# -----------------------------
# COMPUTE QSI
# -----------------------------
if USE_APPROX:
    qsi = EfficientApproxQSI(euclidean=False, batch_size=500, no_batches=10)
else:
    qsi = EfficientQSI(euclidean=False)

score = qsi(representations)

print("\nQSI score (Blosum vs ESM cosine):", score)