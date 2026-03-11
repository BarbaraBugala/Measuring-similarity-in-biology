#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path

from src.data import RepresentationPair

# QSI
from src.qsi import EfficientQSI
from src.qsi import EfficientApproxQSI

# TSI
from src.tsi import EfficientTSI
from src.tsi import EfficientApproxTSI

from config import DISTANCES_QSI_1, DISTANCES_QSI_2, COMPARISON_OF_QSI


USE_APPROX = True   # recommended for large datasets (e.g. 4000 proteins)

# -----------------------------
# LOAD MATRICES
# -----------------------------
print("Loading distance matrices...")

distance_1 = pd.read_csv(DISTANCES_QSI_1, index_col=0).values
distance_2 = pd.read_csv(DISTANCES_QSI_2, index_col=0).values

n = distance_1.shape[0]

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
    d_x=make_distance_function(distance_1),
    d_y=make_distance_function(distance_2),
)


# -----------------------------
# COMPUTE QSI
# -----------------------------
print("\nComputing QSI...")

if USE_APPROX:
    qsi = EfficientApproxQSI(euclidean=False, batch_size=500, no_batches=10)
else:
    qsi = EfficientQSI(euclidean=False)

qsi_score = qsi(representations)

print(f"QSI score ({COMPARISON_OF_QSI}): {qsi_score}")


# -----------------------------
# COMPUTE TSI
# -----------------------------
print("\nComputing TSI...")

if USE_APPROX:
    tsi = EfficientApproxTSI(
        euclidean=False,
        batch_size=500,
        no_batches=10
    )
else:
    tsi = EfficientTSI(euclidean=False)

tsi_score = tsi(representations)

print(f"TSI score ({COMPARISON_OF_QSI}): {tsi_score}")