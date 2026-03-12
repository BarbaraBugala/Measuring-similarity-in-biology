#!/usr/bin/env python3
"""
plot_qsi_tsi_heatmap.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PATHS
# -----------------------------
from config import PLOT_DIR, RESULTS_DIR

scores_file = RESULTS_DIR / "similarities" / "qsi_tsi_scores.csv"

plots_dir = PLOT_DIR
plots_dir.mkdir(parents=True, exist_ok=True)

qsi_output = plots_dir / "qsi_heatmap.png"
tsi_output = plots_dir / "tsi_heatmap.png"


# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading QSI/TSI data...")
df = pd.read_csv(scores_file)


# -----------------------------
# PIVOT MATRICES
# -----------------------------
qsi_matrix = df.pivot_table(
    index="embedding_method",
    columns="sequence_method",
    values="qsi"
)

tsi_matrix = df.pivot_table(
    index="embedding_method",
    columns="sequence_method",
    values="tsi"
)

print("\nQSI matrix:")
print(qsi_matrix)

print("\nTSI matrix:")
print(tsi_matrix)


# -----------------------------
# QSI HEATMAP
# -----------------------------
plt.figure(figsize=(8, 6))

sns.heatmap(
    qsi_matrix,
    annot=True,
    cmap="viridis",
    linewidths=0.5,
    fmt=".2f",
    cbar_kws={"label": "QSI similarity"}
)

plt.title("QSI Similarity: Embeddings vs Sequence Methods")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(qsi_output, dpi=300)
plt.close()

print(f"\nQSI heatmap saved to: {qsi_output}")


# -----------------------------
# TSI HEATMAP
# -----------------------------
plt.figure(figsize=(8, 6))

sns.heatmap(
    tsi_matrix,
    annot=True,
    cmap="viridis",
    linewidths=0.5,
    fmt=".2f",
    cbar_kws={"label": "TSI similarity"}
)

plt.title("TSI Similarity: Embeddings vs Sequence Methods")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(tsi_output, dpi=300)
plt.close()

print(f"\nTSI heatmap saved to: {tsi_output}")