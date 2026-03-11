#!/usr/bin/env python3
"""
plot_distance_heatmap.py

Plot a heatmap of a protein distance matrix.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PATHS
# -----------------------------
from config import RESULTS_DIR, PLOT_DIR

distance_file = RESULTS_DIR / "distances" / "esm2_cosine_distance.csv"

plots_dir = PLOT_DIR
plots_dir.mkdir(parents=True, exist_ok=True)

output_file = plots_dir / "esm2_cosine_distance_heatmap.png"

# -----------------------------
# LOAD MATRIX
# -----------------------------
print("Loading distance matrix...")

df = pd.read_csv(distance_file, index_col=0)

print(df.shape)

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(10, 8))

sns.heatmap(
    df,
    cmap="magma",     # good for distances
    square=True,
    xticklabels=False,
    yticklabels=False,
    cbar_kws={"label": "Distance"}
)

plt.title("ESM2 Cosine Distance Between Proteins")
plt.tight_layout()

# -----------------------------
# SAVE
# -----------------------------
plt.savefig(output_file, dpi=300)
plt.close()

print(f"Saved to {output_file}")