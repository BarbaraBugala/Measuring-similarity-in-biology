#!/usr/bin/env python3
"""
plot_cka_heatmap.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PATHS
# -----------------------------
from config import PLOT_DIR, RESULTS_DIR

cka_file = RESULTS_DIR / "similarities" / "cka_scores.csv"

plots_dir = PLOT_DIR
plots_dir.mkdir(parents=True, exist_ok=True)

output_file = plots_dir / "cka_heatmap.png"

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading CKA data...")
df = pd.read_csv(cka_file)

# -----------------------------
# PIVOT TO MATRIX
# -----------------------------
cka_matrix = df.pivot_table(
    index="esm_method",
    columns="sequence_method",
    values="cka"
)

print("\nCKA matrix:")
print(cka_matrix)

# -----------------------------
# PLOT HEATMAP
# -----------------------------
plt.figure(figsize=(8, 6))

sns.heatmap(
    cka_matrix,
    annot=True,
    cmap="viridis",
    vmin=0,
    vmax=1,
    linewidths=0.5,
    fmt=".2f",
    cbar_kws={"label": "CKA similarity"}
)

plt.title("CKA Similarity: ESM vs Sequence Methods")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

# -----------------------------
# SAVE
# -----------------------------
plt.savefig(output_file, dpi=300)
plt.close()

print(f"\nHeatmap saved to: {output_file}")