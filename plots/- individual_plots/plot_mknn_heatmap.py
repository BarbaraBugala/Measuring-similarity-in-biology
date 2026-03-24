#!/usr/bin/env python3
"""
plot_mknn_heatmaps.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import PLOT_DIR, RESULTS_DIR, MODEL_TYPE

# -----------------------------
# PATHS
# -----------------------------
mknn_file = RESULTS_DIR / "similarities" / MODEL_TYPE / "mknn_scores.csv"
plots_dir = PLOT_DIR
plots_dir.mkdir(parents=True, exist_ok=True)
output_file = plots_dir / MODEL_TYPE / "mknn_heatmaps.png"

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading mKNN data...")
df = pd.read_csv(mknn_file)
k_values = sorted(df["k"].unique())
n_k = len(k_values)

# -----------------------------
# CREATE FIGURE (Fixed Width Logic)
# -----------------------------
# We add an extra "column" with a very small width ratio for the colorbar
fig, axes = plt.subplots(
    1, 
    n_k, 
    figsize=(5 * n_k + 1, 5), # Dynamic width based on k count
    sharey=True,
    gridspec_kw={'width_ratios': [1] * n_k} 
)

# Handle single k case
if n_k == 1:
    axes = [axes]

# Create a global colorbar axes manually to prevent resizing subplots
# [left, bottom, width, height] in figure coordinates
cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6]) 

# -----------------------------
# PLOT HEATMAPS
# -----------------------------
for i, (ax, k) in enumerate(zip(axes, k_values)):
    df_k = df[df["k"] == k]
    
    matrix = df_k.pivot_table(
        index="esm_method",
        columns="sequence_method",
        values="mknn_overlap"
    )

    sns.heatmap(
        matrix,
        annot=True,
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        fmt=".2f",
        ax=ax,
        cbar=(i == 0), # We only need to generate the colorbar once
        cbar_ax=cbar_ax if i == 0 else None,
        cbar_kws={"label": "mKNN overlap"} if i == 0 else None
    )

    ax.set_title(f"k = {k}", fontweight='bold')
    ax.set_xlabel("Sequence method")
    
    # Rotate x-labels for all axes
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

axes[0].set_ylabel("ESM method")
plt.suptitle("Mutual kNN Overlap: ESM vs Sequence Methods", fontsize=16, y=0.98)

# Adjust layout while leaving room for the title and the manual colorbar on the right
plt.tight_layout(rect=[0, 0, 0.9, 0.95])

# -----------------------------
# SAVE
# -----------------------------
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nHeatmaps saved to: {output_file}")