#!/usr/bin/env python3
"""
plot_all_distance_heatmaps.py

Plot all distance matrices from results/distances in one grid figure.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# -----------------------------
# PATHS
# -----------------------------
from config import RESULTS_DIR, PLOT_DIR

distance_dir = RESULTS_DIR / "kernels"

plots_dir = PLOT_DIR
plots_dir.mkdir(parents=True, exist_ok=True)

output_file = plots_dir / "all_kernel_heatmaps.png"

# -----------------------------
# LOAD FILES
# -----------------------------
distance_files = sorted(distance_dir.glob("*.csv"))

if len(distance_files) == 0:
    raise ValueError("No kernel matrices found.")

print(f"Found {len(distance_files)} kernel matrices")

# -----------------------------
# GRID SIZE
# -----------------------------
n = len(distance_files)
cols = min(3, n)   # max 3 per row
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

# If only one row/col, normalize axes
if n == 1:
    axes = [[axes]]
elif rows == 1:
    axes = [axes]

# -----------------------------
# PLOT EACH MATRIX
# -----------------------------
for i, file in enumerate(distance_files):

    r = i // cols
    c = i % cols

    ax = axes[r][c]

    print(f"Loading {file.name}")

    df = pd.read_csv(file, index_col=0)

    sns.heatmap(
        df,
        ax=ax,
        cmap="magma",
        cbar=False,
        xticklabels=False,
        yticklabels=False
    )

    ax.set_title(file.stem)

# -----------------------------
# REMOVE EMPTY PANELS
# -----------------------------
for j in range(i + 1, rows * cols):
    r = j // cols
    c = j % cols
    fig.delaxes(axes[r][c])

# -----------------------------
# FINALIZE
# -----------------------------
plt.suptitle("Kernel Matrices Comparison", fontsize=16)
plt.tight_layout()

# leave space for title
plt.subplots_adjust(top=0.93)

# -----------------------------
# SAVE
# -----------------------------
plt.savefig(output_file, dpi=300)
plt.close()

print(f"\nSaved grid heatmap to {output_file}")