#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Importing your project-specific variables
from config import PLOT_DIR, RESULTS_DIR, ALL_MODELS

def get_cka_data(model_type, is_baseline=False):
    suffix = "_baseline" if is_baseline else ""
    file_path = RESULTS_DIR / "similarities" / model_type / f"cka_scores{suffix}.csv"
    if not file_path.exists():
        return None
    df = pd.read_csv(file_path)
    return df.pivot_table(index="esm_method", columns="sequence_method", values="cka")

# -----------------------------
# SETUP PLOT GRID
# -----------------------------
num_models = len(ALL_MODELS)

# Increase the second value in figsize to make the overall canvas taller
# (Width, Height)
fig, axes = plt.subplots(num_models, 2, figsize=(15, 7 * num_models), squeeze=False)

for i, model_type in enumerate(ALL_MODELS):
    for j, is_baseline in enumerate([False, True]):
        ax = axes[i, j]
        matrix = get_cka_data(model_type, is_baseline)
        
        if matrix is not None:
            # We remove square=True so it can stretch.
            # 'cbar_kws' size is now handled by the subplot geometry.
            sns.heatmap(
                matrix, 
                annot=True, 
                cmap="viridis", 
                vmin=0, 
                vmax=1,
                fmt=".2f", 
                ax=ax, 
                linewidths=1,
                cbar_kws={"fraction": 0.046, "pad": 0.04} # Standard sizing for colorbar height
            )
            
            # This line manually stretches the y-axis cells to be taller
            ax.set_aspect('auto') 
            
            status = "Baseline" if is_baseline else "Normal"
            ax.set_title(f"{model_type}: {status}", fontweight='bold', fontsize=14, pad=20)
            
            # Ensure labels are visible and large enough
            ax.tick_params(labelbottom=True, labelleft=True, labelsize=10)
            ax.set_xlabel("Sequence Method", fontsize=12, labelpad=10)
            ax.set_ylabel("ESM Method", fontsize=12, labelpad=10)
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        else:
            ax.axis('off')

# -----------------------------
# SAVE
# -----------------------------
# hspace adds vertical room between different model rows
plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=5.0)

output_path = PLOT_DIR / "cka_heatmap.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nDone!")