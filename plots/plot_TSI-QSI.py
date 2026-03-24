#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Importing your project-specific variables
from config import PLOT_DIR, RESULTS_DIR, ALL_MODELS

def load_qsi_tsi_data(model_type, is_baseline=False):
    """Helper to load QSI/TSI data."""
    suffix = "_baseline" if is_baseline else ""
    file_path = RESULTS_DIR / "similarities" / model_type / f"qsi_tsi_scores{suffix}.csv"
    
    if not file_path.exists():
        return None
    return pd.read_csv(file_path)

def create_comparison_plot(metric_name, output_filename):
    """Generates a grid for a specific metric (QSI or TSI)."""
    num_models = len(ALL_MODELS)
    
    # Height adjusted for taller plots (7 per model)
    fig, axes = plt.subplots(num_models, 2, figsize=(14, 7 * num_models), squeeze=False)

    for i, model_type in enumerate(ALL_MODELS):
        df_norm = load_qsi_tsi_data(model_type, is_baseline=False)
        df_base = load_qsi_tsi_data(model_type, is_baseline=True)

        for j, (df, is_baseline) in enumerate([(df_norm, False), (df_base, True)]):
            ax = axes[i, j]
            
            if df is not None:
                # Note: Your CSV uses 'embedding_method' as the index for these files
                matrix = df.pivot_table(
                    index="embedding_method", 
                    columns="sequence_method", 
                    values=metric_name.lower()
                )
                
                sns.heatmap(
                    matrix, 
                    annot=True, 
                    cmap="viridis", 
                    fmt=".2f", 
                    ax=ax, 
                    cbar_kws={"shrink": 0.8, "label": f"{metric_name} Score"}
                )
                
                status = "Baseline" if is_baseline else "Normal"
                ax.set_title(f"{model_type}: {metric_name} ({status})", fontweight='bold', pad=15)
                
                # Force descriptions on every plot
                ax.tick_params(labelbottom=True, labelleft=True)
                ax.set_xlabel("Sequence Method")
                ax.set_ylabel("Embedding Method")
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            else:
                ax.text(0.5, 0.5, "Data Missing", ha='center', va='center')

    plt.tight_layout(pad=4.0)
    
    save_path = PLOT_DIR / output_filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {metric_name} comparison to: {save_path}")

# -----------------------------
# EXECUTION
# -----------------------------
# Ensure directory exists
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Generate the two separate comparison grids
create_comparison_plot("QSI", "qsi_heatmap.png")
create_comparison_plot("TSI", "tsi_heatmap.png")