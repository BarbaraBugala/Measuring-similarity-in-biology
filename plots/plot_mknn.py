#!/usr/bin/env python3
"""
plot_mknn_heatmaps_stacked.py
Fully corrected version: each subplot shows correct y-axis labels per model/k.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import PLOT_DIR, RESULTS_DIR, ALL_MODELS

# -----------------------------
# SETUP
# -----------------------------
plots_dir = PLOT_DIR
plots_dir.mkdir(parents=True, exist_ok=True)
output_file = plots_dir / "mknn_heatmap.png"

print("Loading data...")

# -----------------------------
# GET k VALUES FROM FIRST MODEL
# -----------------------------
sample_model = ALL_MODELS[0]
sample_file = RESULTS_DIR / "similarities" / sample_model / "mknn_scores.csv"
df_sample = pd.read_csv(sample_file)

k_values = sorted(df_sample["k"].unique())
n_k = len(k_values)

n_models = len(ALL_MODELS)
n_rows = n_models * 2  # model + baseline

# -----------------------------
# CREATE FIGURE
# -----------------------------
fig, axes = plt.subplots(
    n_rows,
    n_k,
    figsize=(5 * n_k + 1, 4 * n_rows),
    sharex=True,
    sharey=False,  # Do not force y-axis sharing
    gridspec_kw={'width_ratios': [1] * n_k}
)

# Handle edge case if only one row
if n_rows == 1:
    axes = [axes]

# Global colorbar axis
cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])

# -----------------------------
# PLOTTING FUNCTION
# -----------------------------
def plot_row(df, row_axes, model_name, is_baseline, show_cbar=False):
    df = df.copy()
    middle_idx = len(row_axes) // 2

    for i, (ax, k) in enumerate(zip(row_axes, k_values)):
        # Filter data for this k
        df_k = df[df["k"] == k]

        if df_k.empty:
            ax.set_visible(False)
            continue

        # Determine ESM and sequence order for this subplot
        esm_order_k = sorted(df_k["esm_method"].unique())
        seq_order_k = sorted(df_k["sequence_method"].unique())

        # Pivot table
        matrix = df_k.pivot_table(
            index="esm_method",
            columns="sequence_method",
            values="mknn_overlap"
        )

        # Reindex per-subplot
        matrix = matrix.reindex(index=esm_order_k, columns=seq_order_k)

        # Remove _baseline in y-axis only
        matrix.index = [name.replace("_baseline", "") for name in matrix.index]

        # Show y-axis ticks only for first column
        show_yticks = (i == 0)

        sns.heatmap(
            matrix,
            annot=True,
            cmap="viridis",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            fmt=".2f",
            ax=ax,
            cbar=show_cbar and (i == 0),
            cbar_ax=cbar_ax if (show_cbar and i == 0) else None,
            cbar_kws={"label": "mKNN overlap"} if (show_cbar and i == 0) else None,
            yticklabels=show_yticks
        )

        # Y-axis label only for first column
        if show_yticks:
            ax.set_ylabel("ESM method")
        else:
            ax.set_ylabel("")
            ax.set_yticks([])  # completely hide y-axis ticks

        # -----------------------------
        # TITLE
        # -----------------------------
        title = f"k = {k}"
        if i == middle_idx:
            title = f"{model_name}" + (" (Baseline)" if is_baseline else "") + f"\n k = {k}"
        ax.set_title(title, fontsize=10, fontweight='bold')

        # X label
        ax.set_xlabel("Sequence method")

        # Y label only on first column
        if i == 0:
            ax.set_ylabel("ESM method")
        else:
            ax.set_ylabel("")

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right")  # keep y-ticks readable

# -----------------------------
# LOOP OVER MODELS
# -----------------------------
for m_idx, model in enumerate(ALL_MODELS):
    print(f"Processing {model}...")

    model_file = RESULTS_DIR / "similarities" / model / "mknn_scores.csv"
    baseline_file = RESULTS_DIR / "similarities" / model / "mknn_scores_baseline.csv"

    df_model = pd.read_csv(model_file)
    df_baseline = pd.read_csv(baseline_file)

    row_model = m_idx * 2
    row_baseline = m_idx * 2 + 1

    # Model row
    plot_row(
        df_model,
        axes[row_model],
        model_name=model,
        is_baseline=False,
        show_cbar=(m_idx == 0)
    )

    # Baseline row
    plot_row(
        df_baseline,
        axes[row_baseline],
        model_name=model,
        is_baseline=True
    )

# -----------------------------
# FINAL TOUCHES
# -----------------------------
plt.suptitle(
    "Mutual kNN Overlap: Models vs Baselines",
    fontsize=18,
    y=0.995
)

plt.tight_layout(rect=[0, 0, 0.9, 0.97])

# -----------------------------
# SAVE
# -----------------------------
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nStacked heatmaps saved to: {output_file}")