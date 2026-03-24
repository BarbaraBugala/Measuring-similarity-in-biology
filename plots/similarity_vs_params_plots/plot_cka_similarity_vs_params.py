#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import ROOT, ALL_MODELS, SEQ_METHODS, MODEL_SIZES

# -----------------------------
# PATHS
# -----------------------------
RESULTS_ROOT = ROOT / "results"
PLOTS_ROOT = ROOT / "plots"
PLOTS_ROOT.mkdir(exist_ok=True, parents=True)


# -----------------------------
# DATASETS
# -----------------------------
DATASETS = sorted([p for p in RESULTS_ROOT.iterdir() if p.is_dir()])

# -----------------------------
# CKA FILES
# -----------------------------
SIM_FILE = "cka_scores.csv"
BASELINE_FILE = "cka_scores_baseline.csv"

# -----------------------------
# COLOR MAP
# -----------------------------
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown"]

# -----------------------------
# HELPERS
# -----------------------------
def get_embedding(method_name):
    parts = method_name.split("_")
    if parts[-1] == "baseline":
        return parts[-2]
    return parts[-1]

def pivot_cka(df):
    df = df.copy()
    df["embedding"] = df["esm_method"].apply(get_embedding)
    pivot = df.pivot_table(
        index="sequence_method",
        columns="embedding",
        values="cka",
        aggfunc="mean"
    )
    return pivot

# -----------------------------
# PLOTTING
# -----------------------------
def plot_cka_grid_colored_baseline():
    # Infer embeddings from first dataset/model
    sample_df = pd.read_csv(DATASETS[0] / "similarities" / ALL_MODELS[0] / SIM_FILE)
    sample_pivot = pivot_cka(sample_df)
    emb_methods = list(sample_pivot.columns)

    fig, axes = plt.subplots(len(SEQ_METHODS), len(emb_methods), figsize=(12, 18), sharex=True)
    if len(SEQ_METHODS) == 1:
        axes = [axes]

    # Map dataset to color
    dataset_colors = {ds.name: COLORS[i % len(COLORS)] for i, ds in enumerate(DATASETS)}

    for i, bio in enumerate(SEQ_METHODS):
        for j, emb in enumerate(emb_methods):
            ax = axes[i][j]

            for dataset_path in DATASETS:
                dataset_name = dataset_path.name
                color = dataset_colors[dataset_name]

                x_models, y_vals, baseline_vals = [], [], []

                for model in ALL_MODELS:
                    sim_path = dataset_path / "similarities" / model / SIM_FILE
                    base_path = dataset_path / "similarities" / model / BASELINE_FILE
                    if not sim_path.exists():
                        continue

                    df = pd.read_csv(sim_path)
                    base_df = pd.read_csv(base_path)

                    df = pivot_cka(df)
                    base_df = pivot_cka(base_df)

                    if bio not in df.index or emb not in df.columns:
                        continue

                    x_models.append(model)
                    y_vals.append(df.loc[bio, emb])

                    if bio in base_df.index and emb in base_df.columns:
                        baseline_vals.append(base_df.loc[bio, emb])

                if x_models:
                    # Keep consistent order
                    order = [m for m in ALL_MODELS if m in x_models]
                    y_ordered = [y_vals[x_models.index(m)] for m in order]

                    ax.plot(range(len(order)), y_ordered, marker="o", color=color, label=dataset_name)
                    ax.set_xticks(range(len(order)))
                    ax.set_xticklabels([f"{MODEL_SIZES[m]}M" for m in order])

                if baseline_vals:
                    baseline = sum(baseline_vals) / len(baseline_vals)
                    ax.axhline(baseline, linestyle="dotted", color=color)

            ax.set_title(f"{bio} vs {emb}")
            if i == len(SEQ_METHODS) - 1:
                ax.set_xlabel("Model Size")
            if j == 0:
                ax.set_ylabel("Similarity Score")

    # Legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               title="Solid lines: dataset, dotted: baseline")

    plt.suptitle("CKA Similarity Across Datasets", y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    plt.savefig(PLOTS_ROOT / "similarity_vs_params_plots" / "cka_grid.png")
    plt.close()

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    plot_cka_grid_colored_baseline()