from pathlib import Path

# --- Core Paths ---
ROOT = Path(__file__).resolve().parent
PROJECT_NAME = "insuline_50-200"

DATA_DIR = ROOT / "data" / PROJECT_NAME
RESULTS_DIR = ROOT / "results" / PROJECT_NAME
PLOT_DIR = ROOT / "plots" / PROJECT_NAME

# --- Model Configuration ---
MODEL_TYPE = "esm2_150M"
ALL_MODELS = ["esm2_8M", "esm2_35M", "esm2_150M", "esm2_650M", "esm2_3B", "esm2_15B"]

# ---Model Sizes ---
MODEL_SIZES = {
    "esm2_8M": 8,
    "esm2_35M": 35,
    "esm2_150M": 150,
    "esm2_650M": 650,
    "esm2_3B" : 3000,
    "esm2_15B" : 15000
}

# --- Input Files ---
FASTA_FILE = DATA_DIR / f"{PROJECT_NAME}.fasta"
EMBEDDINGS_FILE = DATA_DIR / f"embeddings_esm2_{PROJECT_NAME}"

# --- Method Definitions ---
SEQ_METHODS = ["hamming", "blosum", "blast", "msa_p", "msa_pam"]
EMB_VARIANTS = ["cosine", "euclidean"]
EMB_METHODS = [f"{MODEL_TYPE}_{v}" for v in EMB_VARIANTS]

# --- Path Helpers ---
KERNEL_DIR = RESULTS_DIR / "kernels"
DISTANCE_DIR = RESULTS_DIR / "distances"

def generate_file_map(base_dir, suffix, is_baseline=False):
    """Helper to build file maps for kernels and distances."""
    mapping = {m: base_dir / f"sequence_{m}_{suffix}.csv" for m in ["hamming", "blosum", "blast"]}
    mapping.update({m: base_dir / f"{m}_{suffix}.csv" for m in ["msa_p", "msa_pam"]})
    
    for v in EMB_VARIANTS:
        key = f"{MODEL_TYPE}_{v}"
        if is_baseline:
            mapping[f"{key}_baseline"] = base_dir / f"{key}_{suffix}_PERM.csv"
        else:
            mapping[key] = base_dir / f"{key}_{suffix}.csv"
    return mapping

# --- Final Maps ---
KERNEL_FILES = generate_file_map(KERNEL_DIR, "kernel")
DISTANCE_FILES = generate_file_map(DISTANCE_DIR, "distance")

BASELINE_KERNEL_FILES = generate_file_map(KERNEL_DIR, "kernel", is_baseline=True)
BASELINE_DISTANCE_FILES = generate_file_map(DISTANCE_DIR, "distance", is_baseline=True)

# Baseline method lists
BASELINE_EMB_METHODS = [f"{m}_baseline" for m in EMB_METHODS]
BASELINE_SEQ_METHODS = SEQ_METHODS.copy()