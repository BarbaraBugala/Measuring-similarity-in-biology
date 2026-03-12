from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "hemoglobin_130-160"
RESULTS_DIR = ROOT / "results" / "hemoglobin_130-160"

# proteins for the experiment
FASTA_FILE = DATA_DIR / "hemoglobin_130-160.fasta"

# embeddings of esm2 from the proteins
EMBEDDINGS_FILE = DATA_DIR / "embeddings_esm2_hemoglobin_130-160"

# results of computations
DISTANCE_DIR = RESULTS_DIR

DISTANCE_DIR.mkdir(parents=True, exist_ok=True)

# plot directory
PLOT_DIR = ROOT / "plots" / "hemoglobin_130-160"


# kernels available
KERNEL_DIR = RESULTS_DIR / "kernels"
KERNEL_FILES = {
    "hamming": KERNEL_DIR / "sequence_hamming_kernel.csv",
    "blosum": KERNEL_DIR / "sequence_blosum_kernel.csv",
    "blast": KERNEL_DIR / "sequence_blast_kernel.csv",
    "msa_p": KERNEL_DIR / "msa_p_kernel.csv",
    "msa_pam": KERNEL_DIR / "msa_pam_kernel.csv",
    "esm2_cosine": KERNEL_DIR / "esm2_cosine_kernel.csv",
    "esm2_euclidean": KERNEL_DIR / "esm2_euclidean_kernel.csv",
}

# embedding methods
EMB_METHODS = ["esm2_euclidean", "esm2_cosine"]

# evolutionary methods
SEQ_METHODS = ["hamming", "blosum", "blast", "msa_p", "msa_pam"]