from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# proteins for the experiment
FASTA_FILE = DATA_DIR / "insulin_proteins_50-200.fasta"

# embeddings of esm2 from the proteins
EMBEDDINGS_FILE = DATA_DIR / "embeddings_esm2_insulin_proteins_50-200"

# results of computations
DISTANCE_DIR = RESULTS_DIR / "insulin_proteins_50-200"

DISTANCE_DIR.mkdir(parents=True, exist_ok=True)


# for CKA comparison use kernel
DISTANCES_CKA_1 = DISTANCE_DIR / "sequence_blosum_kernel.csv"
DISTANCES_CKA_2 = DISTANCE_DIR / "esm2_cosine_kernel.csv"
COMPARISON_OF_CKA = "Blosum and Cosine"


# for QSI comparison use distance
DISTANCES_QSI_1 = DISTANCE_DIR / "msa_p_distance.csv"
DISTANCES_QSI_2 = DISTANCE_DIR / "esm2_cosine_distance.csv"
COMPARISON_OF_QSI = "MSA and cosine"