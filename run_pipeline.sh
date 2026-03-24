#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo " Starting Biology Similarity Pipeline..."

# 1. Generate ESM2 Embeddings
echo "Step 1: Generating ESM2 Embeddings..."

python -m models.esm2_embeddings # only once

echo "Step 2, compute embedding distances"
python -m experiments.compute_esm_embedding_distances

# echo "Step 3: compute sequence distances"
# python -m experiments.compute_sequence_blast
# python -m experiments.compute_sequence_blosum
# python -m experiments.compute_sequence_hamming

# echo "Step 4: compute MSA and MSA based distances"
# python -m experiments.compute_msa_mafft
# python -m experiments.compute_msa_p_distance
# python -m experiments.compute_msa_pam


echo "Step 5: compute similarities"
python -m experiments.compare_representations_CKA
python -m experiments.compare_representations_mKNN
python -m experiments.compare_representations_TSI-QSI


echo "Step 6: compute baseline"
python -m experiments.compute_baseline.compute_permutations # only once
python -m experiments.compute_baseline.compare_representations_CKA_baseline
python -m experiments.compute_baseline.compare_representations_mKNN_baseline
python -m experiments.compute_baseline.compare_representations_TSI-QSI_baseline