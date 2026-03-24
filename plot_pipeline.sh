#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo " Starting Biology Similarity PLot Pipeline..."

# 1. Generate ESM2 Embeddings
echo "Step 1: Generating ESM2 Embeddings..."

python -m plots.plot_cka
python -m plots.plot_mknn
python -m plots.plot_TSI-QSI