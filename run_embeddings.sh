#!/bin/bash

#SBATCH --job-name=esm-embeddings
#SBATCH --account=hai_1136
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=/p/home/jusers/bugala1/juwels/logs/%x.%j.out
#SBATCH --error=/p/home/jusers/bugala1/juwels/logs/%x.%j.err

source $HOME/.bashrc

cd /p/home/jusers/bugala1/juwels/Measuring-similarity-in-biology

srun python3 models/esm2_embeddings.py