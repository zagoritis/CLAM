#!/bin/bash
#SBATCH --job-name=clam_ek100_trans
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu-a100-80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=24:00:00

# Step B: train the multi-query model with transition-grounded coverage targets
# (COVERAGE_SOURCE=transition). Identical to 8(5) except the non-anchor slots are
# matched to external transition successors instead of slot-0 self-distilled modes.
# Watch val/diverse_set_recall@5: does it exceed the single-head topk_set_recall@5
# (~22.4)? Also watch that mt5r / topk_set_recall stay healthy (slot 0 intact).

module load Miniconda3/24.7.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate clam

cd /home/s4076893/Desktop/CLAM || exit 1

YOUR_ROOT_PATH=/home/s4076893/Desktop
YOUR_FEATURE_DIR=epickitchens100/features/rgb_kinetics_bninception

PYTHON=/home/s4076893/.conda/envs/clam/bin/python

$PYTHON main.py \
  --cfg configs/ek100/transition.yaml \
  --opts \
  DATA.DATA_ROOT_PATH ${YOUR_ROOT_PATH} \
  DATA.FEAT_DIR ${YOUR_FEATURE_DIR}
