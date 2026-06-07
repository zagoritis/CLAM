#!/bin/bash
#SBATCH --job-name=clam_ek100_evaltrans
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu-a100-80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=04:00:00

# Step A probe: fuse the action-transition prior P(next|prev) into a TRAINED
# checkpoint at eval time and sweep the fusion weight. No training. Watch
# val/topk_set_recall@5 (the single-head ceiling, ~22.4 un-fused): does it clear
# the ceiling? W=0 reproduces the un-fused 8(5) baseline as a sanity check.

module load Miniconda3/24.7.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate clam

cd /home/s4076893/Desktop/CLAM || exit 1

YOUR_ROOT_PATH=/home/s4076893/Desktop
YOUR_FEATURE_DIR=epickitchens100/features/rgb_kinetics_bninception
PYTHON=/home/s4076893/.conda/envs/clam/bin/python

# >>> EDIT ME: the committed 8(5) run directory under checkpoints/ <<<
CKPT=MAMBA-4-4-1024_bs128_lr0.0005_wd0.0_20260606-21:13:01
# gt_prev = last observed GT action (upper bound); past_head = end-to-end soft mixture
FUSE_SOURCE=gt_prev

for W in 0 0.5 1 2 4; do
  echo "================ TRANSITION_WEIGHT=${W}  FUSE_SOURCE=${FUSE_SOURCE} ================"
  $PYTHON main.py \
    --cfg configs/ek100/transition_eval.yaml \
    --opts \
    DATA.DATA_ROOT_PATH ${YOUR_ROOT_PATH} \
    DATA.FEAT_DIR ${YOUR_FEATURE_DIR} \
    TEST.CKPT_PATH ${CKPT} \
    MODEL.DIVERSE_SET.TRANSITION_WEIGHT ${W} \
    MODEL.DIVERSE_SET.TRANSITION_FUSE_SOURCE ${FUSE_SOURCE}
done
