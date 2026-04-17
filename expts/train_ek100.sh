#!/bin/bash
#SBATCH --job-name=clam_ek100
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu-a100-80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=24:00:00

source ~/anaconda3/etc/profile.d/conda.sh
conda activate scalant

YOUR_ROOT_PATH=/home/s4076893/Desktop
YOUR_FEATURE_DIR=epickitchens100/features/rgb_kinetics_bninception

python main.py \
  --cfg configs/ek100/default.yaml \
  --opts \
  DATA.DATA_ROOT_PATH ${YOUR_ROOT_PATH} \
  DATA.FEAT_DIR ${YOUR_FEATURE_DIR}