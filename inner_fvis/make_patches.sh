#!/bin/bash
#SBATCH -p gpu17,gpu22,gpu24
#SBATCH --gres=gpu:a100:1
#SBATCH -o logs/inner_fvis_%j.out
#SBATCH -e logs/inner_fvis_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=64G


export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

ARCH=resnet50
TARGET_LAYER=layer4
PATCH_SIZE=64
TOPK=50

FILES_TXT=/BS/feature_viz/work/code/VITAL_fork/inner_fvis/resnet50/neuron_layer4/files_all.txt
SAVE_DIR=/BS/feature_viz/work/code/VITAL_fork/inner_fvis/resnet50/neuron_layer4

mkdir -p logs
mkdir -p "${SAVE_DIR}"

python text2patch_parallel.py \
  --arch "${ARCH}" \
  --target_layer "${TARGET_LAYER}" \
  --patch_size "${PATCH_SIZE}" \
  --topk_patches "${TOPK}" \
  --files_txt "${FILES_TXT}" \
  --save_dir "${SAVE_DIR}" \
  --batch_size_images 8 \
  --max_patch_batch 4096 \
  --num_workers 8 \
  --use_amp