#!/usr/bin/env bash
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
source ~/.bashrc || true
conda activate catvton

# 토큰을 --hf_token으로 넘겨도 되고, env 변수(HUGGINGFACE_TOKEN 등)만으로도 동작합니다.
python train.py \
  --list_file DATA/VITON-HD/vitonhd_train_mask.csv \
  --output_dir ckpts/catvton_sd35_mask \
  --sd3_model stabilityai/stable-diffusion-3.5-large \
  --size_h 512 --size_w 384 \
  --mask_based \
  --lr 1e-5 --batch_size 1 --max_steps 16000 \
  --cond_dropout_p 0.1