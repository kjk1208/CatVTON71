#!/usr/bin/env bash
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
source ~/.bashrc || true
conda activate catvton

torchrun --standalone --nnodes=1 --nproc_per_node=2 ddp_train_without_crossattn.py \
  --config configs/catvton_sd35.yaml \
  --mixed_precision bf16 \
  --save_name input_noise \
  --cond_dropout_p 0.1 \
  --no_prefer_xformers \
  --strip_cross_attention \
  --preview_infer_steps 16 \
  --preview_strength 0.6 \
  --hole_loss_weight 2.0 \
  --norm_match_clip 5.0

