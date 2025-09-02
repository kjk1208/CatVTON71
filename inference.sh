#!/usr/bin/env bash
set -euo pipefail

# ----- Required -----
DATA_CSV=${1:-"./DATA/VITON-HD/vitonhd_test_mask.csv"}
CKPT=${2:-"logs/20250830_151751_ddp/models/[Train]_[195]_[0.1800].ckpt"}

# ----- Optional -----
OUTDIR=${OUTDIR:-"infer_out"}
SD3_MODEL=${SD3_MODEL:-"stabilityai/stable-diffusion-3.5-large"}
SIZE_H=${SIZE_H:-512}
SIZE_W=${SIZE_W:-384}
STEPS=${STEPS:-16}
BS=${BS:-32}
SEED=${SEED:-1234}
MIXED_PRECISION=${MIXED_PRECISION:-"fp16"}   # fp16 | bf16 | fp32
INVERT_MASK=${INVERT_MASK:-1}                # 1 to invert
SAVE_PANEL=${SAVE_PANEL:-0}                  # 1 to save panels
NO_SAVE_CONCAT=${NO_SAVE_CONCAT:-0}          # 1 to skip saving concat
NO_SAVE_LEFT=${NO_SAVE_LEFT:-0}              # 1 to skip left-half
HF_TOKEN=${HF_TOKEN:-""}                     # if needed for gated repo
DEVICE=${DEVICE:-"cpu"}                     # auto | cuda | cpu

# If CPU is requested, force fp32 for compatibility
if [[ "$DEVICE" == "cpu" ]]; then
  MIXED_PRECISION="fp32"
fi

EXTRA_ARGS=()
[[ "$INVERT_MASK" == "1" ]] && EXTRA_ARGS+=("--invert_mask")
[[ "$SAVE_PANEL" == "1" ]] && EXTRA_ARGS+=("--save_panel")
[[ "$NO_SAVE_CONCAT" == "1" ]] && EXTRA_ARGS+=("--no_save_concat")
[[ "$NO_SAVE_LEFT" == "1" ]] && EXTRA_ARGS+=("--no_save_left")
[[ -n "$HF_TOKEN" ]] && EXTRA_ARGS+=("--hf_token" "$HF_TOKEN")
EXTRA_ARGS+=("--device" "$DEVICE")

python inference.py \
  --data_csv "$DATA_CSV" \
  --ckpt "$CKPT" \
  --sd3_model "$SD3_MODEL" \
  --outdir "$OUTDIR" \
  --size_h "$SIZE_H" \
  --size_w "$SIZE_W" \
  --steps "$STEPS" \
  --batch_size "$BS" \
  --seed "$SEED" \
  --mixed_precision "$MIXED_PRECISION" \
  "${EXTRA_ARGS[@]}"
