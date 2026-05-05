#!/usr/bin/env bash
# Baseline (stand_convlstm) short-training run for A/B against ST-LSTM+zigzag.
# Identical settings: 3000 step / seq=6 / val every 500 step / lr=5e-5 / bs=2 / 6 GPUs.
set -euo pipefail

WANDB_MODE=offline python train.py \
  --config config_liosam_baseline_short.yaml \
  --gpus 1,2,3,4,5,6 \
  --batch_size 2 \
  --lr 0.00005
