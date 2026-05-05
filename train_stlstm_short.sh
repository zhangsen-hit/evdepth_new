#!/usr/bin/env bash
# Short-training validation for ST-LSTM + zigzag U-Net.
# 3000 step / seq=6 / val every 500 step. Goal: confirm loss decreases & no NaN.
set -euo pipefail

WANDB_MODE=offline python train.py \
  --config config_liosam_stlstm_short.yaml \
  --gpus 1,2,3,4,5,6 \
  --batch_size 2 \
  --lr 0.00005
