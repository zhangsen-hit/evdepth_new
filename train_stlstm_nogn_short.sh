#!/usr/bin/env bash
# ST-LSTM + zigzag, GN OFF — speed ablation (otherwise identical to stlstm short).
set -euo pipefail

WANDB_MODE=offline python train.py \
  --config config_liosam_stlstm_nogn_short.yaml \
  --gpus 1,2,3,4,5,6 \
  --batch_size 2 \
  --lr 0.00005
