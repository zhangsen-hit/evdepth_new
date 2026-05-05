#!/usr/bin/env bash
# Phase 2: 3 additional variants — v8 (M-only + no_s0 stacked), v9 (RMSNorm), v10 (InstanceNorm).
# Identical training settings to phase 1 so the comparison is clean.
set -euo pipefail

VARIANTS=(
  v8_m_only_no_s0
  v9_rms
  v10_in
)

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

for tag in "${VARIANTS[@]}"; do
  cfg="config_liosam_stlstm_${tag}.yaml"
  log="/tmp/stlstm_${tag}.log"
  out_dir="local_loss_runs/${tag}_3k"
  mkdir -p "$out_dir"

  echo ""
  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] running $tag (config=$cfg)"
  echo "=============================================================="

  WANDB_MODE=offline python train.py \
    --config "$cfg" \
    --gpus 1,2,3,4,5,6 \
    --batch_size 2 \
    --lr 0.00005 \
    > "$log" 2>&1

  if grep -q "训练完成" "$log"; then
    echo "[$(date +%H:%M:%S)] $tag SUCCESS"
  else
    echo "[$(date +%H:%M:%S)] $tag FAILED — see $log"
    grep -A3 -i "Traceback\|Error" "$log" | head -20 || true
  fi

  for f in train_loss train_lr train_delta1 val_loss val_delta1 val_rmse; do
    [ -f "local_loss/$f.txt" ] && cp "local_loss/$f.txt" "$out_dir/"
  done
  grep -A1 "Timing" "$log" \
    | tr '\r' '\n' \
    | grep -E "E2DepthUNet|Depth Loss" \
    | tail -6 > "$out_dir/timing.txt" || true
done

echo ""
echo "Phase 2 done — snapshots in local_loss_runs/{v8,v9,v10}_3k/."
