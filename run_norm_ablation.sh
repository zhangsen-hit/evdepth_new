#!/usr/bin/env bash
# Run 7 GN-ablation variants sequentially. After each run, snapshot
# local_loss/*.txt and the per-step fwd timing into local_loss_runs/<tag>_3k/.
set -euo pipefail

VARIANTS=(
  v1_m_only
  v2_gates_only
  v3_no_stage0
  v4_only_stage0
  v5_no_adapter
  v6_ln
  v7_bn
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

  # Snapshot metrics & timing
  for f in train_loss train_lr train_delta1 val_loss val_delta1 val_rmse; do
    [ -f "local_loss/$f.txt" ] && cp "local_loss/$f.txt" "$out_dir/"
  done
  # Capture per-step E2DepthUNet timing (last 3 reports)
  grep -A1 "Timing" "$log" \
    | tr '\r' '\n' \
    | grep -E "E2DepthUNet|Depth Loss" \
    | tail -6 > "$out_dir/timing.txt" || true
done

echo ""
echo "All $((${#VARIANTS[@]})) variants finished. Snapshots in local_loss_runs/."
