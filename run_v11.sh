#!/usr/bin/env bash
# Single run: v11 = v5 + v8 + v10 stacked.
set -euo pipefail
cd "$(dirname "$0")"

tag="v11_in_m_only_no_s0_no_adapter"
cfg="config_liosam_stlstm_${tag}.yaml"
log="/tmp/stlstm_${tag}.log"
out_dir="local_loss_runs/${tag}_3k"
mkdir -p "$out_dir"

echo "[$(date +%H:%M:%S)] running $tag"

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
echo "Snapshot in $out_dir"
