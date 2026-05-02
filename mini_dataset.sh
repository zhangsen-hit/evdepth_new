WANDB_MODE=offline python train.py \
  --config config_liosam.yaml \
  --gpus 1,2,3,4,5,6 \
  --batch_size 2 \
  --lr 0.00005 \
  # --max_epochs 200
