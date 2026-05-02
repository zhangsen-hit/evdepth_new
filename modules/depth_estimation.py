"""
Depth Estimation Module for PyTorch Lightning
Based on detection.py but adapted for depth estimation
"""
from typing import Any, Optional, Tuple, Union, Dict, List
from warnings import warn
import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch as th
import torch.nn.functional as F
import torch.distributed as dist
from pytorch_lightning.utilities.types import STEP_OUTPUT

from data.utils.types import DataType, LstmStates, DatasetSamplingMode
from utils.evaluation.depth import DepthEvaluator
from utils.padding import InputPadderFromShape
from modules.data.merge_mixed_batches import merge_mixed_batches
from modules.data.rnn_states_across_batches import RNNStates
from models.depth_head.depth_estimator import DepthEstimator


def _finest_depth_pred(predictions: Dict[str, th.Tensor]) -> th.Tensor:
    """取最高分辨率预测：优先 depth_1，否则 depth_2；不可对 .get 传入 predictions[\"depth_2\"] 作默认值（会先求值）。"""
    if predictions is None:
        raise TypeError("predictions must not be None")
    if "depth_1" in predictions:
        return predictions["depth_1"]
    if "depth_2" in predictions:
        return predictions["depth_2"]
    finest_k = None
    finest_scale: Optional[int] = None
    for k in predictions:
        if isinstance(k, str) and k.startswith("depth_"):
            suf = k[len("depth_") :]
            if suf.isdigit():
                s = int(suf)
                if finest_scale is None or s < finest_scale:
                    finest_scale = s
                    finest_k = k
    if finest_k is None:
        raise KeyError(
            "predictions 需含 depth_1、depth_2 或任一 depth_<scale>；"
            f"当前 keys={tuple(predictions.keys())}"
        )
    return predictions[finest_k]


class DepthOutput:
    """Output keys for depth estimation"""
    DEPTH_PRED = 'depth_pred'
    DEPTH_GT = 'depth_gt'
    EV_REPR = 'ev_repr'
    DEPTH_VIZ_MASK = 'depth_viz_mask'  # (T or B, 1, H, W) bool, for viz: invalid -> black
    SKIP_VIZ = 'skip_viz'


class Module(pl.LightningModule):
    """Depth Estimation Module"""
    def __init__(self, full_config: Dict[str, Any]):
        super().__init__()
        
        self.full_config = full_config
        
        self.mdl_config = full_config['model']

        # Depth representation range (meters)
        depth_config = self.mdl_config.get('depth_range', {})
        self.depth_min = float(depth_config.get('min', 0.5) if isinstance(depth_config, dict) else 0.5)
        self.depth_max = float(depth_config.get('max', 80.0) if isinstance(depth_config, dict) else 80.0)
        self._log_depth_min = math.log(self.depth_min)
        self._log_depth_max = math.log(self.depth_max)
        self._log_depth_denom = self._log_depth_max - self._log_depth_min
        in_res_hw = tuple(self.mdl_config['backbone']['in_res_hw'])
        self.input_padder = InputPadderFromShape(desired_hw=in_res_hw)
        
        # Build depth estimation model（只依賴 models.backbone / models.fpn / models.depth_head）
        self.mdl = DepthEstimator(self.mdl_config)

        # 针对 train/val/test 三个阶段，分别维护一份 RNN 隐状态管理器（扁平命名，避免嵌套映射）
        self.train_rnn_states = RNNStates()
        self.val_rnn_states = RNNStates()
        self.test_rnn_states = RNNStates()

    def log_depth_to_norm_log_depth(self, log_depth: th.Tensor) -> th.Tensor:
        """
        log(depth) -> norm_log, where norm_log in [0, 1] corresponds to depth in [depth_min, depth_max].
        """
        log_depth = log_depth.to(dtype=th.float32)
        log_depth_clamped = torch.clamp(log_depth, min=self._log_depth_min, max=self._log_depth_max)
        return (log_depth_clamped - self._log_depth_min) / max(self._log_depth_denom, 1e-6)
    
    def setup(self, stage: Optional[str] = None) -> None:
        dataset_name = self.full_config['dataset']['name']
        # 各阶段独立的辅助信息（扁平命名）
        self.train_hw: Optional[Tuple[int, int]] = None
        self.val_hw: Optional[Tuple[int, int]] = None
        self.test_hw: Optional[Tuple[int, int]] = None

        self.train_batch_size: Optional[int] = None
        self.val_batch_size: Optional[int] = None
        self.test_batch_size: Optional[int] = None

        self.train_depth_evaluator: Optional[DepthEvaluator] = None
        self.val_depth_evaluator: Optional[DepthEvaluator] = None
        self.test_depth_evaluator: Optional[DepthEvaluator] = None

        self.train_sampling_mode: Optional[DatasetSamplingMode] = None
        self.val_sampling_mode: Optional[DatasetSamplingMode] = None
        self.test_sampling_mode: Optional[DatasetSamplingMode] = None
        
        self.started_training = True
        
        dataset_train_sampling = self.full_config['dataset']['train']['sampling']
        dataset_eval_sampling = self.full_config['dataset']['eval']['sampling']
        assert dataset_train_sampling in iter(DatasetSamplingMode)
        assert dataset_eval_sampling in (DatasetSamplingMode.STREAM, DatasetSamplingMode.RANDOM)
        
        # Get depth range from config
        depth_config = self.full_config['model'].get('depth_range', {})
        min_depth = depth_config.get('min', 0.1) if isinstance(depth_config, dict) else 0.1
        max_depth = depth_config.get('max', 100.0) if isinstance(depth_config, dict) else 100.0
        
        if stage == 'fit':  # train + val
            self.train_config = self.full_config['training']
            self.train_metrics_config = self.full_config['logging']['train']['metrics']
            
            # 与 val 相同：每 epoch 在本地算深度指标；compute 只控制是否同步到 Lightning/W&B
            self._log_train_depth_metrics = self.train_metrics_config.get('compute', False)
            self.train_depth_evaluator = DepthEvaluator(min_depth=min_depth, max_depth=max_depth)
            self.val_depth_evaluator = DepthEvaluator(min_depth=min_depth, max_depth=max_depth)

            self.train_sampling_mode = dataset_train_sampling
            self.val_sampling_mode = dataset_eval_sampling

            # train/val 的 hw 和 batch_size 在首次看到 batch 时确定
            self.train_hw = None
            self.val_hw = None
            self.train_batch_size = None
            self.val_batch_size = None
            self.started_training = False

            # epoch 级别 loss 累加器（用于 local_loss 文件写入）
            self._epoch_loss_sum = 0.0
            self._epoch_loss_count = 0
            # 验证 loss 累加器
            self._val_epoch_loss_sum = 0.0
            self._val_epoch_loss_count = 0

            # 新一轮训练开始时，清空本地的指标日志文件（仅在主进程执行）
            try:
                import os
                if getattr(getattr(self, "trainer", None), "is_global_zero", True):
                    os.makedirs("local_loss", exist_ok=True)
                    for name in ("train_loss.txt", "train_lr.txt", "train_delta1.txt",
                                 "val_loss.txt", "val_rmse.txt", "val_delta1.txt"):
                        path = os.path.join("local_loss", name)
                        if os.path.exists(path):
                            os.remove(path)
            except Exception:
                pass
        elif stage == 'validate':
            self.val_depth_evaluator = DepthEvaluator(min_depth=min_depth, max_depth=max_depth)
            self.val_sampling_mode = dataset_eval_sampling
            self.val_hw = None
            self.val_batch_size = None
        elif stage == 'test':
            self.test_depth_evaluator = DepthEvaluator(min_depth=min_depth, max_depth=max_depth)
            self.test_sampling_mode = dataset_eval_sampling
            self.test_hw = None
            self.test_batch_size = None
        else:
            raise NotImplementedError
    
    def forward(self,
                event_tensor: th.Tensor,
                previous_states: Optional[LstmStates] = None,
                retrieve_depth: bool = True,
                targets: Optional[th.Tensor] = None,
                masks: Optional[th.Tensor] = None) \
            -> Tuple[Union[Dict[str, th.Tensor], None], Union[Dict[str, th.Tensor], None], LstmStates]:
        return self.mdl(x=event_tensor,
                       previous_states=previous_states,
                       retrieve_depth=retrieve_depth,
                       targets=targets,
                       masks=masks)
    
    def get_worker_id_from_batch(self, batch: Any) -> int:
        return batch['worker_id']
    
    def get_data_from_batch(self, batch: Any):
        return batch['data']
    
    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        batch = merge_mixed_batches(batch)
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)
        
        mode = "train"
        self.started_training = True
        step = self.trainer.global_step
        ev_tensor_sequence = data[DataType.EV_REPR]
        depth_gt_sequence = data[DataType.DEPTH]  # Ground truth depth in log space
        depth_mask_sequence = data.get(DataType.DEPTH_MASK, None)  # Valid depth mask
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        token_mask_sequence = data.get(DataType.TOKEN_MASK, None)
        
        self.train_rnn_states.reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)
        
        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = ev_tensor_sequence[0].shape[0]
        if self.train_batch_size is None:
            self.train_batch_size = batch_size
        else:
            assert self.train_batch_size == batch_size
        
        prev_states = self.train_rnn_states.get_states(worker_id=worker_id)
        
        total_loss = 0.0
        losses_dict = {}
        num_valid_frames = 0
        
        # Process sequence
        viz_depth_preds = []
        viz_depth_gts = []
        viz_ev_reprs = []
        for tidx in range(sequence_len):
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            
            if token_mask_sequence is not None:
                token_masks = self.input_padder.pad_token_mask(token_mask=token_mask_sequence[tidx])
            else:
                token_masks = None
            
            if self.train_hw is None:
                self.train_hw = tuple(ev_tensors.shape[-2:])
            else:
                assert self.train_hw == ev_tensors.shape[-2:]
            
            # Get depth ground truth
            depth_gt = depth_gt_sequence[tidx].to(dtype=self.dtype)
            depth_gt = self.input_padder.pad_tensor_ev_repr(depth_gt)  # Pad to same size
            # B 方案：GT 目标从 log(depth) -> norm_log，让 loss 直接在 norm_log 上计算
            depth_gt_norm_log = self.log_depth_to_norm_log_depth(depth_gt)
            
            # Get depth mask
            if depth_mask_sequence is not None:
                depth_mask = depth_mask_sequence[tidx]
                depth_mask = self.input_padder.pad_tensor_ev_repr(depth_mask)
            else:
                depth_mask = None
            
            # Forward pass
            predictions, losses, prev_states = self.mdl(
                x=ev_tensors,
                previous_states=prev_states,
                retrieve_depth=True,
                targets=depth_gt_norm_log,
                masks=depth_mask,
                token_mask=token_masks
            )
            
            if losses is not None:
                total_loss += losses['loss']
                num_valid_frames += 1
                
                # Accumulate losses for logging
                for k, v in losses.items():
                    if k not in losses_dict:
                        losses_dict[k] = 0.0
                    losses_dict[k] += v
        
        self.train_rnn_states.save_states_and_detach(worker_id=worker_id, states=prev_states)
        
        # Average losses over sequence
        if num_valid_frames > 0:
            total_loss = total_loss / num_valid_frames
            for k in losses_dict:
                losses_dict[k] = losses_dict[k] / num_valid_frames
        
        # Use last frame for visualization and evaluation
        depth_pred_viz = _finest_depth_pred(predictions)
        depth_gt_viz_log = depth_gt_sequence[-1].to(dtype=self.dtype)
        depth_gt_viz_log = self.input_padder.pad_tensor_ev_repr(depth_gt_viz_log)
        depth_gt_viz = self.log_depth_to_norm_log_depth(depth_gt_viz_log)
        mask_viz = depth_mask_sequence[-1] if depth_mask_sequence is not None else None
        if mask_viz is not None:
            mask_viz = self.input_padder.pad_tensor_ev_repr(mask_viz)
        # Input resolution (same as ev_repr) for consistent viz size
        input_hw = tuple(ev_tensor_sequence[-1].shape[-2:])
        # Align GT and mask to prediction resolution for evaluator only (do not overwrite depth_gt_viz)
        pred_hw = depth_pred_viz.shape[-2:]
        depth_gt_for_metrics = depth_gt_viz
        if depth_gt_for_metrics.shape[-2:] != pred_hw:
            depth_gt_for_metrics = F.interpolate(depth_gt_for_metrics, size=pred_hw, mode='bilinear', align_corners=False)
        mask_for_metrics = mask_viz
        if mask_for_metrics is not None and mask_for_metrics.shape[-2:] != pred_hw:
            mask_for_metrics = F.interpolate(mask_for_metrics.float(), size=pred_hw, mode='nearest').bool()
        
        # Logging
        prefix = f'{mode}/'
        log_dict = {f'{prefix}{k}': v for k, v in losses_dict.items()}
        self.log_dict(log_dict, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)

        # 累加当前 step 的 loss，在 on_train_epoch_end 中写入 epoch 平均值
        try:
            self._epoch_loss_sum += float(total_loss.detach().cpu().item())
            self._epoch_loss_count += 1
        except Exception:
            pass
        
        # Add to depth evaluator (at pred resolution)
        if self.train_depth_evaluator is not None:
            self.train_depth_evaluator.add_predictions(depth_pred_viz, depth_gt_for_metrics, mask_for_metrics)
            
            depth_metrics_every_n_steps = self.train_metrics_config.get('depth_metrics_every_n_steps')
            if depth_metrics_every_n_steps is not None and \
                    step > 0 and step % depth_metrics_every_n_steps == 0:
                self.run_depth_evaluator(mode=mode, write_local_train_delta1=False)
        
        # Viz: same resolution as event (input_hw) so no padding/size mismatch in logs
        depth_pred_for_viz = F.interpolate(depth_pred_viz, size=input_hw, mode='bilinear', align_corners=False)
        # Viz mask: align to viz resolution (input_hw) to avoid 346 vs 348 mismatch
        if mask_viz is not None and mask_viz.shape[-2:] != input_hw:
            mask_viz_for_viz = F.interpolate(mask_viz.float(), size=input_hw, mode="nearest").bool()
        else:
            mask_viz_for_viz = mask_viz
        output = {
            DepthOutput.DEPTH_PRED: depth_pred_for_viz,
            DepthOutput.DEPTH_GT: depth_gt_viz,
            DepthOutput.EV_REPR: ev_tensor_sequence[-1],
            # 训练阶段可视化用：用于把 padding/无效区域显示为黑色
            DepthOutput.DEPTH_VIZ_MASK: mask_viz_for_viz,
            DepthOutput.SKIP_VIZ: False,
            'loss': total_loss
        }
        # ==== 单 batch 调试：只保存一张深度预测图 ====
        if getattr(self, "save_debug_depth", False) and not hasattr(self, "_saved_debug_depth"):
            import os
            import matplotlib.pyplot as plt

            os.makedirs("debug_depth", exist_ok=True)

            # 取 batch 里的第 0 张图，(H, W)，norm_log 已经是 log(depth) 在 [0, 1] 的归一化
            depth_norm_log = depth_pred_for_viz[0, 0].detach().cpu().numpy().astype(np.float32)

            out_path = os.path.join(
                "debug_depth",
                f"depth_pred_epoch{self.current_epoch:03d}_step{self.trainer.global_step:06d}.png"
            )
            # 深度可视化统一方案：jet + log 空间 + 红近蓝远 + 黑无效
            import matplotlib.cm as cm
            depth_norm_log_clipped = np.clip(depth_norm_log, 0.0, 1.0)
            # jet 默认：低值=蓝，高值=红；我们需要红近蓝远，因此反转归一化
            depth_norm_rev = 1.0 - depth_norm_log_clipped
            depth_rgb = (cm.get_cmap("jet")(depth_norm_rev)[:, :, :3] * 255).astype(np.uint8)
            # 如果有 mask，则无效区域置黑
            try:
                if mask_viz is not None:
                    # 对齐到要保存的 debug 深度图尺寸，避免 padding 后宽高不一致
                    target_hw = tuple(depth_pred_for_viz.shape[-2:])
                    mask_dbg_t = F.interpolate(mask_viz.float(), size=target_hw, mode="nearest").bool()
                    mask_dbg = mask_dbg_t[0, 0].detach().cpu().numpy().astype(bool)
                else:
                    mask_dbg = None
            except Exception:
                mask_dbg = None
            if mask_dbg is not None:
                depth_rgb[~mask_dbg] = 0
            plt.imsave(out_path, depth_rgb)

            # 标记只保存一次，避免多 batch 时狂刷图
            self._saved_debug_depth = False

        return output

    def _val_test_step_impl(self, batch: Any, mode: str) -> Optional[STEP_OUTPUT]:
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)
        
        assert mode in ("val", "test")
        ev_tensor_sequence = data[DataType.EV_REPR]
        depth_gt_sequence = data[DataType.DEPTH]
        depth_mask_sequence = data.get(DataType.DEPTH_MASK, None)
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        
        if mode == "val":
            self.val_rnn_states.reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)
        else:
            self.test_rnn_states.reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)
        
        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = ev_tensor_sequence[0].shape[0]
        if mode == "val":
            if self.val_batch_size is None:
                self.val_batch_size = batch_size
            else:
                assert self.val_batch_size == batch_size
            prev_states = self.val_rnn_states.get_states(worker_id=worker_id)
        else:
            if self.test_batch_size is None:
                self.test_batch_size = batch_size
            else:
                assert self.test_batch_size == batch_size
            prev_states = self.test_rnn_states.get_states(worker_id=worker_id)
        
        # 收集整段序列用于 epoch 可视化（仅 batch 内第 0 个样本）
        viz_depth_preds = []
        viz_depth_gts = []
        viz_ev_reprs = []
        viz_depth_masks = []  # 与 viz_depth_gts 同分辨率，True=有效，用于 GT 无效值显示为无限远
        val_loss_sum = 0.0
        val_loss_count = 0
        # Process sequence
        for tidx in range(sequence_len):
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            
            if mode == "val":
                if self.val_hw is None:
                    self.val_hw = tuple(ev_tensors.shape[-2:])
                else:
                    assert self.val_hw == ev_tensors.shape[-2:]
            else:
                if self.test_hw is None:
                    self.test_hw = tuple(ev_tensors.shape[-2:])
                else:
                    assert self.test_hw == ev_tensors.shape[-2:]
            
            # 验证阶段：同时计算 loss，用于写入 val_loss.txt
            if mode == "val":
                depth_gt_tgt = depth_gt_sequence[tidx].to(dtype=self.dtype)
                depth_gt_tgt = self.input_padder.pad_tensor_ev_repr(depth_gt_tgt)
                depth_gt_tgt = self.log_depth_to_norm_log_depth(depth_gt_tgt)
                if depth_mask_sequence is not None:
                    depth_mask_tgt = depth_mask_sequence[tidx]
                    depth_mask_tgt = self.input_padder.pad_tensor_ev_repr(depth_mask_tgt)
                else:
                    depth_mask_tgt = None
                predictions, val_losses, prev_states = self.mdl(
                    x=ev_tensors,
                    previous_states=prev_states,
                    retrieve_depth=True,
                    targets=depth_gt_tgt,
                    masks=depth_mask_tgt,
                )
                if val_losses is not None:
                    val_loss_sum += float(val_losses['loss'].detach())
                    val_loss_count += 1
            else:
                predictions, _, prev_states = self.mdl(
                    x=ev_tensors,
                    previous_states=prev_states,
                    retrieve_depth=True,
                    targets=None,
                    masks=None,
                )

            # 收集当前时间步用于可视化（仅取 batch 内第 0 个样本）
            pred_t = _finest_depth_pred(predictions)  # (B, 1, H', W')
            viz_hw = tuple(ev_tensor_sequence[tidx][0].shape[-2:])
            depth_pred_for_viz_t = F.interpolate(
                pred_t,
                size=viz_hw,
                mode='bilinear',
                align_corners=False,
            )
            viz_depth_preds.append(depth_pred_for_viz_t[0])  # (1, H, W)

            # 可视化用原始分辨率 GT，与 pred 的 viz_hw 一致，避免 callback 里形状 (260,346) vs (260,348)
            d_gt_raw = depth_gt_sequence[tidx].to(dtype=self.dtype)[0]  # (1,H,W) or (H,W) in log(depth)
            if d_gt_raw.dim() == 2:
                d_gt_raw = d_gt_raw.unsqueeze(0)
            # B 方案：viz 用 norm_log
            d_gt_norm_log = self.log_depth_to_norm_log_depth(d_gt_raw)
            viz_depth_gts.append(d_gt_norm_log)
            # 与 GT 同帧、同分辨率的有效 mask（True=有效），无 mask 时填空
            if depth_mask_sequence is not None:
                m = depth_mask_sequence[tidx][0]  # (1,H,W) or (H,W)
                if m.dim() == 3:
                    m = m[0]
                viz_depth_masks.append(m)  # (H, W)
            else:
                viz_depth_masks.append(None)

            viz_ev_reprs.append(ev_tensor_sequence[tidx][0])

        if mode == "val":
            self.val_rnn_states.save_states_and_detach(worker_id=worker_id, states=prev_states)
            # 将本 batch 序列的平均 val loss 累积到 epoch 累加器
            if val_loss_count > 0:
                self._val_epoch_loss_sum += val_loss_sum / val_loss_count
                self._val_epoch_loss_count += 1
        else:
            self.test_rnn_states.save_states_and_detach(worker_id=worker_id, states=prev_states)
        
        # Use last frame for evaluation
        depth_pred = _finest_depth_pred(predictions)
        # Pad GT and mask to input size
        depth_gt = depth_gt_sequence[-1].to(dtype=self.dtype)
        depth_gt = self.input_padder.pad_tensor_ev_repr(depth_gt)
        depth_gt = self.log_depth_to_norm_log_depth(depth_gt)  # log(depth) -> norm_log
        if depth_mask_sequence is not None:
            depth_mask = depth_mask_sequence[-1]
            depth_mask = self.input_padder.pad_tensor_ev_repr(depth_mask)
        else:
            depth_mask = None
        
        # Input resolution for viz (same as ev_repr)
        viz_hw = tuple(depth_gt.shape[-2:])
        # Align GT and mask to prediction resolution for metrics only
        pred_hw = depth_pred.shape[-2:]
        depth_gt_for_metrics = depth_gt
        if depth_gt_for_metrics.shape[-2:] != pred_hw:
            depth_gt_for_metrics = F.interpolate(depth_gt_for_metrics, size=pred_hw, mode='bilinear', align_corners=False)
        depth_mask_for_metrics = depth_mask
        if depth_mask_for_metrics is not None and depth_mask_for_metrics.shape[-2:] != pred_hw:
            depth_mask_for_metrics = F.interpolate(depth_mask_for_metrics.float(), size=pred_hw, mode='nearest').bool()
        
        # Add to evaluator (at pred resolution)
        if self.started_training:
            if mode == "val" and self.val_depth_evaluator is not None:
                self.val_depth_evaluator.add_predictions(depth_pred, depth_gt_for_metrics, depth_mask_for_metrics)
            if mode == "test" and self.test_depth_evaluator is not None:
                self.test_depth_evaluator.add_predictions(depth_pred, depth_gt_for_metrics, depth_mask_for_metrics)
        
        # Viz: same分辨率的整个序列（仅 batch 的第 0 个样本）
        if len(viz_depth_preds) > 0:
            depth_pred_for_viz_seq = th.stack(viz_depth_preds, dim=0)  # (T, 1, H, W)
            depth_gt_for_viz_seq = th.stack(viz_depth_gts, dim=0)      # (T, 1, H, W)
            ev_repr_for_viz_seq = th.stack(viz_ev_reprs, dim=0)        # (T, C, H, W)
            # mask: 每帧 (H,W)，True=有效；若某帧无 mask 则整段传 None
            if all(m is not None for m in viz_depth_masks):
                depth_viz_mask_seq = th.stack(viz_depth_masks, dim=0)  # (T, H, W)
            else:
                depth_viz_mask_seq = None
        else:
            depth_pred_for_viz_seq = F.interpolate(
                depth_pred, size=viz_hw, mode='bilinear', align_corners=False
            ).unsqueeze(0)
            depth_gt_for_viz_seq = depth_gt.unsqueeze(0)
            ev_repr_for_viz_seq = ev_tensor_sequence[-1][0].unsqueeze(0)
            depth_viz_mask_seq = None

        output = {
            DepthOutput.DEPTH_PRED: depth_pred_for_viz_seq,
            DepthOutput.DEPTH_GT: depth_gt_for_viz_seq,
            DepthOutput.EV_REPR: ev_repr_for_viz_seq,
            DepthOutput.DEPTH_VIZ_MASK: depth_viz_mask_seq,
            DepthOutput.SKIP_VIZ: False,
        }
        
        return output
    
    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode="val")
    
    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode="test")
    
    def run_depth_evaluator(self, mode: str, write_local_train_delta1: bool = True):
        if mode == "train":
            depth_evaluator = self.train_depth_evaluator
            batch_size = self.train_batch_size
            hw_tuple = self.train_hw
        elif mode == "val":
            depth_evaluator = self.val_depth_evaluator
            batch_size = self.val_batch_size
            hw_tuple = self.val_hw
        else:
            depth_evaluator = self.test_depth_evaluator
            batch_size = self.test_batch_size
            hw_tuple = self.test_hw
        if depth_evaluator is None:
            warn(f'depth_evaluator is None in {mode=}', UserWarning, stacklevel=2)
            return
        assert batch_size is not None
        assert hw_tuple is not None
        if depth_evaluator.has_data():
            metrics = depth_evaluator.evaluate_buffer()
            assert metrics is not None

            prefix = f'{mode}/'
            step = self.trainer.global_step
            log_dict = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    value = torch.tensor(v)
                elif isinstance(v, np.ndarray):
                    value = torch.from_numpy(v)
                elif isinstance(v, torch.Tensor):
                    value = v
                else:
                    raise NotImplementedError
                assert value.ndim == 0, f'tensor must be a scalar.\n{v=}\n{type(v)=}\n{value=}\n{type(value)=}'
                log_dict[f'{prefix}{k}'] = value.to(self.device)

            # 训练集深度指标仅当 logging.train.metrics.compute 为真时同步到 Lightning / WandB
            if mode != "train" or self._log_train_depth_metrics:
                self.log_dict(log_dict, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                    for k, v in log_dict.items():
                        dist.reduce(log_dict[k], dst=0, op=dist.ReduceOp.SUM)
                        if dist.get_rank() == 0:
                            log_dict[k] /= dist.get_world_size()
                if self.trainer.is_global_zero:
                    self.logger.log_metrics(metrics=log_dict, step=step)

            if self.trainer.is_global_zero:
                # 与 train_loss / val_delta1 相同：以 epoch 为行追加写入 local_loss（步间子评估不写 train_delta1）
                try:
                    import os
                    epoch = self.current_epoch
                    os.makedirs("local_loss", exist_ok=True)
                    if mode == "train" and write_local_train_delta1 and "delta1" in metrics:
                        file_mode = "w" if epoch == 0 else "a"
                        with open(os.path.join("local_loss", "train_delta1.txt"), file_mode) as f:
                            f.write(f"{epoch}\t{metrics['delta1']}\n")
                    if mode == "val":
                        if "rmse" in metrics:
                            file_mode = "w" if epoch == 0 else "a"
                            with open(os.path.join("local_loss", "val_rmse.txt"), file_mode) as f:
                                f.write(f"{epoch}\t{metrics['rmse']}\n")
                        if "delta1" in metrics:
                            file_mode = "w" if epoch == 0 else "a"
                            with open(os.path.join("local_loss", "val_delta1.txt"), file_mode) as f:
                                f.write(f"{epoch}\t{metrics['delta1']}\n")
                except Exception:
                    pass

            depth_evaluator.reset_buffer()
        else:
            warn(f'depth_evaluator has no data in {mode=}', UserWarning, stacklevel=2)
    
    def on_train_epoch_end(self) -> None:
        mode = "train"
        depth_metrics_every_n_steps = self.train_metrics_config.get('depth_metrics_every_n_steps')
        if self.train_depth_evaluator is not None and \
                depth_metrics_every_n_steps is None and \
                self.train_hw is not None:
            self.run_depth_evaluator(mode=mode)

        # 将本 epoch 的平均 train loss 写入本地文件（仅 rank 0）
        if self.global_rank == 0 and self._epoch_loss_count > 0:
            try:
                import os
                epoch = self.current_epoch
                avg_loss = self._epoch_loss_sum / self._epoch_loss_count
                os.makedirs("local_loss", exist_ok=True)
                file_mode = "w" if epoch == 0 else "a"
                with open(os.path.join("local_loss", "train_loss.txt"), file_mode) as f:
                    f.write(f"{epoch}\t{avg_loss}\n")
            except Exception:
                pass
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

        # 将本 epoch 结束时的学习率写入本地文件（仅 rank 0）
        if self.global_rank == 0:
            try:
                import os
                epoch = self.current_epoch
                lr = self.trainer.optimizers[0].param_groups[0]['lr']
                os.makedirs("local_loss", exist_ok=True)
                file_mode = "w" if epoch == 0 else "a"
                with open(os.path.join("local_loss", "train_lr.txt"), file_mode) as f:
                    f.write(f"{epoch}\t{lr}\n")
            except Exception:
                pass
    
    def on_validation_epoch_end(self) -> None:
        mode = "val"
        if self.started_training and self.val_depth_evaluator is not None and self.val_depth_evaluator.has_data():
            self.run_depth_evaluator(mode=mode)

        # 将本 epoch 的平均验证 loss 写入本地文件（仅 rank 0）
        if self.global_rank == 0 and self._val_epoch_loss_count > 0:
            try:
                import os
                epoch = self.current_epoch
                avg_val_loss = self._val_epoch_loss_sum / self._val_epoch_loss_count
                os.makedirs("local_loss", exist_ok=True)
                file_mode = "w" if epoch == 0 else "a"
                with open(os.path.join("local_loss", "val_loss.txt"), file_mode) as f:
                    f.write(f"{epoch}\t{avg_val_loss}\n")
            except Exception:
                pass
        self._val_epoch_loss_sum = 0.0
        self._val_epoch_loss_count = 0
    
    def on_test_epoch_end(self) -> None:
        mode = "test"
        if self.test_depth_evaluator is not None and self.test_depth_evaluator.has_data():
            self.run_depth_evaluator(mode=mode)
    
    def configure_optimizers(self) -> Any:
        lr = self.train_config['learning_rate']
        weight_decay = self.train_config.get('weight_decay', 0)
        optimizer = th.optim.AdamW(self.mdl.parameters(), lr=lr, weight_decay=weight_decay)
        
        scheduler_params = self.train_config.get('lr_scheduler', {})
        if not scheduler_params.get('use', False):
            return optimizer
        
        total_steps = scheduler_params.get('total_steps')
        assert total_steps is not None
        assert total_steps > 0
        final_div_factor_pytorch = scheduler_params['final_div_factor'] / scheduler_params['div_factor']
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            div_factor=scheduler_params['div_factor'],
            final_div_factor=final_div_factor_pytorch,
            total_steps=total_steps,
            pct_start=scheduler_params['pct_start'],
            cycle_momentum=False,
            anneal_strategy='cos')
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": 'learning_rate',
        }
        
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
    

