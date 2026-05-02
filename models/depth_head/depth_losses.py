"""
Depth estimation loss functions
Includes SILog (Scale-Invariant Log) and image gradient loss

重要说明：
- 训练数据与网络原始输出目前都是“log(depth)”标量空间；
- 本文件会在 loss 计算之前，把 `pred` 和 `target` 都转换到
  “归一化 log(depth)（min_depth=0.5, max_depth=80）”空间，再计算误差。
"""
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaleInvariantMaskedLoss(nn.Module):
    """
    SI loss in normalized log domain: inputs are already in norm_log space;
    masked mean/var over valid pixels only.
    """

    def __init__(self, weight: float = 1.0, n_lambda: float = 1.0):
        super().__init__()
        self.weight = float(weight)
        self.n_lambda = float(n_lambda)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff = pred - target
        if mask is None:
            valid = torch.isfinite(diff.flatten())
            d = diff.flatten()[valid]
        else:
            m = mask
            if m.dtype != torch.bool:
                m = m.bool()
            if m.shape != pred.shape:
                raise ValueError(f"mask shape {m.shape} != pred shape {pred.shape}")
            finite = torch.isfinite(diff)
            mv = (m & finite).flatten()
            d = diff.flatten()[mv]

        if d.numel() == 0:
            return pred.new_zeros(())

        mse = (d ** 2).mean()
        mean_d = d.mean()
        return self.weight * (mse - self.n_lambda * (mean_d ** 2))


class E2DepthMultiScaleGradientLoss(nn.Module):
    """E2Depth-style multi-scale gradients on pooled residual (no kornia)."""

    def __init__(self, start_scale: int = 1, num_scales: int = 4):
        super().__init__()
        self.num_scales = int(num_scales)
        ks = []
        base = max(1, int(start_scale))
        for s in range(self.num_scales):
            k = base * (2 ** s)
            ks.append(nn.AvgPool2d(kernel_size=k, stride=k))
        self.multi_scales = nn.ModuleList(ks)

        gx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        gy = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", gx)
        self.register_buffer("sobel_y", gy)

    def _spatial_gradient(self, diff: torch.Tensor) -> torch.Tensor:
        """Sobel on each channel -> (B, C, 2, H, W)."""
        b, c, h, w = diff.shape
        x = diff.reshape(b * c, 1, h, w).to(dtype=torch.float32)
        kx = self.sobel_x.to(device=x.device, dtype=x.dtype)
        ky = self.sobel_y.to(device=x.device, dtype=x.dtype)
        gx = F.conv2d(x, kx, padding=1).view(b, c, h, w)
        gy = F.conv2d(x, ky, padding=1).view(b, c, h, w)
        return torch.stack((gx, gy), dim=2)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff = prediction - target
        diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
        if mask is not None:
            md = mask if mask.dtype == torch.bool else mask.bool()
            diff = torch.where(md, diff, torch.zeros_like(diff))

        loss_value = diff.new_zeros(())
        device_type = prediction.device.type
        autocast_on = torch.is_autocast_enabled() if device_type == "cuda" else False

        for m_pool in self.multi_scales:
            pd = diff
            ksz = getattr(m_pool, "kernel_size", None)
            if ksz is not None:
                k0 = int(ksz) if isinstance(ksz, int) else int(ksz[0])
            else:
                k0 = 1
            pd_min = min(pd.shape[-2], pd.shape[-1])
            if pd_min < k0:
                continue
            if autocast_on:
                with torch.amp.autocast(device_type=device_type, enabled=False):
                    d_s = m_pool(pd.float()).to(dtype=diff.dtype)
            else:
                d_s = m_pool(pd.float()).to(dtype=diff.dtype)

            if d_s.shape[-1] < 2 or d_s.shape[-2] < 2:
                continue

            if autocast_on:
                with torch.amp.autocast(device_type=device_type, enabled=False):
                    delta_diff = self._spatial_gradient(d_s.float())
            else:
                delta_diff = self._spatial_gradient(d_s.float())

            loss_value = loss_value + delta_diff.abs().mean(dim=(3, 4)).sum().to(loss_value.dtype)

        return loss_value / max(float(self.num_scales), 1.0)


class BerHuLoss(nn.Module):
    """
    Reverse Huber (BerHu) loss
    Behaves like L1 for small errors and L2 for large errors
    """
    def __init__(self, threshold: float = 0.2):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            pred: predicted depth (normalized log space)
            target: ground truth depth (normalized log space)
            mask: valid depth mask
        """
        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        # If there are no valid pixels, return zero loss to avoid max() on empty tensor
        if pred.numel() == 0 or target.numel() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        diff = torch.abs(pred - target)
        delta = self.threshold * torch.max(diff).item()
        
        # BerHu: L1 for small errors, L2 for large errors
        berhu_loss = torch.where(
            diff <= delta,
            diff,
            (diff ** 2 + delta ** 2) / (2 * delta)
        )
        
        return berhu_loss.mean()


class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic loss
    Invariant to global scale of the scene

    可选：远距像素软加权
        w_i = 1 + alpha * clamp((t_i - t0) / (1 - t0), 0, 1)
    其中 t_i 为 target 在 norm_log 空间的值（∈[0,1]）。
    alpha=0 时严格等价于未加权版本。权重只依赖 target，且会 detach，不参与反传。
    """
    def __init__(self, lambd: float = 1.0, far_weight_alpha: float = 0.0, far_weight_t0: float = 0.3):
        super().__init__()
        self.lambd = lambd
        self.far_weight_alpha = float(far_weight_alpha)
        self.far_weight_t0 = float(far_weight_t0)
        if not (0.0 <= self.far_weight_t0 < 1.0):
            raise ValueError(f"far_weight_t0 must be in [0, 1), got {far_weight_t0}")

    def _far_weight(self, target_norm_log: torch.Tensor) -> torch.Tensor:
        """基于 target 的 per-pixel 远距权重；alpha=0 时为全 1。"""
        if self.far_weight_alpha == 0.0:
            return torch.ones_like(target_norm_log)
        denom = max(1.0 - self.far_weight_t0, 1e-6)
        w = 1.0 + self.far_weight_alpha * torch.clamp(
            (target_norm_log - self.far_weight_t0) / denom, 0.0, 1.0
        )
        return w.detach()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            pred: predicted depth (normalized log space)
            target: ground truth depth (normalized log space)
            mask: valid depth mask
        """
        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        # If there are no valid pixels, return zero loss
        if pred.numel() == 0 or target.numel() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        diff = pred - target

        if self.far_weight_alpha == 0.0:
            loss = torch.mean(diff ** 2) - self.lambd * (torch.mean(diff) ** 2)
            return loss

        # 加权版 SILog：用 target 作为远近度量计算权重，保持尺度不变项形式
        w = self._far_weight(target)
        w_sum = w.sum().clamp(min=1e-6)
        mean_wdiff2 = (w * diff ** 2).sum() / w_sum
        mean_wdiff = (w * diff).sum() / w_sum
        loss = mean_wdiff2 - self.lambd * (mean_wdiff ** 2)
        return loss


class MultiScaleGradientLoss(nn.Module):
    """
    E2Depth 风格的多尺度梯度匹配损失 (Hidalgo-Carrio et al., 3DV 2020)。

    注意：这里的 "多尺度" 指的是**同一张最终 residual 的不同下采样尺度**，
    而不是 decoder 的多个输出头。公式：

        R = pred_final - target
        R^s = downsample^s(R)      # s = 0, 1, 2, 3
        L_grad = (1 / n_valid) * sum_s sum_u ( |∂x R^s(u)| + |∂y R^s(u)| )

    作用是鼓励深度平滑、同时保留清晰的深度不连续边界。

    稀疏 GT / mask 处理：
    - residual 与 mask 同步 avg_pool2d 下采样；
    - 下采样过程中使用 "只对有效像素求均值" 的方式，避免无效像素把残差人为拉向 0；
    - 梯度项只在 **相邻两格都有效** 的位置计入损失；
    - 某尺度有效对数过少时，不会触发除零。
    """

    def __init__(self, num_scales: int = 4, mask_threshold: float = 0.0):
        super().__init__()
        self.num_scales = int(num_scales)
        # 下采样后一个 2x2 块的有效占比严格大于该阈值，才视为该粗尺度块有效
        # 0.0 表示 "只要块内有一个有效像素就保留"，对稀疏 LiDAR 比较友好
        self.mask_threshold = float(mask_threshold)

    @staticmethod
    def _gradient_xy(x: torch.Tensor):
        grad_x = x[:, :, :, :-1] - x[:, :, :, 1:]
        grad_y = x[:, :, :-1, :] - x[:, :, 1:, :]
        return grad_x, grad_y

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            pred:   最终（最高分辨率）深度预测，(B, 1, H, W)，与 target 在同一空间
                    （当前项目是 norm_log 空间）。
            target: GT 深度，(B, 1, H, W)，同上空间。
            mask:   有效像素 mask，(B, 1, H, W) bool/0-1，可为 None。
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have same shape, got {pred.shape} vs {target.shape}"
            )

        residual = pred - target

        if mask is not None:
            mask_f = mask.to(residual.dtype)
            # 把无效像素处的 residual 置 0；后续用 mask 归一化，等价于 "对有效像素求均值"
            cur_res = residual * mask_f
            cur_mask = mask_f
        else:
            cur_res = residual
            cur_mask = None

        total_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
        total_count = torch.zeros((), device=pred.device, dtype=pred.dtype)

        for s in range(self.num_scales):
            if s > 0:
                # —— 同步下采样 residual 和 mask ——
                if cur_mask is not None:
                    weighted = F.avg_pool2d(cur_res, kernel_size=2, stride=2)
                    mask_avg = F.avg_pool2d(cur_mask, kernel_size=2, stride=2)
                    # 仅对块内有效像素求均值：sum(res*mask)/sum(mask)
                    cur_res = weighted / mask_avg.clamp(min=1e-6)
                    cur_mask = (mask_avg > self.mask_threshold).to(cur_res.dtype)
                    cur_res = cur_res * cur_mask  # 无效块再置 0，防止数值泄漏
                else:
                    cur_res = F.avg_pool2d(cur_res, kernel_size=2, stride=2)

            # 防御：太小的 feature map 无法做差分
            if cur_res.shape[-1] < 2 or cur_res.shape[-2] < 2:
                break

            grad_x, grad_y = self._gradient_xy(cur_res)

            if cur_mask is not None:
                # 只有相邻两格都有效，对应位置的梯度才计入
                mx = cur_mask[:, :, :, :-1] * cur_mask[:, :, :, 1:]
                my = cur_mask[:, :, :-1, :] * cur_mask[:, :, 1:, :]
                total_loss = total_loss + (grad_x.abs() * mx).sum() + (grad_y.abs() * my).sum()
                total_count = total_count + mx.sum() + my.sum()
            else:
                total_loss = total_loss + grad_x.abs().sum() + grad_y.abs().sum()
                total_count = total_count + torch.tensor(
                    float(grad_x.numel() + grad_y.numel()),
                    device=pred.device, dtype=pred.dtype,
                )

        # 整体按有效像素数归一化；有效像素过少时 clamp 到 1 可避免除零
        # （此时 total_loss 本身也近似为 0，不会制造虚假梯度）
        return total_loss / total_count.clamp(min=1.0)


class LaplacianLoss(nn.Module):
    """对归一化 log 深度作 Laplacian 匹配，加强深度不连续处的二阶监督（仅训练期）。"""

    def __init__(self):
        super().__init__()
        k = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel", k, persistent=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        k = self.kernel.to(device=pred.device, dtype=pred.dtype)
        lap_p = F.conv2d(pred, k, padding=1)
        lap_t = F.conv2d(target, k, padding=1)
        diff = (lap_p - lap_t).abs()
        if mask is not None:
            m = mask
            if m.dtype != torch.bool:
                m = m.bool()
            if m.shape != diff.shape:
                m = m.expand_as(diff)
            diff = diff[m]
        if diff.numel() == 0:
            return pred.new_tensor(0.0)
        return diff.mean()


def event_edge_weight(event_repr: torch.Tensor, sigma: float = 3.0) -> torch.Tensor:
    """
    从事件表示得到边缘权重图：活动强处权重大。返回 (B, 1, H, W)，值域约 [1, 1+sigma]。
    """
    with torch.no_grad():
        activity = event_repr.abs().sum(dim=1, keepdim=True).float()
        b = activity.shape[0]
        a_max = activity.view(b, -1).max(dim=1).values.view(b, 1, 1, 1).clamp(min=1e-6)
        activity = activity / a_max
        return 1.0 + float(sigma) * activity


class DepthLoss(nn.Module):
    """
    Combined depth loss.

    composition == "full": multi-resolution SILog + masked multi-scale gradients + extras.
    composition == "sigrad_only": masked scale-invariant (norm_log) + E2Depth multi-scale Sobel(grad).
    """

    def __init__(
        self,
        composition: str = "full",
        silog_weight: float = 1.0,
        grad_weight: float = 0.25,
        silog_lambda: float = 1.0,
        scales: Optional[List[int]] = None,
        depth_min: float = 0.5,
        depth_max: float = 80.0,
        far_weight_alpha: float = 0.0,
        far_weight_t0: float = 0.3,
        lap_weight: float = 0.0,
        event_edge_sigma: float = 3.0,
        event_edge_grad_ratio: float = 0.5,
        scale1_weight_mul: float = 1.0,
        si_weight: float = 1.0,
        si_lambda: float = 1.0,
        grad_start_scale: int = 1,
        grad_num_scales: int = 4,
    ):
        super().__init__()
        self.composition = composition
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)

        if self.depth_min <= 0 or self.depth_max <= 0:
            raise ValueError(f"depth_min/depth_max must be > 0, got {depth_min=} {depth_max=}")
        if self.depth_min >= self.depth_max:
            raise ValueError(f"depth_min must be < depth_max, got {depth_min=} {depth_max=}")

        self._log_depth_min = math.log(self.depth_min)
        self._log_depth_max = math.log(self.depth_max)
        self._log_depth_denom = self._log_depth_max - self._log_depth_min

        if composition == "sigrad_only":
            self.si_loss = ScaleInvariantMaskedLoss(weight=si_weight, n_lambda=si_lambda)
            self.ms_grad_e2 = E2DepthMultiScaleGradientLoss(
                start_scale=grad_start_scale, num_scales=grad_num_scales
            )
            self.grad_weight = float(grad_weight)
            self.silog_loss = None
            self.grad_loss = None
            self.lap_loss = None
            self.scales = []
            self.silog_weight = 0.0
            self.lap_weight = 0.0
            self.event_edge_grad_ratio = 0.0
            self.scale1_weight_mul = 1.0
            self.event_edge_sigma = float(event_edge_sigma)
        else:
            if scales is None:
                scales = [1, 2, 4, 8, 16]
            self.silog_weight = silog_weight
            self.grad_weight = grad_weight
            self.scales = scales
            self.lap_weight = float(lap_weight)
            self.event_edge_sigma = float(event_edge_sigma)
            self.event_edge_grad_ratio = float(event_edge_grad_ratio)
            self.scale1_weight_mul = float(scale1_weight_mul)
            self.silog_loss = SILogLoss(
                lambd=silog_lambda,
                far_weight_alpha=far_weight_alpha,
                far_weight_t0=far_weight_t0,
            )
            self.grad_loss = MultiScaleGradientLoss(num_scales=4)
            self.lap_loss = LaplacianLoss()
            self.si_loss = None
            self.ms_grad_e2 = None

    def log_depth_to_norm_log_depth(self, log_depth: torch.Tensor) -> torch.Tensor:
        """
        兼容旧流程：log(depth) -> 归一化 log 空间(norm_log)。

        当前 B 方案下，网络输出与 GT 目标已被预处理到 norm_log，
        loss 不再对 pred 做该变换，函数仅保留以兼容其它调用/调试。
        """
        log_depth_clamped = torch.clamp(log_depth, min=self._log_depth_min, max=self._log_depth_max)
        return (log_depth_clamped - self._log_depth_min) / max(self._log_depth_denom, 1e-6)

    @staticmethod
    def _masked_avg_pool_to_size(
        target: torch.Tensor,
        mask: torch.Tensor,
        out_hw,
    ):
        """Masked adaptive average pooling to arbitrary output size.

        设 sum_m  = pool(mask_float)      （块内有效像素占比）
           sum_tm = pool(target*mask_float)
        则 target_down = sum_tm / sum_m  在有效块处取值，其余位置置 0；
           mask_down    = (sum_m > 0)    （块内至少一个有效像素）。

        - 使用 adaptive_avg_pool2d，兼容非整数下采样比（如 260→65, 348→44 等）。
        - 强制 fp32 计算，避免 AMP/fp16 下分母过小导致的数值问题。
        - 若未提供 mask，则退化为普通 adaptive avg pool（返回 mask_down=None）。

        Args:
            target: (B, 1, H, W) 在 [0, 1] 的 norm_log 空间。
            mask:   (B, 1, H, W) bool 或浮点 0/1，或 None。
            out_hw: 目标 (h, w)。

        Returns:
            target_down: (B, 1, h, w) 与 target 同 dtype。
            mask_down:   (B, 1, h, w) bool；无输入 mask 时返回 None。
        """
        out_hw = tuple(out_hw)
        src_dtype = target.dtype
        device_type = target.device.type

        # 跳过 AMP，避免 fp16 下 1/den 的数值敏感问题
        autocast_enabled = torch.is_autocast_enabled() if device_type == 'cuda' else False

        def _compute():
            t32 = target.float()
            if mask is None:
                out = F.adaptive_avg_pool2d(t32, output_size=out_hw)
                return out.to(src_dtype), None
            m32 = (mask.to(torch.float32) if mask.dtype != torch.float32 else mask.float())
            num = F.adaptive_avg_pool2d(t32 * m32, output_size=out_hw)
            den = F.adaptive_avg_pool2d(m32, output_size=out_hw)
            mask_down = den > 0.0
            # den=0 处 num 也=0；先 clamp 再 where，避免 0/0 造 NaN
            t_down = num / den.clamp(min=1e-6)
            t_down = torch.where(mask_down, t_down, torch.zeros_like(t_down))
            return t_down.to(src_dtype), mask_down

        if autocast_enabled:
            with torch.amp.autocast(device_type=device_type, enabled=False):
                return _compute()
        return _compute()

    # def forward(self, predictions: dict, target: torch.Tensor, mask: torch.Tensor = None):
    #     """
    #     Args:
    #         predictions: dict with keys like 'depth_2', 'depth_4', 'depth_8', 'depth_16'
    #         target: ground truth depth (B, 1, H, W) in log space
    #         mask: valid depth mask (B, 1, H, W)
        
    #     Returns:
    #         dict of losses
    #     """
    #     total_loss = 0.0
    #     losses_dict = {}
        
    #     # Process each scale
    #     for scale in self.scales:
    #         key = f'depth_{scale}'
    #         if key not in predictions:
    #             continue
            
    #         pred = predictions[key]
            
    #         # Downsample target and mask to match prediction scale
    #         target_scaled = F.interpolate(
    #             target, size=pred.shape[-2:], mode='bilinear', align_corners=False
    #         )
            
    #         if mask is not None:
    #             mask_scaled = F.interpolate(
    #                 mask.float(), size=pred.shape[-2:], mode='nearest'
    #             ).bool()
    #         else:
    #             mask_scaled = None
            
    #         # Compute individual losses
    #         berhu = self.berhu_loss(pred, target_scaled, mask_scaled)
    #         silog = self.silog_loss(pred, target_scaled, mask_scaled)
    #         grad = self.grad_loss(pred, target_scaled, mask_scaled)
            
    #         # Weighted combination
    #         scale_loss = (
    #             self.berhu_weight * berhu +
    #             self.silog_weight * silog +
    #             self.grad_weight * grad
    #         )
            
    #         # Weight loss by scale (higher weight for finer scales)
    #         scale_weight = 1.0 / scale  # Higher resolution gets more weight
    #         total_loss += scale_weight * scale_loss
            
    #         # Store individual losses for logging
    #         losses_dict[f'berhu_{scale}'] = berhu.item()
    #         losses_dict[f'silog_{scale}'] = silog.item()
    #         losses_dict[f'grad_{scale}'] = grad.item()
    #         losses_dict[f'loss_{scale}'] = scale_loss.item()
        
    #     # Normalize by number of scales
    #     total_loss = total_loss / len(self.scales)
    #     losses_dict['loss'] = total_loss
        
    #     return total_loss, losses_dict

    def _forward_sigrad_only(
        self,
        predictions,
        target: torch.Tensor,
        mask,
        event_repr: Optional[torch.Tensor],
    ):
        _ = event_repr
        finest_key = "depth_1"
        if finest_key not in predictions:
            raise KeyError(
                f"sigrad_only requires '{finest_key}' in predictions, keys={tuple(predictions.keys())}"
            )
        pred_final = predictions[finest_key]
        target_hw = target.shape[-2:]
        if pred_final.shape[-2:] != target_hw:
            pred_final = F.interpolate(
                pred_final, size=target_hw, mode="bilinear", align_corners=False
            )
        pred_final_norm_log = torch.clamp(pred_final, 0.0, 1.0)
        target_norm_log = torch.clamp(target, 0.0, 1.0)
        m = mask
        if m is not None and m.dtype != torch.bool:
            m = m.bool()

        si = self.si_loss(pred_final_norm_log, target_norm_log, m)
        assert self.ms_grad_e2 is not None
        g_e2 = self.ms_grad_e2(pred_final_norm_log, target_norm_log, m)
        total_loss = si + self.grad_weight * g_e2
        losses_dict = {
            "si_masked": si.detach(),
            "grad_e2_multiscale": g_e2.detach(),
            "loss": total_loss,
        }
        return total_loss, losses_dict

    def forward(
        self,
        predictions,
        target,
        mask=None,
        event_repr: Optional[torch.Tensor] = None,
    ):
        if self.composition == "sigrad_only":
            return self._forward_sigrad_only(predictions, target, mask, event_repr)
        # composition == "full": multi-scale SILog + MultiScaleGradientLoss + optional lap/event edge
        total_loss = torch.zeros((), device=target.device, dtype=target.dtype)
        losses_dict = {}

        target_hw = target.shape[-2:]
        target_norm_log = torch.clamp(target, 0.0, 1.0)
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()

        # ---- (1) SILog 主损失：每层 pred 对齐自己分辨率的 masked-pool target/mask ----
        for scale in self.scales:
            key = f'depth_{scale}'
            if key not in predictions:
                continue

            pred = predictions[key]
            pred_hw = pred.shape[-2:]
            pred_norm_log = torch.clamp(pred, 0.0, 1.0)

            if pred_hw == target_hw:
                tgt_for_silog = target_norm_log
                mask_for_silog = mask
            else:
                tgt_for_silog, mask_for_silog = self._masked_avg_pool_to_size(
                    target_norm_log, mask, pred_hw
                )

            silog = self.silog_loss(pred_norm_log, tgt_for_silog, mask_for_silog)
            scale_loss = self.silog_weight * silog

            if self.lap_weight > 0.0 and scale <= 2:
                lap = self.lap_loss(pred_norm_log, tgt_for_silog, mask_for_silog)
                scale_loss = scale_loss + self.lap_weight * lap
                losses_dict[f'lap_{scale}'] = lap.detach()

            scale_weight = 1.0 / float(scale)  # 高分辨率头基础权重更大
            if scale == 1 and self.scale1_weight_mul != 1.0:
                scale_weight = scale_weight * self.scale1_weight_mul
            total_loss = total_loss + scale_weight * scale_loss

            losses_dict[f'silog_{scale}'] = silog.detach()

        # ---- (2) E2Depth 多尺度 gradient loss：最精细预测 + 可选事件边缘加权 ----
        finest_scale = min(self.scales) if len(self.scales) > 0 else None
        finest_key = f'depth_{finest_scale}' if finest_scale is not None else None
        if finest_key is not None and finest_key in predictions:
            pred_final = predictions[finest_key]
            if pred_final.shape[-2:] != target_hw:
                pred_final = F.interpolate(
                    pred_final, size=target_hw, mode='bilinear', align_corners=False
                )
            pred_final_norm_log = torch.clamp(pred_final, 0.0, 1.0)

            grad_multiscale = self.grad_loss(
                pred_final_norm_log, target_norm_log, mask
            )
            total_loss = total_loss + self.grad_weight * grad_multiscale
            losses_dict['grad_multiscale'] = grad_multiscale.detach()

            if (
                event_repr is not None
                and self.event_edge_grad_ratio > 0.0
            ):
                edge_w = event_edge_weight(event_repr, sigma=self.event_edge_sigma)
                if edge_w.shape[-2:] != target_hw:
                    edge_w = F.interpolate(
                        edge_w, size=target_hw, mode="nearest"
                    )
                g_edge = self.grad_loss(
                    pred_final_norm_log * edge_w,
                    target_norm_log * edge_w,
                    mask,
                )
                w_edge = self.grad_weight * self.event_edge_grad_ratio
                total_loss = total_loss + w_edge * g_edge
                losses_dict['grad_multiscale_event_edge'] = g_edge.detach()

        losses_dict['loss'] = total_loss
        return total_loss, losses_dict
