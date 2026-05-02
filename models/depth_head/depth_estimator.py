"""
Depth Estimator - Similar to YoloXDetector but for depth estimation
Uses the same backbone and FPN, but replaces detection head with depth head.

原始實現在 `models/depth_estimator.py`，現移至 `models/depth_head/depth_estimator.py`，
作為只依賴 backbone / fpn / depth_head 的統一入口。
"""
from typing import Dict, Optional, Tuple, Union, Any

import torch as th

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from models.backbone import build_recurrent_backbone
from models.fpn.build import build_yolox_fpn
from .depth_head import build_depth_head
from .depth_losses import DepthLoss
from utils.timers import CudaTimer

from data.utils.types import BackboneFeatures, LstmStates


class DepthEstimator(th.nn.Module):
    """
    Depth estimation network
    Reuses backbone and FPN from detection, adds UNet-style depth decoder;
    optional E2DepthConvLSTMUNet end-to-end path (no FPN / legacy depth_head).
    """

    def __init__(self, model_cfg: Dict[str, Any]):
        super().__init__()
        backbone_cfg = model_cfg["backbone"]
        loss_cfg = model_cfg.get("loss", {})
        depth_range_cfg = model_cfg.get("depth_range", {})
        min_depth = depth_range_cfg.get("min", 0.5) if isinstance(depth_range_cfg, dict) else 0.5
        max_depth = depth_range_cfg.get("max", 80.0) if isinstance(depth_range_cfg, dict) else 80.0

        # Build backbone
        self.backbone = build_recurrent_backbone(backbone_cfg)
        self._e2depth = bool(getattr(self.backbone, "is_end_to_end_depth", False))

        if self._e2depth:
            self.fpn = None
            self.depth_head = None
        else:
            fpn_cfg = model_cfg.get("fpn")
            head_cfg = model_cfg.get("head")
            if not isinstance(fpn_cfg, dict) or not isinstance(head_cfg, dict):
                raise ValueError(
                    "model.fpn and model.head must be dict configs when backbone is not E2DepthConvLSTMUNet"
                )
            in_channels = self.backbone.get_stage_dims(tuple(fpn_cfg["in_stages"]))
            self.fpn = build_yolox_fpn(fpn_cfg, in_channels=in_channels)
            skip_quarter_ch = self.backbone.get_stage_dims((1,))[0]
            self.depth_head = build_depth_head(
                head_cfg, in_channels=in_channels, skip_quarter_channels=skip_quarter_ch
            )

        far_weight_cfg = loss_cfg.get("far_weight", {}) or {}
        if not isinstance(far_weight_cfg, dict):
            far_weight_cfg = {}

        composition = loss_cfg.get("composition", "full")

        loss_common = dict(
            composition=composition,
            depth_min=min_depth,
            depth_max=max_depth,
        )

        if composition == "sigrad_only":
            self.loss_fn = DepthLoss(
                **loss_common,
                si_weight=float(loss_cfg.get("si_weight", 1.0)),
                si_lambda=float(loss_cfg.get("si_lambda", loss_cfg.get("silog_lambda", 1.0))),
                grad_weight=float(loss_cfg.get("grad_weight", 0.25)),
                grad_start_scale=int(loss_cfg.get("grad_start_scale", 1)),
                grad_num_scales=int(loss_cfg.get("grad_num_scales", 4)),
                berhu_weight=float(loss_cfg.get("berhu_weight", 0.0)),
                berhu_threshold=float(loss_cfg.get("berhu_threshold", 0.2)),
            )
        else:
            self.loss_fn = DepthLoss(
                composition="full",
                depth_min=min_depth,
                depth_max=max_depth,
                silog_weight=loss_cfg.get("silog_weight", 1.0),
                grad_weight=loss_cfg.get("grad_weight", 0.25),
                silog_lambda=loss_cfg.get("silog_lambda", 1.0),
                scales=loss_cfg.get("scales", [1, 2, 4, 8, 16]),
                far_weight_alpha=far_weight_cfg.get("alpha", 0.0),
                far_weight_t0=far_weight_cfg.get("t0", 0.3),
                lap_weight=loss_cfg.get("lap_weight", 0.0),
                event_edge_sigma=loss_cfg.get("event_edge_sigma", 3.0),
                event_edge_grad_ratio=loss_cfg.get("event_edge_grad_ratio", 0.5),
                scale1_weight_mul=loss_cfg.get("scale1_weight_mul", 2.5),
            )

    def forward_backbone(
        self,
        x: th.Tensor,
        previous_states: Optional[LstmStates] = None,
        token_mask: Optional[th.Tensor] = None,
    ) -> Tuple[BackboneFeatures, LstmStates]:
        """Forward through backbone"""
        with CudaTimer(device=x.device, timer_name="Backbone"):
            backbone_features, states = self.backbone(x, previous_states, token_mask)
        return backbone_features, states

    def forward_depth(
        self,
        backbone_features: BackboneFeatures,
        targets: Optional[th.Tensor] = None,
        masks: Optional[th.Tensor] = None,
        event_repr: Optional[th.Tensor] = None,
    ) -> Tuple[Dict[str, th.Tensor], Union[Dict[str, th.Tensor], None]]:
        """
        Forward through FPN and depth head

        Args:
            backbone_features: features from backbone
            targets: ground truth depth in log space (B, 1, H, W) [optional for training]
            masks: valid depth mask (B, 1, H, W) [optional for training]

        Returns:
            predictions: dict with multi-scale depth predictions
            losses: dict with loss values (None if not training)
        """
        device = next(iter(backbone_features.values())).device

        with CudaTimer(device=device, timer_name="FPN"):
            fpn_features = self.fpn(backbone_features)

        with CudaTimer(device=device, timer_name="Depth Head"):
            predictions = self.depth_head(
                fpn_features, feat_skip_quarter=backbone_features[1]
            )

        losses = None
        if targets is not None:
            with CudaTimer(device=device, timer_name="Depth Loss"):
                total_loss, losses_dict = self.loss_fn(
                    predictions, targets, masks, event_repr
                )
                losses = losses_dict

        return predictions, losses

    def forward(
        self,
        x: th.Tensor,
        previous_states: Optional[LstmStates] = None,
        retrieve_depth: bool = True,
        targets: Optional[th.Tensor] = None,
        masks: Optional[th.Tensor] = None,
        token_mask: Optional[th.Tensor] = None,
    ) -> Tuple[
        Union[Dict[str, th.Tensor], None],
        Union[Dict[str, th.Tensor], None],
        LstmStates,
    ]:
        """
        Full forward pass

        Args:
            x: input event representation
            previous_states: RNN states from previous timestep
            retrieve_depth: whether to compute depth predictions
            targets: ground truth depth in log space
            masks: valid depth mask
            token_mask: token mask for backbone

        Returns:
            predictions: dict with depth predictions (None if retrieve_depth=False)
            losses: dict with loss values (None if not training)
            states: updated RNN states
        """
        predictions, losses = None, None

        if self._e2depth:
            with CudaTimer(device=x.device, timer_name="E2DepthUNet"):
                predictions, states = self.backbone.forward_depth_and_states(
                    x, previous_states
                )
            if not retrieve_depth:
                assert targets is None
                return None, None, states
            if targets is not None:
                with CudaTimer(device=x.device, timer_name="Depth Loss"):
                    _, losses_dict = self.loss_fn(
                        predictions, targets, masks, None
                    )
                losses = losses_dict
            return predictions, losses, states

        backbone_features, states = self.forward_backbone(
            x, previous_states, token_mask
        )

        if not retrieve_depth:
            assert targets is None
            return predictions, losses, states

        event_repr = x if targets is not None else None
        predictions, losses = self.forward_depth(
            backbone_features=backbone_features,
            targets=targets,
            masks=masks,
            event_repr=event_repr,
        )

        return predictions, losses, states


