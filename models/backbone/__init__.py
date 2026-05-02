from typing import Dict, Any

from .mobilenet_rnn import MobileNetRNN
from .e2depth_unet import E2DepthConvLSTMUNet
from .base import BaseDetector


def build_recurrent_backbone(backbone_cfg: Dict[str, Any]):
    """
    構建循環 backbone 的統一入口。
    原先位於 `models.detection.recurrent_backbone`，現移至 `models.backbone`。
    """
    name = backbone_cfg["name"]
    if name == "MobileNetRNN":
        return MobileNetRNN(backbone_cfg)
    if name == "E2DepthConvLSTMUNet":
        return E2DepthConvLSTMUNet(backbone_cfg)
    raise NotImplementedError(
        f"Backbone '{name}' not implemented. Expected 'MobileNetRNN' or 'E2DepthConvLSTMUNet'."
    )


__all__ = ["MobileNetRNN", "E2DepthConvLSTMUNet", "BaseDetector", "build_recurrent_backbone"]

