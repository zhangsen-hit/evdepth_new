"""
E2Depth-style ConvLSTM U-Net backbone: head + 3 encoder ConvLSTM stages + bottleneck residuals
+ symmetric decoder + prediction head (sigmoid in norm_log).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from data.utils.types import LstmStates
from models.backbone.base import BaseDetector
from models.backbone.rnn import DWSConvLSTM2d, StandardConvLSTM2d


def _relu_inplace() -> nn.ReLU:
    return nn.ReLU(inplace=True)


def _build_convlstm(
    encoder_lstm_type: str,
    dim: int,
    lstm_cfg: Dict[str, Any],
    T_max_chrono_init: Optional[int],
) -> nn.Module:
    t = encoder_lstm_type
    drop = lstm_cfg.get("drop_cell_update", 0)
    if t == "stand_convlstm":
        return StandardConvLSTM2d(
            dim=dim,
            cell_update_dropout=drop,
            T_max_chrono_init=T_max_chrono_init,
        )
    if t == "dws_convlstm":
        return DWSConvLSTM2d(
            dim=dim,
            dws_conv=lstm_cfg.get("dws_conv", True),
            dws_conv_only_hidden=lstm_cfg.get("dws_conv_only_hidden", True),
            dws_conv_kernel_size=lstm_cfg.get("dws_conv_kernel_size", 3),
            cell_update_dropout=drop,
            T_max_chrono_init=T_max_chrono_init,
        )
    raise ValueError(
        f"encoder_lstm_type must be 'stand_convlstm' or 'dws_convlstm', got {t!r}"
    )


class _Conv_relu(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        padding: int,
        use_bn: bool,
    ):
        bias = not use_bn
        layers = [nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(_relu_inplace())
        super().__init__(*layers)


class _ResidualBlock256(nn.Module):
    def __init__(self, use_bn: bool):
        super().__init__()
        bias = not use_bn
        self.conv1 = nn.Conv2d(256, 256, 3, 1, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(256) if use_bn else nn.Identity()
        self.act1 = _relu_inplace()
        self.conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(256) if use_bn else nn.Identity()
        self.act2 = _relu_inplace()

    def forward(self, x: th.Tensor) -> th.Tensor:
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        return self.act2(x + y)


class E2DepthConvLSTMUNet(BaseDetector):
    """
    LiOSAM / E2Depth-style U-Net with three encoder ConvLSTMs.

    Marks itself for DepthEstimator via `is_end_to_end_depth=True`.
    Use `forward_depth_and_states(x, prev_states)` for predictions + recurrent state.
    """

    is_end_to_end_depth = True

    def __init__(self, mdl_config: Dict[str, Any]):
        super().__init__()
        self.in_channels = int(mdl_config["input_channels"])
        use_bn = bool(mdl_config.get("use_batchnorm", False))
        encoder_lstm_type = str(mdl_config.get("encoder_lstm_type", "stand_convlstm"))

        lstm_cfg = mdl_config.get("encoder_lstm", {}) or {}
        if not isinstance(lstm_cfg, dict):
            lstm_cfg = {}
        self._three_t_max = mdl_config.get("T_max_chrono_init_encoder", [4, 8, 16])
        if isinstance(self._three_t_max, (list, tuple)) and len(self._three_t_max) >= 3:
            t_vals = tuple(self._three_t_max[:3])
        else:
            t_vals = (4, 8, 16)

        # --- Head ---
        bias = not use_bn
        self.head_conv = nn.Conv2d(
            self.in_channels, 32, kernel_size=5, stride=1, padding=2, bias=bias
        )
        self.head_bn = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        self.head_relu = _relu_inplace()

        # --- Encoders ---
        self.enc_conv0 = _Conv_relu(32, 64, 5, 2, 2, use_bn)
        self.lstm0 = _build_convlstm(encoder_lstm_type, 64, lstm_cfg, t_vals[0])
        self.enc_conv1 = _Conv_relu(64, 128, 5, 2, 2, use_bn)
        self.lstm1 = _build_convlstm(encoder_lstm_type, 128, lstm_cfg, t_vals[1])
        self.enc_conv2 = _Conv_relu(128, 256, 5, 2, 2, use_bn)
        self.lstm2 = _build_convlstm(encoder_lstm_type, 256, lstm_cfg, t_vals[2])

        # --- Bottleneck ---
        self.bot_res = nn.Sequential(_ResidualBlock256(use_bn), _ResidualBlock256(use_bn))

        # --- Decoders ---
        self.dec0_conv = _Conv_relu(256, 128, 5, 1, 2, use_bn)
        self.dec1_conv = _Conv_relu(128, 64, 5, 1, 2, use_bn)
        self.dec2_conv = _Conv_relu(64, 32, 5, 1, 2, use_bn)

        self.pred_conv = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def get_stage_dims(self, stages: Tuple[int, ...]):  # noqa: ARG002
        raise NotImplementedError("E2DepthConvLSTMUNet does not expose FPN stages")

    def get_strides(self, stages: Tuple[int, ...]):  # noqa: ARG002
        raise NotImplementedError("E2DepthConvLSTMUNet does not expose FPN stages")

    def forward(
        self,
        x: th.Tensor,
        prev_states: Optional[LstmStates] = None,
        token_mask: Optional[th.Tensor] = None,
    ):
        """
        Interface shim (unused by DepthEstimator in end-to-end mode).
        Keeps recurrent contract for tools that might call backbone.forward.
        """
        _, states = self.forward_depth_and_states(x, prev_states)
        return {}, states

    def forward_depth_and_states(
        self,
        x: th.Tensor,
        prev_states: Optional[LstmStates] = None,
    ) -> Tuple[Dict[str, th.Tensor], LstmStates]:
        if prev_states is None:
            prev_states_list: List[Any] = [None, None, None]
        else:
            prev_states_list = list(prev_states)
            while len(prev_states_list) < 3:
                prev_states_list.append(None)

        ho = self.head_conv(x)
        ho = self.head_bn(ho)
        head_feat = self.head_relu(ho)

        e0 = self.enc_conv0(head_feat)
        s0_tuple = prev_states_list[0]
        h0_tuple = (
            None
            if s0_tuple is None or len(s0_tuple) < 2
            else (s0_tuple[0], s0_tuple[1])
        )
        # lstm* 返回 (h_t, c_t)；须一并存入 state，否则会丢掉 h 并在下一步把 Tensor 误当 tuple 拆成 3D slice
        e0_out, c0 = self.lstm0(e0, h0_tuple)
        state0 = (e0_out, c0)

        e1 = self.enc_conv1(e0_out)
        s1 = prev_states_list[1]
        h1_tuple = None if s1 is None or len(s1) < 2 else (s1[0], s1[1])
        e1_out, c1 = self.lstm1(e1, h1_tuple)
        state1 = (e1_out, c1)

        e2 = self.enc_conv2(e1_out)
        s2 = prev_states_list[2]
        h2_tuple = None if s2 is None or len(s2) < 2 else (s2[0], s2[1])
        e2_out, c2 = self.lstm2(e2, h2_tuple)
        state2 = (e2_out, c2)

        b = self.bot_res(e2_out)

        d0_in = b + e2_out
        d0 = F.interpolate(d0_in, scale_factor=2, mode="bilinear", align_corners=False)
        d0 = self.dec0_conv(d0)
        d0 = d0 + e1_out

        d1 = F.interpolate(d0, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1_conv(d1)
        d1 = d1 + e0_out

        d2 = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2_conv(d2)
        d2 = d2 + head_feat

        out_logits = self.pred_conv(d2)
        depth_1 = th.sigmoid(out_logits)

        predictions = {"depth_1": depth_1}
        new_states = (state0, state1, state2)
        return predictions, new_states
