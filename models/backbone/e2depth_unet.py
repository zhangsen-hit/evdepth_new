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
from models.backbone.rnn import (
    DWSConvLSTM2d,
    DWSConvSTLSTM2d,
    FusedDWSConvSTLSTM2d,
    StandardConvLSTM2d,
    _gn_num_groups,
)


def _relu_inplace() -> nn.ReLU:
    return nn.ReLU(inplace=True)


def _build_convlstm(
    encoder_lstm_type: str,
    dim: int,
    lstm_cfg: Dict[str, Any],
    T_max_chrono_init: Optional[int],
    stage_use_norm: bool = True,
) -> nn.Module:
    """`stage_use_norm=False` forces this stage's ST-LSTM cell to skip all
    norms (per-stage GN ablation), regardless of cell-level cfg."""
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
    if t == "stlstm":
        use_gn = bool(lstm_cfg.get("use_group_norm", True)) and stage_use_norm
        return DWSConvSTLSTM2d(
            dim=dim,
            dws_conv=lstm_cfg.get("dws_conv", True),
            dws_conv_only_hidden=lstm_cfg.get("dws_conv_only_hidden", True),
            dws_conv_kernel_size=lstm_cfg.get("dws_conv_kernel_size", 5),
            dws_on_spatial_m=lstm_cfg.get("dws_on_spatial_m", True),
            cell_update_dropout=drop,
            T_max_chrono_init=T_max_chrono_init,
            use_group_norm=use_gn,
            norm_type=lstm_cfg.get("norm_type", "gn"),
            norm_components=lstm_cfg.get("norm_components", None),
        )
    if t == "fused_stlstm":
        use_gn = bool(lstm_cfg.get("use_group_norm", True)) and stage_use_norm
        return FusedDWSConvSTLSTM2d(
            dim=dim,
            dws_conv=lstm_cfg.get("dws_conv", True),
            dws_conv_only_hidden=lstm_cfg.get("dws_conv_only_hidden", True),
            dws_conv_kernel_size=lstm_cfg.get("dws_conv_kernel_size", 5),
            dws_on_spatial_m=lstm_cfg.get("dws_on_spatial_m", True),
            cell_update_dropout=drop,
            T_max_chrono_init=T_max_chrono_init,
            use_group_norm=use_gn,
            norm_type=lstm_cfg.get("norm_type", "gn"),
            norm_components=lstm_cfg.get("norm_components", None),
        )
    raise ValueError(
        f"encoder_lstm_type must be 'stand_convlstm', 'dws_convlstm', 'stlstm', or 'fused_stlstm', got {t!r}"
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


class _LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW (ConvNeXt convention).

    Permutes to NHWC, runs nn.LayerNorm over the channel dim per spatial
    location, permutes back. Affine scale + bias.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _DSConv(nn.Module):
    """Depthwise + pointwise replacement for `_Conv_relu`.

    DW (k×k, stride=s, groups=in_ch) → LN → ReLU6 → PW (1×1, in_ch→out_ch) → LN → ReLU6
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int, padding: int):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel, stride=stride,
            padding=padding, groups=in_ch, bias=False,
        )
        self.norm1 = _LayerNorm2d(in_ch)
        self.act1 = nn.ReLU6(inplace=True)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm2 = _LayerNorm2d(out_ch)
        self.act2 = nn.ReLU6(inplace=True)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.act1(self.norm1(self.dw(x)))
        x = self.act2(self.norm2(self.pw(x)))
        return x


class _InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual block with LN + ReLU6.

    1×1 expand (C → t·C) → LN → ReLU6
    3×3 dw   (groups=t·C) → LN → ReLU6
    1×1 project (t·C → C) → LN              ← linear bottleneck (no activation)
    + residual

    For our bottleneck (256→256, stride=1) the residual is always active.
    """

    def __init__(self, channels: int, expand_ratio: int = 4):
        super().__init__()
        hidden = channels * expand_ratio
        self.expand = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.norm1 = _LayerNorm2d(hidden)
        self.act1 = nn.ReLU6(inplace=True)
        self.dw = nn.Conv2d(
            hidden, hidden, kernel_size=3, stride=1, padding=1,
            groups=hidden, bias=False,
        )
        self.norm2 = _LayerNorm2d(hidden)
        self.act2 = nn.ReLU6(inplace=True)
        self.project = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.norm3 = _LayerNorm2d(channels)

    def forward(self, x: th.Tensor) -> th.Tensor:
        identity = x
        y = self.act1(self.norm1(self.expand(x)))
        y = self.act2(self.norm2(self.dw(y)))
        y = self.norm3(self.project(y))
        return identity + y


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

        # --- Lightweight block flags (IR + DSConv phase) ---
        # Defaults preserve current architecture (regular convs + ResBlock256).
        use_dsconv_encoder = bool(mdl_config.get("use_dsconv_encoder", False))
        use_dsconv_decoder = bool(mdl_config.get("use_dsconv_decoder", False))
        bottleneck_block = str(mdl_config.get("bottleneck_block", "residual"))
        ir_expand_ratio = int(mdl_config.get("ir_expand_ratio", 4))
        if bottleneck_block not in ("residual", "ir"):
            raise ValueError(
                f"bottleneck_block must be 'residual' or 'ir', got {bottleneck_block!r}"
            )

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

        # Per-stage GN override (ablation 3/4): default all True
        gn_stages_cfg = lstm_cfg.get("gn_stages", [True, True, True])
        if not (
            isinstance(gn_stages_cfg, (list, tuple)) and len(gn_stages_cfg) == 3
        ):
            raise ValueError(
                f"encoder_lstm.gn_stages must be a length-3 bool list, got {gn_stages_cfg!r}"
            )
        gn_stages = [bool(v) for v in gn_stages_cfg]

        # --- Encoders ---
        def _enc(in_ch: int, out_ch: int) -> nn.Module:
            if use_dsconv_encoder:
                return _DSConv(in_ch, out_ch, kernel=5, stride=2, padding=2)
            return _Conv_relu(in_ch, out_ch, 5, 2, 2, use_bn)

        self.enc_conv0 = _enc(32, 64)
        self.lstm0 = _build_convlstm(encoder_lstm_type, 64, lstm_cfg, t_vals[0], stage_use_norm=gn_stages[0])
        self.enc_conv1 = _enc(64, 128)
        self.lstm1 = _build_convlstm(encoder_lstm_type, 128, lstm_cfg, t_vals[1], stage_use_norm=gn_stages[1])
        self.enc_conv2 = _enc(128, 256)
        self.lstm2 = _build_convlstm(encoder_lstm_type, 256, lstm_cfg, t_vals[2], stage_use_norm=gn_stages[2])

        # --- ST-LSTM zigzag wiring (PredRNN/PredRNN++ style) ---
        self.cell_type = encoder_lstm_type
        self.use_zigzag = (
            encoder_lstm_type in ("stlstm", "fused_stlstm")
            and bool(lstm_cfg.get("zigzag", False))
        )
        # Adapter-level GN switch (ablation 5). Controls only the zigzag adapter
        # GroupNorms — independent of cell-level GN.
        adapter_use_norm = bool(lstm_cfg.get("zigzag_adapter_norm", True))

        def _adapter_gn(c: int) -> nn.Module:
            return nn.GroupNorm(_gn_num_groups(c), c) if adapter_use_norm else nn.Identity()

        self._encoder_dims = (64, 128, 256)
        if self.use_zigzag:
            d0, d1, d2 = self._encoder_dims
            # Forward adapters: stage l M -> stage l+1 M (downsample 2x, expand channels)
            self.m_adapters_forward = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(d0, d1, kernel_size=1, bias=False),
                        _adapter_gn(d1),
                        nn.SiLU(inplace=True),
                    ),
                    nn.Sequential(
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(d1, d2, kernel_size=1, bias=False),
                        _adapter_gn(d2),
                        nn.SiLU(inplace=True),
                    ),
                ]
            )
            # Zigzag: last stage (256, /8) at t-1 -> first stage (64, /2) at t
            # 1x1 project channels, then bilinear upsample (in forward), then 5x5 dws refine.
            self.m_adapter_zigzag = nn.Sequential(
                nn.Conv2d(d2, d0, kernel_size=1, bias=False),
                _adapter_gn(d0),
                nn.SiLU(inplace=True),
            )
            self.m_adapter_zigzag_refine = nn.Sequential(
                nn.Conv2d(d0, d0, kernel_size=5, padding=2, groups=d0, bias=False),
                _adapter_gn(d0),
                nn.SiLU(inplace=True),
                nn.Conv2d(d0, d0, kernel_size=1, bias=False),
            )

        # --- Bottleneck ---
        if bottleneck_block == "ir":
            self.bot_res = nn.Sequential(
                _InvertedResidual(256, expand_ratio=ir_expand_ratio),
                _InvertedResidual(256, expand_ratio=ir_expand_ratio),
            )
        else:
            self.bot_res = nn.Sequential(_ResidualBlock256(use_bn), _ResidualBlock256(use_bn))

        # --- Decoders ---
        def _dec(in_ch: int, out_ch: int) -> nn.Module:
            if use_dsconv_decoder:
                return _DSConv(in_ch, out_ch, kernel=5, stride=1, padding=2)
            return _Conv_relu(in_ch, out_ch, 5, 1, 2, use_bn)

        self.dec0_conv = _dec(256, 128)
        self.dec1_conv = _dec(128, 64)
        self.dec2_conv = _dec(64, 32)

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

        s0_tuple = prev_states_list[0]
        s1_tuple = prev_states_list[1]
        s2_tuple = prev_states_list[2]
        h0_tuple = (
            None
            if s0_tuple is None or len(s0_tuple) < 2
            else (s0_tuple[0], s0_tuple[1])
        )
        h1_tuple = (
            None
            if s1_tuple is None or len(s1_tuple) < 2
            else (s1_tuple[0], s1_tuple[1])
        )
        h2_tuple = (
            None
            if s2_tuple is None or len(s2_tuple) < 2
            else (s2_tuple[0], s2_tuple[1])
        )

        is_stlstm = self.cell_type in ("stlstm", "fused_stlstm")

        if is_stlstm:
            # Determine M fed into stage 0 — from zigzag link or stage-0's own t-1 M.
            if self.use_zigzag:
                m_for_lstm0: Optional[th.Tensor] = None
                if s2_tuple is not None and len(s2_tuple) >= 3:
                    m_last = self.m_adapter_zigzag(s2_tuple[2])
                    e0_h = head_feat.shape[2] // 2
                    e0_w = head_feat.shape[3] // 2
                    m_last = F.interpolate(
                        m_last,
                        size=(e0_h, e0_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    m_for_lstm0 = self.m_adapter_zigzag_refine(m_last)
            else:
                m_for_lstm0 = (
                    s0_tuple[2]
                    if s0_tuple is not None and len(s0_tuple) >= 3
                    else None
                )

            e0 = self.enc_conv0(head_feat)
            e0_out, c0, m0 = self.lstm0(e0, h0_tuple, m_for_lstm0)
            state0 = (e0_out, c0, m0)

            m_for_lstm1 = (
                self.m_adapters_forward[0](m0)
                if self.use_zigzag
                else (
                    s1_tuple[2]
                    if s1_tuple is not None and len(s1_tuple) >= 3
                    else None
                )
            )
            e1 = self.enc_conv1(e0_out)
            e1_out, c1, m1 = self.lstm1(e1, h1_tuple, m_for_lstm1)
            state1 = (e1_out, c1, m1)

            m_for_lstm2 = (
                self.m_adapters_forward[1](m1)
                if self.use_zigzag
                else (
                    s2_tuple[2]
                    if s2_tuple is not None and len(s2_tuple) >= 3
                    else None
                )
            )
            e2 = self.enc_conv2(e1_out)
            e2_out, c2, m2 = self.lstm2(e2, h2_tuple, m_for_lstm2)
            state2 = (e2_out, c2, m2)
        else:
            # ConvLSTM path (legacy). lstm* 返回 (h_t, c_t)；
            # 须一并存入 state，否则会丢掉 h 并在下一步把 Tensor 误当 tuple 拆成 3D slice。
            e0 = self.enc_conv0(head_feat)
            e0_out, c0 = self.lstm0(e0, h0_tuple)
            state0 = (e0_out, c0)

            e1 = self.enc_conv1(e0_out)
            e1_out, c1 = self.lstm1(e1, h1_tuple)
            state1 = (e1_out, c1)

            e2 = self.enc_conv2(e1_out)
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
