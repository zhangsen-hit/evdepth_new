"""
Smoke-test the 7 GN ablation variants of E2DepthConvLSTMUNet (ST-LSTM + zigzag).
For each variant we build the model, run 2 timesteps, verify shapes & no NaN.
Uses a tiny B/H/W for speed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch as th

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.backbone.e2depth_unet import E2DepthConvLSTMUNet  # noqa: E402


VARIANTS = {
    "ref_full_gn": {  # current GN-on baseline for sanity
        "use_group_norm": True,
        "norm_type": "gn",
    },
    "v1_m_only":     {"norm_components": ["spatial", "memory"]},
    "v2_gates_only": {"norm_components": ["temporal", "spatial"]},
    "v3_no_stage0":  {"gn_stages": [False, True, True]},
    "v4_only_stage0":{"gn_stages": [True, False, False]},
    "v5_no_adapter": {"zigzag_adapter_norm": False},  # cell GN still on
    "v6_ln":         {"norm_type": "ln"},
    "v7_bn":         {"norm_type": "bn"},
    "v8_m_only_no_s0": {
        "norm_components": ["spatial", "memory"],
        "gn_stages": [False, True, True],
    },
    "v9_rms":        {"norm_type": "rms"},
    "v10_in":        {"norm_type": "in"},
    "v11_in_m_only_no_s0_no_adapter": {
        "norm_type": "in",
        "norm_components": ["spatial", "memory"],
        "gn_stages": [False, True, True],
        "zigzag_adapter_norm": False,
    },
    # v12 = v11 + DSConv encoder + DSConv decoder + IR(t=4) bottleneck.
    "v12_full_lightweight": {
        "norm_type": "in",
        "norm_components": ["spatial", "memory"],
        "gn_stages": [False, True, True],
        "zigzag_adapter_norm": False,
        "use_dsconv_encoder": True,
        "use_dsconv_decoder": True,
        "bottleneck_block": "ir",
        "ir_expand_ratio": 4,
    },
    # v13 = v11 with the fused ST-LSTM cell.
    "v13_fused_cell": {
        "encoder_lstm_type": "fused_stlstm",
        "norm_type": "in",
        "norm_components": ["spatial", "memory"],
        "gn_stages": [False, True, True],
        "zigzag_adapter_norm": False,
    },
}


def _build(extra: dict) -> E2DepthConvLSTMUNet:
    # Allow `extra` to patch either encoder_lstm or top-level backbone fields:
    # keys in TOP_LEVEL_KEYS go to mdl_config, others go to encoder_lstm.
    TOP_LEVEL_KEYS = {
        "encoder_lstm_type",
        "use_dsconv_encoder", "use_dsconv_decoder",
        "bottleneck_block", "ir_expand_ratio",
    }
    top_extra = {k: v for k, v in extra.items() if k in TOP_LEVEL_KEYS}
    cell_extra = {k: v for k, v in extra.items() if k not in TOP_LEVEL_KEYS}

    cfg = {
        "input_channels": 2,
        "in_res_hw": [64, 80],
        "use_batchnorm": False,
        "encoder_lstm_type": "stlstm",
        "T_max_chrono_init_encoder": [4, 8, 16],
        "encoder_lstm": {
            "dws_conv": True,
            "dws_conv_only_hidden": True,
            "dws_conv_kernel_size": 5,
            "drop_cell_update": 0.0,
            "zigzag": True,
            "dws_on_spatial_m": True,
            "use_group_norm": True,
            "norm_type": "gn",
            "gn_stages": [True, True, True],
            "zigzag_adapter_norm": True,
            **cell_extra,
        },
        **top_extra,
    }
    return E2DepthConvLSTMUNet(cfg)


def main():
    device = "cuda" if th.cuda.is_available() else "cpu"
    B, C, H, W = 2, 2, 64, 80
    print(f"device={device}, input shape ({B},{C},{H},{W})\n")
    for name, extra in VARIANTS.items():
        th.manual_seed(0)
        net = _build(extra).to(device).train()
        states = None
        try:
            for t in range(2):
                x = th.randn(B, C, H, W, device=device)
                preds, states = net.forward_depth_and_states(x, states)
                d = preds["depth_1"]
                assert d.shape == (B, 1, H, W), f"shape {tuple(d.shape)}"
                assert th.isfinite(d).all(), "non-finite"
            loss = preds["depth_1"].mean()
            loss.backward()
            params = sum(p.numel() for p in net.parameters())
            print(f"  {name:18s} OK   params={params/1e6:5.2f}M")
        except Exception as e:
            print(f"  {name:18s} FAIL  {type(e).__name__}: {e}")
            raise
    print("\nALL VARIANT SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
