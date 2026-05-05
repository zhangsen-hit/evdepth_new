"""
Smoke test for E2DepthConvLSTMUNet with the ST-LSTM + zigzag changes.

Runs 3 timesteps in three configurations:
  1. stand_convlstm   — regression check; legacy state shape (h, c)
  2. stlstm w/o zigzag — per-stage M persists temporally
  3. stlstm w/ zigzag  — last-stage M(t-1) -> first-stage M(t)

For each, assert output shape, per-stage state structure (len 2 or 3 with the
correct (C, H, W)), no NaN in forward/backward, and that grads exist on the
new zigzag adapter params.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch as th

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.backbone.e2depth_unet import E2DepthConvLSTMUNet  # noqa: E402


def _build(cell_type: str, zigzag: bool) -> E2DepthConvLSTMUNet:
    cfg = {
        "input_channels": 2,
        "in_res_hw": [256, 344],
        "use_batchnorm": False,
        "encoder_lstm_type": cell_type,
        "T_max_chrono_init_encoder": [4, 8, 16],
        "encoder_lstm": {
            "dws_conv": True,
            "dws_conv_only_hidden": True,
            "dws_conv_kernel_size": 3 if cell_type != "stlstm" else 5,
            "drop_cell_update": 0.0,
            "zigzag": zigzag,
            "dws_on_spatial_m": True,
            "use_group_norm": True,
        },
    }
    return E2DepthConvLSTMUNet(cfg)


def _expected_state_shapes(B: int, H: int, W: int):
    # encoder dims: (64, 128, 256), spatial: /2, /4, /8
    return [
        (64, H // 2, W // 2),
        (128, H // 4, W // 4),
        (256, H // 8, W // 8),
    ]


def _check_state(states, B: int, H: int, W: int, expect_len: int, tag: str):
    assert isinstance(states, tuple) and len(states) == 3, f"[{tag}] states must be 3-tuple"
    expected = _expected_state_shapes(B, H, W)
    for stage_idx, (state, (C, eh, ew)) in enumerate(zip(states, expected)):
        assert isinstance(state, tuple), f"[{tag}] stage {stage_idx} not a tuple"
        assert len(state) == expect_len, (
            f"[{tag}] stage {stage_idx} expected len {expect_len}, got {len(state)}"
        )
        for k, t in enumerate(state):
            assert t.shape == (B, C, eh, ew), (
                f"[{tag}] stage {stage_idx} elem {k} shape {tuple(t.shape)} != "
                f"({B}, {C}, {eh}, {ew})"
            )
            assert th.isfinite(t).all(), f"[{tag}] stage {stage_idx} elem {k} non-finite"


def _run_case(cell_type: str, zigzag: bool, device: str = "cpu"):
    tag = f"{cell_type}{'+zigzag' if zigzag else ''}"
    print(f"\n=== {tag} ===")
    th.manual_seed(0)
    net = _build(cell_type, zigzag).to(device).train()

    B, C, H, W = 2, 2, 256, 344
    expect_len = 3 if cell_type == "stlstm" else 2

    states = None
    losses = []
    for t in range(3):
        x = th.randn(B, C, H, W, device=device)
        preds, states = net.forward_depth_and_states(x, states)
        depth = preds["depth_1"]
        assert depth.shape == (B, 1, H, W), f"[{tag}] depth shape {tuple(depth.shape)}"
        assert th.isfinite(depth).all(), f"[{tag}] depth non-finite at t={t}"
        _check_state(states, B, H, W, expect_len, tag)
        losses.append(depth.mean())
        print(f"  t={t}: depth ok, state ok ({expect_len}-tuple per stage)")

    total = sum(losses)
    total.backward()

    n_params_with_grad = 0
    n_nan_grad = 0
    for n, p in net.named_parameters():
        if p.grad is not None:
            n_params_with_grad += 1
            if not th.isfinite(p.grad).all():
                n_nan_grad += 1
                print(f"  !! NaN/Inf grad in {n}")
    print(f"  params with grad: {n_params_with_grad}, NaN/Inf grad params: {n_nan_grad}")
    assert n_nan_grad == 0, f"[{tag}] NaN/Inf in gradients"

    if zigzag:
        # Confirm zigzag adapter modules received gradient
        adapter_names = [
            "m_adapter_zigzag.0.weight",
            "m_adapter_zigzag_refine.0.weight",
            "m_adapters_forward.0.1.weight",
            "m_adapters_forward.1.1.weight",
        ]
        for name in adapter_names:
            p = dict(net.named_parameters())[name]
            assert p.grad is not None, f"[{tag}] no grad on {name}"
            assert p.grad.abs().sum().item() > 0, (
                f"[{tag}] zero grad on {name} — adapter unused?"
            )
        print("  zigzag adapter grads non-zero ✓")

    n_total = sum(p.numel() for p in net.parameters())
    print(f"  total params: {n_total/1e6:.2f}M")


def main():
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"device: {device}")
    for cell_type, zigzag in [
        ("stand_convlstm", False),
        ("stlstm", False),
        ("stlstm", True),
    ]:
        _run_case(cell_type, zigzag, device=device)
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
