"""
Microbenchmark: per-cell forward time for the 3 encoder stages.

Compares StandardConvLSTM2d (baseline) against DWSConvSTLSTM2d (full / various
ablations) so we can attribute the slowdown to specific components.

Run: python scripts/bench_lstm_cells.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch as th

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.backbone.rnn import (  # noqa: E402
    DWSConvSTLSTM2d,
    StandardConvLSTM2d,
)


def _sync():
    if th.cuda.is_available():
        th.cuda.synchronize()


def time_cell(cell, x, h_c, m=None, iters=200, warmup=50, want_m=False):
    cell.eval()
    with th.no_grad():
        for _ in range(warmup):
            if want_m:
                cell(x, h_c, m)
            else:
                cell(x, h_c)
    _sync()
    t0 = time.perf_counter()
    with th.no_grad():
        for _ in range(iters):
            if want_m:
                cell(x, h_c, m)
            else:
                cell(x, h_c)
    _sync()
    return (time.perf_counter() - t0) * 1000 / iters  # ms / call


def make_inputs(dim, h, w, device, dtype, with_m=False):
    x = th.randn(1, dim, h, w, device=device, dtype=dtype)
    h_c = (th.randn_like(x), th.randn_like(x))
    m = th.randn_like(x) if with_m else None
    return x, h_c, m


def main():
    device = "cuda" if th.cuda.is_available() else "cpu"
    dtype = th.float16 if device == "cuda" else th.float32  # match 16-mixed training
    print(f"device={device} dtype={dtype}\n")

    # Encoder stages: (dim, H, W) for 256x344 input
    stages = [(64, 128, 172), (128, 64, 86), (256, 32, 43)]

    configs = [
        ("baseline_ConvLSTM",
         lambda dim: StandardConvLSTM2d(dim=dim).to(device).to(dtype),
         False),
        ("STLSTM_full (k5+dws_m+GN)",
         lambda dim: DWSConvSTLSTM2d(dim=dim, dws_conv=True, dws_conv_only_hidden=True,
                                     dws_conv_kernel_size=5, dws_on_spatial_m=True,
                                     use_group_norm=True).to(device).to(dtype),
         True),
        ("STLSTM_no_GN (k5+dws_m, GN off)",
         lambda dim: DWSConvSTLSTM2d(dim=dim, dws_conv=True, dws_conv_only_hidden=True,
                                     dws_conv_kernel_size=5, dws_on_spatial_m=True,
                                     use_group_norm=False).to(device).to(dtype),
         True),
        ("STLSTM_no_dws_m (k5, no_dws_m, GN on)",
         lambda dim: DWSConvSTLSTM2d(dim=dim, dws_conv=True, dws_conv_only_hidden=True,
                                     dws_conv_kernel_size=5, dws_on_spatial_m=False,
                                     use_group_norm=True).to(device).to(dtype),
         True),
        ("STLSTM_k3 (k3+dws_m+GN)",
         lambda dim: DWSConvSTLSTM2d(dim=dim, dws_conv=True, dws_conv_only_hidden=True,
                                     dws_conv_kernel_size=3, dws_on_spatial_m=True,
                                     use_group_norm=True).to(device).to(dtype),
         True),
        ("STLSTM_lean (k3, no_dws_m, no_GN)",
         lambda dim: DWSConvSTLSTM2d(dim=dim, dws_conv=True, dws_conv_only_hidden=True,
                                     dws_conv_kernel_size=3, dws_on_spatial_m=False,
                                     use_group_norm=False).to(device).to(dtype),
         True),
    ]

    rows = []
    for name, builder, with_m in configs:
        per_stage = []
        for dim, h, w in stages:
            cell = builder(dim)
            x, hc, m = make_inputs(dim, h, w, device, dtype, with_m=with_m)
            t = time_cell(cell, x, hc, m=m, want_m=with_m)
            per_stage.append(t)
            del cell
            th.cuda.empty_cache() if device == "cuda" else None
        rows.append((name, per_stage))

    # Print table
    hdr = f"{'config':40s}  " + "  ".join(f"d={d:3d}" for d, _, _ in stages) + "   3-stage sum"
    print(hdr)
    print("-" * len(hdr))
    for name, per_stage in rows:
        s = sum(per_stage)
        cells = "  ".join(f"{t:5.3f}" for t in per_stage)
        print(f"{name:40s}  {cells}   {s:6.3f} ms")

    base_sum = sum(rows[0][1])
    print()
    print(f"Relative to baseline (3-stage sum = {base_sum:.3f} ms):")
    for name, per_stage in rows[1:]:
        rel = sum(per_stage) / base_sum
        print(f"  {name:40s}  {rel:.2f}x")


if __name__ == "__main__":
    main()
