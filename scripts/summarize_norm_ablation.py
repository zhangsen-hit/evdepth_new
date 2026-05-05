"""
Read snapshots in local_loss_runs/<run>_3k/ and emit one comparison table
covering: baseline ConvLSTM, ST-LSTM full GN, ST-LSTM no GN, plus the 7
norm-ablation variants. Reports val_RMSE/δ1, val_loss, final train_loss/δ1
and per-step E2DepthUNet timing.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "local_loss_runs"

# Order matters for readability of the printed table.
ENTRIES = [
    ("baseline_3k",         "baseline (ConvLSTM)"),
    ("stlstm_zigzag_3k",    "ST-LSTM full GN  (ref)"),
    ("stlstm_nogn_3k",      "ST-LSTM no GN"),
    ("v1_m_only_3k",        "v1 only spatial+memory"),
    ("v2_gates_only_3k",    "v2 only temporal+spatial"),
    ("v3_no_stage0_3k",     "v3 stage1+2 GN, no s0"),
    ("v4_only_stage0_3k",   "v4 only stage0 GN"),
    ("v5_no_adapter_3k",    "v5 cell GN, adapters off"),
    ("v6_ln_3k",            "v6 LN (groups=1)"),
    ("v7_bn_3k",            "v7 BN"),
    ("v8_m_only_no_s0_3k",  "v8 v1+v3 stacked"),
    ("v9_rms_3k",           "v9 RMSNorm"),
    ("v10_in_3k",           "v10 InstanceNorm"),
    ("v11_in_m_only_no_s0_no_adapter_3k", "v11 IN+M-only+no_s0+no_ad"),
    ("v12_full_lightweight_3k",           "v12 v11 + DSConv + IR(t=4)"),
    ("v11b_chrono_fix_3k",                "v11b v11 + chrono fix"),
    ("v13_fused_cell_3k",                 "v13 v11 + fused cell"),
]


def _last(path: Path) -> float | None:
    if not path.exists():
        return None
    last = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        last = float(line.split("\t")[-1])
    return last


def _timing_mean(path: Path) -> float | None:
    """Pull the mean E2DepthUNet ms across the captured 'Timing statistics' lines."""
    if not path.exists():
        return None
    vals = []
    for line in path.read_text().splitlines():
        if "E2DepthUNet" not in line:
            continue
        # format: "E2DepthUNet: mean=8.82 ms, median=8.77 ms"
        try:
            mean_part = line.split("mean=")[1]
            ms = float(mean_part.split("ms")[0].strip())
            vals.append(ms)
        except Exception:
            pass
    return sum(vals) / len(vals) if vals else None


def main():
    rows = []
    for tag, label in ENTRIES:
        d = RUNS / tag
        rows.append({
            "label": label,
            "train_loss": _last(d / "train_loss.txt"),
            "train_d1":   _last(d / "train_delta1.txt"),
            "val_loss":   _last(d / "val_loss.txt"),
            "val_rmse":   _last(d / "val_rmse.txt"),
            "val_d1":     _last(d / "val_delta1.txt"),
            "fwd_ms":     _timing_mean(d / "timing.txt"),
        })

    def fmt(v, w, prec):
        if v is None:
            return " " * w + "  -"
        return f"{v:>{w}.{prec}f}"

    hdr = (
        f"{'config':28s}  "
        f"{'train_loss':>10s}  {'train_d1':>9s}  "
        f"{'val_loss':>9s}  {'val_RMSE':>9s}  {'val_d1':>7s}  "
        f"{'fwd_ms':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['label']:28s}  "
            f"{fmt(r['train_loss'],10,4)}  {fmt(r['train_d1'],9,4)}  "
            f"{fmt(r['val_loss'],9,4)}  {fmt(r['val_rmse'],9,3)}  "
            f"{fmt(r['val_d1'],7,4)}  {fmt(r['fwd_ms'],7,2)}"
        )

    base = next(r for r in rows if r["label"].startswith("baseline"))
    ref = next(r for r in rows if "full GN" in r["label"])
    print()
    print("--- relative to baseline (ConvLSTM) ---")
    print(f"{'config':28s}  {'val_RMSE Δ%':>13s}  {'val_d1 Δ%':>11s}  {'fwd_ms ratio':>13s}")
    for r in rows:
        if r["val_rmse"] is None:
            continue
        rmse_pct = (r["val_rmse"] - base["val_rmse"]) / base["val_rmse"] * 100
        d1_pct = (r["val_d1"] - base["val_d1"]) / base["val_d1"] * 100 if r["val_d1"] else None
        ratio = r["fwd_ms"] / base["fwd_ms"] if r["fwd_ms"] and base["fwd_ms"] else None
        print(
            f"{r['label']:28s}  "
            f"{rmse_pct:>+12.2f}%  "
            f"{(f'{d1_pct:>+10.2f}%' if d1_pct is not None else '       -'):>11s}  "
            f"{(f'{ratio:>11.2f}x' if ratio is not None else '         -'):>13s}"
        )

    # vs full-GN reference: who keeps RMSE within +0.10m? who is fastest among them?
    print()
    print("--- relative to ST-LSTM full GN reference ---")
    print(f"{'config':28s}  {'ΔRMSE (m)':>10s}  {'Δd1':>8s}  {'fwd_ms saving':>14s}")
    for r in rows:
        if r["label"].startswith("baseline") or r["val_rmse"] is None:
            continue
        d_rmse = r["val_rmse"] - ref["val_rmse"]
        d_d1 = r["val_d1"] - ref["val_d1"]
        d_ms = (ref["fwd_ms"] - r["fwd_ms"]) if r["fwd_ms"] and ref["fwd_ms"] else None
        print(
            f"{r['label']:28s}  "
            f"{d_rmse:>+9.3f}  "
            f"{d_d1:>+7.4f}  "
            f"{(f'{d_ms:>+11.2f} ms' if d_ms is not None else '          -'):>14s}"
        )


if __name__ == "__main__":
    main()
