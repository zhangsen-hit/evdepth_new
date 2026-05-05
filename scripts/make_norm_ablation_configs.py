"""
Generate 7 variant configs by patching encoder_lstm fields of the existing
config_liosam_stlstm_short.yaml. Output files: config_liosam_stlstm_<tag>.yaml.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "config_liosam_stlstm_short.yaml"

VARIANTS = {
    # tag           encoder_lstm patch
    "v1_m_only":      {"norm_components": ["spatial", "memory"]},
    "v2_gates_only":  {"norm_components": ["temporal", "spatial"]},
    "v3_no_stage0":   {"gn_stages": [False, True, True]},
    "v4_only_stage0": {"gn_stages": [True, False, False]},
    "v5_no_adapter":  {"zigzag_adapter_norm": False},
    "v6_ln":          {"norm_type": "ln"},
    "v7_bn":          {"norm_type": "bn"},
    # v8: combine v1's "M-branch focus" with v3's "skip stage0" — both findings stacked.
    "v8_m_only_no_s0": {
        "norm_components": ["spatial", "memory"],
        "gn_stages": [False, True, True],
    },
    "v9_rms":         {"norm_type": "rms"},
    "v10_in":         {"norm_type": "in"},
    # v11: stack the three independently-validated wins —
    #   v10 (InstanceNorm) + v8 (M-branch only, skip stage0) + v5 (adapters off).
    "v11_in_m_only_no_s0_no_adapter": {
        "norm_type": "in",
        "norm_components": ["spatial", "memory"],
        "gn_stages": [False, True, True],
        "zigzag_adapter_norm": False,
    },
}


def main():
    with open(BASE) as f:
        base = yaml.safe_load(f)
    for tag, patch in VARIANTS.items():
        cfg = yaml.safe_load(yaml.safe_dump(base))  # deep-copy
        cfg["model"]["backbone"]["encoder_lstm"].update(patch)
        out = REPO / f"config_liosam_stlstm_{tag}.yaml"
        with open(out, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"  wrote {out.name}: encoder_lstm += {patch}")


if __name__ == "__main__":
    main()
