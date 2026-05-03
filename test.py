#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估脚本：基于 run_depth.py 保存的预测 .npy（米，256x344），
对照测试场景 npz 中的 GT，逐帧计算 8 项深度指标后取均值并输出。

指标：Abs Rel、Sq Rel、RMSE、RMSE log、SI log、delta<1.25、delta<1.25^2、delta<1.25^3。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.genx_utils.liosam_sequence import (
    DEFAULT_DEPTH_KEY,
    DEFAULT_DEPTH_MASK_KEY,
    center_crop_tensor_2d,
    load_liosam_index,
)


def _load_gt_and_mask(
    npz_path: Path,
    depth_key: str,
    depth_mask_key: Optional[str],
    center_crop_hw: Tuple[int, int],
    min_depth: float,
    max_depth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """加载 GT 深度（米）与有效 mask；与训练 / 验证一致地 center_crop 到 (256,344)。"""
    data = np.load(str(npz_path), allow_pickle=True)
    depth_np = np.asarray(data[depth_key], dtype=np.float32)
    finite = np.isfinite(depth_np)
    depth_safe = np.where(finite, np.clip(depth_np, min_depth, max_depth), max_depth)

    depth_t = torch.from_numpy(depth_safe).float()
    if depth_t.dim() == 2:
        depth_t = depth_t.unsqueeze(0)
    ch, cw = center_crop_hw
    depth_t = center_crop_tensor_2d(depth_t, ch, cw)

    if depth_mask_key and depth_mask_key in data.files:
        mask_t = torch.from_numpy(np.asarray(data[depth_mask_key], dtype=bool))
        if mask_t.dim() == 2:
            mask_t = mask_t.unsqueeze(0)
    else:
        mask_t = (
            torch.from_numpy(finite).bool()
            & (torch.from_numpy(depth_safe) > min_depth)
            & (torch.from_numpy(depth_safe) < max_depth)
        )
        if mask_t.dim() == 2:
            mask_t = mask_t.unsqueeze(0)
    mask_t = center_crop_tensor_2d(mask_t, ch, cw)

    return depth_t[0].cpu().numpy(), mask_t[0].cpu().numpy()


def _compute_metrics(pred_m: np.ndarray, gt_m: np.ndarray, mask: np.ndarray) -> Optional[Dict[str, float]]:
    """对单帧的有效像素计算 8 项指标。若有效像素为 0 则返回 None。"""
    valid = mask & np.isfinite(pred_m) & np.isfinite(gt_m) & (pred_m > 0) & (gt_m > 0)
    if not valid.any():
        return None

    p = pred_m[valid].astype(np.float64)
    g = gt_m[valid].astype(np.float64)
    p = np.clip(p, 1e-3, None)
    g = np.clip(g, 1e-3, None)

    diff = p - g
    abs_rel = float(np.mean(np.abs(diff) / g))
    sq_rel = float(np.mean(diff * diff / g))
    rmse = float(np.sqrt(np.mean(diff * diff)))

    log_diff = np.log(p) - np.log(g)
    rmse_log = float(np.sqrt(np.mean(log_diff * log_diff)))
    # SI log（Eigen 等惯例：100 * sqrt(mean(d^2) - mean(d)^2)）
    var_d = float(np.mean(log_diff * log_diff) - np.mean(log_diff) ** 2)
    si_log = float(100.0 * np.sqrt(max(var_d, 0.0)))

    ratio = np.maximum(p / g, g / p)
    delta1 = float(np.mean(ratio < 1.25))
    delta2 = float(np.mean(ratio < 1.25 ** 2))
    delta3 = float(np.mean(ratio < 1.25 ** 3))

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "si_log": si_log,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_root",
        type=str,
        default="test_results",
        help="run_depth.py 输出目录，下含 <scene>/preds/*.npy",
    )
    parser.add_argument("--config", type=str, default="config_liosam.yaml")
    parser.add_argument("--dataset_root", type=str, default="/home/zs/Research3/dataset")
    parser.add_argument("--scenes", type=str, nargs="+", default=["22", "23"])
    parser.add_argument(
        "--output",
        type=str,
        default="test_results/metrics.json",
        help="结果文件输出路径（同时打印到终端）",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    ds_cfg = config["dataset"]
    center_crop_hw = tuple(int(v) for v in ds_cfg["center_crop_hw"])
    depth_key = ds_cfg.get("depth_key", DEFAULT_DEPTH_KEY)
    depth_mask_key = ds_cfg.get("depth_mask_key", DEFAULT_DEPTH_MASK_KEY)
    depth_range = ds_cfg.get("depth_range", {})
    min_depth = float(depth_range.get("min", 0.5))
    max_depth = float(depth_range.get("max", 80.0))

    pred_root = Path(args.pred_root)

    keys = ["abs_rel", "sq_rel", "rmse", "rmse_log", "si_log", "delta1", "delta2", "delta3"]
    sums: Dict[str, float] = {k: 0.0 for k in keys}
    count = 0
    missing_npy = 0
    skipped_no_valid = 0
    per_scene: Dict[str, Dict[str, float]] = {}

    for scene in args.scenes:
        scene_path = Path(args.dataset_root) / scene
        if not scene_path.is_dir():
            print(f"[WARN] 场景目录不存在: {scene_path}，跳过")
            continue
        entries = load_liosam_index(scene_path)
        scene_pred_dir = pred_root / scene / "preds"
        if not scene_pred_dir.is_dir():
            print(f"[WARN] 预测目录不存在: {scene_pred_dir}，跳过")
            continue

        scene_sums: Dict[str, float] = {k: 0.0 for k in keys}
        scene_count = 0
        for frame_id, _, fname in entries:
            stem = f"{frame_id:06d}"
            npy_path = scene_pred_dir / f"{stem}.npy"
            if not npy_path.exists():
                missing_npy += 1
                continue
            pred_m = np.load(str(npy_path)).astype(np.float32)
            gt_m, mask = _load_gt_and_mask(
                scene_path / fname,
                depth_key=depth_key,
                depth_mask_key=depth_mask_key,
                center_crop_hw=center_crop_hw,
                min_depth=min_depth,
                max_depth=max_depth,
            )
            if pred_m.shape != gt_m.shape:
                print(
                    f"[WARN] 形状不一致 {npy_path}: pred={pred_m.shape}, gt={gt_m.shape}，跳过"
                )
                continue

            m = _compute_metrics(pred_m, gt_m, mask)
            if m is None:
                skipped_no_valid += 1
                continue
            for k in keys:
                sums[k] += m[k]
                scene_sums[k] += m[k]
            count += 1
            scene_count += 1

        if scene_count > 0:
            per_scene[scene] = {k: scene_sums[k] / scene_count for k in keys}
            per_scene[scene]["num_frames"] = scene_count
            print(f"\n[SCENE {scene}] 有效帧 {scene_count}")
            for k in keys:
                print(f"  {k:<10s} = {per_scene[scene][k]:.6f}")
        else:
            print(f"[SCENE {scene}] 没有可评估的帧")

    if count == 0:
        print("\n[ERROR] 没有任何可评估的帧。请先运行 run_depth.py 生成预测。")
        return

    overall = {k: sums[k] / count for k in keys}
    print("\n========== 全部场景平均（每帧均值） ==========")
    for k in keys:
        print(f"  {k:<10s} = {overall[k]:.6f}")
    print(f"  num_frames = {count}, missing_npy = {missing_npy}, skipped_no_valid = {skipped_no_valid}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall": {**overall, "num_frames": count},
                "per_scene": per_scene,
                "missing_npy": missing_npy,
                "skipped_no_valid": skipped_no_valid,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
